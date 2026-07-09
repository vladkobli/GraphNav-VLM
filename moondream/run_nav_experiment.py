import argparse
import hashlib
import json
import math
import re
from textwrap import dedent
import time
from collections import deque, Counter
from pathlib import Path
from collections import deque, Counter, defaultdict

from ollama import chat


STATE_START = "START"
STATE_EXPLORING = "EXPLORING"
STATE_STOP = "STOP"
STATE_CANCELLED = "CANCELLED"

ACTION_MOVE = "MOVE"
ACTION_STOP = "STOP_TARGET_REACHED"
ACTION_CANCEL = "CANCELLED"

STOP_CONFIDENCE_THRESHOLD = 0.75
MAX_INVALID_STOP_ATTEMPTS = 2
MAX_REJECTED_MOVE_STREAK = 3

QWEN_TASK_PRIOR_PROMPT_VERSION = "qwen_task_prior_v1"
QWEN_PLANNER_PROMPT_VERSION = "qwen_visual_nav_planner_v1"
QWEN_TARGET_REACHED_PROMPT_VERSION = "qwen_target_reached_checker_v1"
MOONDREAM_PROMPT_VERSION_DEFAULT = "moondream_scene_description_v1"

QWEN_TASK_PRIOR_NUM_PREDICT = 220
QWEN_PLANNER_NUM_PREDICT = 700
QWEN_TARGET_REACHED_NUM_PREDICT = 180

LOW_EVIDENCE_BUDGET = 10

HIGH_EVIDENCE_BASES = {
    "direct_visual_evidence",
    "contextual_visual_evidence",
    "navigation_affordance",
}

VISUALLY_JUSTIFIED_BASES = {
    "direct_visual_evidence",
    "contextual_visual_evidence",
    "navigation_affordance",
}

MODEL_DECISION_MODES = {
    "llm_rank_unvisited_neighbors",
    "visual_policy",
}

BRANCH_MEMORY_WINDOW = 8
BRANCH_AVOID_BAD_SCORE = 3
BRANCH_DEAD_END_BAD_SCORE = 6
BRANCH_PROMISING_SCORE = 2

def is_low_evidence_decision(decision):
    return decision.get("decision_basis", "uncertain") not in HIGH_EVIDENCE_BASES


def normalize_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def angle_deg(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def direction_label(relative_bearing_deg):
    a = normalize_deg(relative_bearing_deg)

    sectors = [
        ("front", 0),
        ("front_left", 45),
        ("left", 90),
        ("back_left", 135),
        ("back", 180),
        ("back_right", -135),
        ("right", -90),
        ("front_right", -45),
    ]

    best_name = min(
        sectors,
        key=lambda x: abs(normalize_deg(a - x[1]))
    )[0]

    return best_name


def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = {node["id"]: node for node in data["nodes"]}
    return data, nodes


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def graph_edge_count(nodes):
    edges = set()

    for node_id, node in nodes.items():
        for nb in node.get("neighbors", []):
            edges.add(tuple(sorted((node_id, nb))))

    return len(edges)


def graph_directed_edge_count(nodes):
    return sum(len(node.get("neighbors", [])) for node in nodes.values())


def graph_dataset_version(graph_data):
    meta = graph_data.get("meta", {})
    return (
        meta.get("dataset_version")
        or meta.get("version")
        or meta.get("name")
        or "unknown"
    )


def graph_moondream_prompt_version(graph_data):
    meta = graph_data.get("meta", {})
    return (
        meta.get("moondream_prompt_version")
        or meta.get("description_prompt_version")
        or MOONDREAM_PROMPT_VERSION_DEFAULT
    )


def node_pose(node):
    pose = node["pose"]
    return {
        "x": pose["x"],
        "y": pose["y"],
        "z": pose.get("z", 0.0),
        "yaw_deg": pose["yaw_deg"],
    }


def get_image_description(image_entry):
    desc = image_entry.get("description", "")

    if desc:
        return desc.strip()

    sem = image_entry.get("semantic_description", {})
    if sem:
        return (
            f"Scene: {sem.get('scene', '')}\n"
            f"Visible objects: {sem.get('visible_objects', '')}\n"
            f"Distinctive landmarks: {sem.get('distinctive_landmarks', '')}\n"
            f"Navigation cues: {sem.get('navigation_cues', '')}"
        ).strip()

    return ""


def node_semantic_summary(node, max_chars=900):
    if node.get("summary", "").strip():
        return node["summary"].strip()

    parts = []

    for img in sorted(node.get("images", []), key=lambda x: x.get("stop_index", 999)):
        desc = get_image_description(img)
        if not desc:
            continue

        first_line = desc.split("\n")[0].strip()
        camera_id = img.get("camera_id", "unknown")
        parts.append(f"{camera_id}: {first_line}")

    summary = " | ".join(parts)

    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."

    return summary if summary else "No semantic description available."


def current_observations(node, max_desc_chars=700):
    observations = []

    for img in sorted(node.get("images", []), key=lambda x: x.get("stop_index", 999)):
        desc = get_image_description(img)
        if not desc:
            desc = "No description available."

        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars] + "..."

        observations.append({
            "image_id": img.get("image_id"),
            "camera_id": img.get("camera_id"),
            "stop_name": img.get("stop_name"),
            "relative_yaw_deg": img.get("relative_yaw_deg"),
            "heading_abs_deg": img.get("heading_abs_deg"),
            "description": desc,
        })

    return observations


def closest_current_view_to_bearing(node, relative_bearing_deg):
    images = node.get("images", [])

    if not images:
        return None

    best = None
    best_err = 999.0

    for img in images:
        rel_yaw = img.get("relative_yaw_deg")
        if rel_yaw is None:
            continue

        err = abs(normalize_deg(relative_bearing_deg - rel_yaw))

        if err < best_err:
            best_err = err
            best = img

    if best is None:
        return None

    sem = best.get("semantic_description", {})
    desc = get_image_description(best)

    return {
        "camera_id": best.get("camera_id"),
        "relative_yaw_deg": best.get("relative_yaw_deg"),
        "heading_abs_deg": best.get("heading_abs_deg"),
        "angular_error_deg": round(best_err, 1),

        "scene": sem.get("scene", ""),
        "scene_type": sem.get("scene_type", "unknown"),
        "navigation_cues": sem.get("navigation_cues", ""),
        "traversable_openings": sem.get("traversable_openings", ""),
        "visible_objects": sem.get("visible_objects", ""),
        "distinctive_landmarks": sem.get("distinctive_landmarks", ""),

        "description": desc if desc else "No description available.",
    }
    

def current_views_toward_bearing(node, relative_bearing_deg, top_k=2):
    views = []

    for img in node.get("images", []):
        rel_yaw = img.get("relative_yaw_deg")
        if rel_yaw is None:
            continue

        err = abs(normalize_deg(relative_bearing_deg - rel_yaw))
        desc = get_image_description(img)
        sem = img.get("semantic_description", {})

        views.append({
            "camera_id": img.get("camera_id"),
            "relative_yaw_deg": rel_yaw,
            "heading_abs_deg": img.get("heading_abs_deg"),
            "angular_error_deg": round(err, 1),

            "scene": sem.get("scene", ""),
            "scene_type": sem.get("scene_type", "unknown"),
            "navigation_cues": sem.get("navigation_cues", ""),
            "traversable_openings": sem.get("traversable_openings", ""),
            "visible_objects": sem.get("visible_objects", ""),
            "distinctive_landmarks": sem.get("distinctive_landmarks", ""),

            "description": desc if desc else "No description available.",
        })

    views = sorted(views, key=lambda v: v["angular_error_deg"])
    return views[:top_k]


def build_candidate_neighbors(current_node, nodes, visited_counts, memory_mode, allowed_neighbor_ids=None):
    candidates = []

    cur_pose = current_node["pose"]
    cur_x = cur_pose["x"]
    cur_y = cur_pose["y"]
    cur_yaw = cur_pose["yaw_deg"]

    neighbors_to_show = current_node.get("neighbors", [])

    if allowed_neighbor_ids is not None:
        allowed_neighbor_ids = set(allowed_neighbor_ids)
        neighbors_to_show = [n for n in neighbors_to_show if n in allowed_neighbor_ids]

    for nb_id in neighbors_to_show:
        nb = nodes[nb_id]
        nb_pose = nb["pose"]

        dx = nb_pose["x"] - cur_x
        dy = nb_pose["y"] - cur_y

        global_bearing = angle_deg(dx, dy)
        relative_bearing = normalize_deg(global_bearing - cur_yaw)
        distance = math.hypot(dx, dy)

        visited_count = visited_counts[nb_id]

        if memory_mode == "oracle" or visited_count > 0:
            neighbor_summary = node_semantic_summary(nb)
        else:
            neighbor_summary = "unknown_unvisited_node"

        views_toward_neighbor = current_views_toward_bearing(
            current_node,
            relative_bearing,
            top_k=3
        )
        candidates.append({
            "node_id": nb_id,
            "distance_m": round(distance, 2),
            "bearing_from_current_deg": round(relative_bearing, 1),
            "direction": direction_label(relative_bearing),
            "visited_count": visited_count,
            "neighbor_summary": neighbor_summary,
            "views_toward_neighbor": views_toward_neighbor,
        })

    return candidates


def shortest_path(nodes, start, goals):
    goals = set(goals)
    q = deque([(start, [start])])
    seen = {start}

    while q:
        node_id, path = q.popleft()

        if node_id in goals:
            return path

        for nb in nodes[node_id].get("neighbors", []):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, path + [nb]))

    return None


def path_distance(nodes, path):
    total = 0.0

    for a, b in zip(path[:-1], path[1:]):
        pa = nodes[a]["pose"]
        pb = nodes[b]["pose"]
        total += math.hypot(pb["x"] - pa["x"], pb["y"] - pa["y"])

    return total

def is_recent_loop_choice(chosen, current, route, window=4):
    recent = route[-window:]
    return chosen in recent

def extract_json_object(text):
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response:\n{text}")

    return json.loads(match.group(0))


def normalize_for_match(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def clamp01(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default

    return max(0.0, min(1.0, value))


def build_evidence_source_text(observations, candidates, visited_memory):
    parts = []

    for obs in observations:
        parts.append(obs.get("description", ""))

    for c in candidates:
        audit = c.get("visual_audit", {})

        for view in c.get("views_toward_neighbor", []):
            parts.append(view.get("description", ""))
            parts.append(view.get("scene", ""))
            parts.append(view.get("scene_type", ""))
            parts.append(view.get("navigation_cues", ""))
            parts.append(view.get("traversable_openings", ""))
            parts.append(view.get("visible_objects", ""))
            parts.append(view.get("distinctive_landmarks", ""))

        parts.append(c.get("neighbor_summary", ""))
        parts.extend(str(v) for v in audit.values())

    for m in visited_memory:
        parts.append(m.get("semantic_summary", ""))

    return "\n".join(parts)


def sanitize_decision(decision, evidence_source_text):
    decision["confidence"] = clamp01(decision.get("confidence", 0.0))
    decision["stop_confidence"] = clamp01(decision.get("stop_confidence", 0.0))

    source_l = normalize_for_match(evidence_source_text)

    basis = decision.get("decision_basis", "uncertain")

    visual_quote = str(decision.get("visual_evidence_quote", "")).strip()
    affordance_quote = str(decision.get("affordance_quote", "")).strip()
    assumption = str(decision.get("commonsense_assumption", "")).strip()

    visual_quote_l = normalize_for_match(visual_quote)
    affordance_quote_l = normalize_for_match(affordance_quote)

    visual_quote_is_valid = (
        visual_quote_l
        and visual_quote_l != "none"
        and visual_quote_l in source_l
    )

    affordance_quote_is_valid = (
        affordance_quote_l
        and affordance_quote_l != "none"
        and affordance_quote_l in source_l
    )

    speculative_words = [
        "could lead",
        "might contain",
        "likely to have",
        "probably",
        "could potentially",
        "might be",
        "likely place",
    ]

    reason_l = normalize_for_match(decision.get("reason", ""))
    assumption_l = normalize_for_match(assumption)

    # Direct/contextual visual evidence needs visual_evidence_quote.
    if basis in ["direct_visual_evidence", "contextual_visual_evidence"]:
        if not visual_quote_is_valid:
            if assumption and assumption.lower() != "none":
                decision["decision_basis"] = "commonsense_prior"
                decision["visual_evidence_quote"] = "none"
                decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.5)
                decision["reason"] = (
                    "Downgraded from visual evidence to commonsense prior because "
                    "the visual evidence quote was missing or not found."
                )
            else:
                decision["decision_basis"] = "exploration"
                decision["visual_evidence_quote"] = "none"
                decision["commonsense_assumption"] = "none"
                decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.3)
                decision["reason"] = (
                    "Downgraded to exploration because no supporting visual quote was found."
                )

    # Navigation affordance needs affordance_quote.
    if basis == "navigation_affordance":
        if not affordance_quote_is_valid:
            decision["decision_basis"] = "exploration"
            decision["affordance_quote"] = "none"
            decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.3)
            decision["reason"] = (
                "Downgraded to exploration because no supporting affordance quote was found."
            )

    # Speculation should not be counted as visual evidence.
    if any(w in reason_l or w in assumption_l for w in speculative_words):
        decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.4)

        if decision.get("decision_basis") in [
            "direct_visual_evidence",
            "contextual_visual_evidence",
            "navigation_affordance",
        ]:
            decision["decision_basis"] = "commonsense_prior"
            decision["visual_evidence_quote"] = "none"


    bad_affordance_text = normalize_for_match(
        f"{decision.get('affordance_quote', '')} "
        f"{decision.get('reason', '')} "
        f"{decision.get('navigation_affordance', '')}"
    )

    bad_affordance_phrases = [
        "closed door",
        "closed doorway",
        "closed doors",
        "no clear traversable",
        "no clear path",
        "no clear navigation",
    ]

    if decision.get("decision_basis") == "navigation_affordance":
        if any(p in bad_affordance_text for p in bad_affordance_phrases):
            decision["navigation_affordance"] = "blocked_or_unclear"
            decision["decision_basis"] = "insufficient_visual_evidence"
            decision["move_is_visually_justified"] = False
            decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.2)
            decision["reason"] = (
                "Downgraded because a closed door or unclear path is not a confirmed traversable route."
            )
        
    if decision.get("decision_basis") in VISUALLY_JUSTIFIED_BASES:
        decision["move_is_visually_justified"] = True
    else:
        decision["move_is_visually_justified"] = False
        
    return decision


def short_description(desc, max_chars=260):
    desc = desc.replace("\n", " | ").strip()
    if len(desc) > max_chars:
        return desc[:max_chars] + "..."
    return desc


def make_task_analysis_schema():
    return {
        "type": "object",
        "properties": {
            "target_environment": {
                "type": "string",
                "enum": ["indoor", "outdoor", "either", "unknown"]
            },
            "target_type": {
                "type": "string",
                "enum": ["object", "place", "area", "room", "building", "route", "unknown"]
            },
            "environment_reason": {
                "type": "string"
            }
        },
        "required": [
            "target_environment",
            "target_type",
            "environment_reason"
        ]
    }


def analyze_task_environment(model, task):
    prompt = f"""
Analyze the navigation task for a mobile robot.

Task:
{task}

Decide the most likely target environment.

Rules:
- indoor: target is likely inside a building, room, hallway, lab, classroom, office, lobby, etc.
- outdoor: target is likely outside, in a courtyard, street, garden, exterior walkway, parking area, etc.
- either: target can reasonably be indoors or outdoors.
- unknown: the task does not give enough information.
- Do not overfit to one object; reason generally from the task wording.

Return only JSON.
""".strip()

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You classify robot navigation tasks. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.0,
            "num_predict": 120,
        },
        format=make_task_analysis_schema(),
    )

    if isinstance(response, dict):
        content = response["message"]["content"]
    else:
        content = response.message.content

    return extract_json_object(content)


def make_task_prior_schema():
    return {
        "type": "object",
        "properties": {
            "target_environment": {
                "type": "string",
                "enum": ["indoor", "outdoor", "either", "unknown"]
            },
            "target_place_type": {
                "type": "string",
                "enum": [
                    "room",
                    "corridor",
                    "public_area",
                    "kitchen",
                    "office",
                    "classroom",
                    "bathroom",
                    "entrance_exit",
                    "object_location",
                    "unknown"
                ]
            },
            "target_synonyms": {
                "type": "array",
                "items": {"type": "string"}
            },
            "likely_contexts": {
                "type": "array",
                "items": {"type": "string"}
            },
            "unlikely_contexts": {
                "type": "array",
                "items": {"type": "string"}
            },
            "useful_visual_cues": {
                "type": "array",
                "items": {"type": "string"}
            },
            "search_strategy": {
                "type": "string"
            }
        },
        "required": [
            "target_environment",
            "target_place_type",
            "target_synonyms",
            "likely_contexts",
            "unlikely_contexts",
            "useful_visual_cues",
            "search_strategy"
        ]
    }


def analyze_task_prior(model, task):
    prompt = f"""
Analyze this mobile robot navigation task.

Task:
{task}

Infer general commonsense search priors.

Examples:
- bathrooms are often near corridors, public indoor areas, signs, doors, sinks, tiled rooms.
- classrooms are usually indoors, near corridors, doors, lecture rooms, desks, chairs, boards.
- coffee may be near kitchens, vending machines, cafeterias, counters, indoor common areas.
- whiteboards are usually indoors, in classrooms, meeting rooms, labs, or offices.
- apples may be in kitchens, cafeterias, dining areas, tables, counters, or food areas.
- For "Find a blackboard": target_synonyms should include blackboard/chalkboard/board; likely_contexts should include classroom/lecture hall/teaching room.
- For "Find a sink": target_synonyms should include sink/washbasin/faucet; likely_contexts should include bathroom/kitchen/lab/utility room.
- For "Find a coffee machine": target_synonyms should include coffee machine/coffee maker/espresso machine; likely_contexts should include kitchen/break room/cafeteria/common area.

Rules:
- Do not assume the target is visible.
- Do not use any map-specific knowledge.
- These are only commonsense priors, not visual evidence.
- If the task asks for an object, keep object names in target_synonyms.
- Put rooms or areas where the object is usually found in likely_contexts, not in target_synonyms.
- Keep lists short and general.

Return only JSON.
""".strip()

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You infer general task priors for robot navigation. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.0,
            "num_predict": QWEN_TASK_PRIOR_NUM_PREDICT,
        },
        format=make_task_prior_schema(),
    )

    if isinstance(response, dict):
        content = response["message"]["content"]
    else:
        content = response.message.content

    return extract_json_object(content)


def add_unique(items, new_items):
    result = list(items or [])
    seen = {str(x).lower() for x in result}

    for item in new_items:
        key = str(item).lower()
        if key not in seen:
            result.append(item)
            seen.add(key)

    return result


PLACE_TARGET_TERMS = [
    "classroom", "lecture room", "lecture hall", "seminar room",
    "bathroom", "restroom", "toilet", "wc", "kitchen", "office",
    "lobby", "corridor", "hallway", "entrance", "exit", "outside",
    "outdoor", "yard", "parking", "street", "sidewalk"
]

GENERIC_PLACE_TERMS = set(PLACE_TARGET_TERMS + [
    "room", "rooms", "public area", "public indoor area", "indoor public area",
    "meeting room", "common area", "cafeteria", "dining area", "lab",
    "laboratory", "utility room", "copy room", "reception", "pantry",
    "building", "entrance area", "door", "doors", "sign", "signs",
])

OBJECT_CONTAINER_PRIORS = [
    {
        "terms": ["blackboard", "chalkboard", "whiteboard", "board"],
        "synonyms": ["blackboard", "chalkboard", "whiteboard", "writing board", "board"],
        "likely_contexts": [
            "classroom", "lecture hall", "lecture room", "seminar room",
            "teaching room", "meeting room", "auditorium"
        ],
        "cues": [
            "podium", "lectern", "desks", "chairs", "auditorium seating",
            "classroom door", "lecture hall", "chalk tray", "projector"
        ],
        "unlikely_contexts": [
            "outdoor courtyard", "parking area", "street", "garden",
            "bathroom", "toilet stall", "kitchen"
        ],
        "strategy": (
            "first find a classroom, lecture hall, seminar room, or meeting room; "
            "then look for the board itself"
        ),
    },
    {
        "terms": ["sink", "washbasin", "wash basin", "faucet", "tap"],
        "synonyms": ["sink", "washbasin", "wash basin", "basin", "faucet", "tap"],
        "likely_contexts": [
            "bathroom", "restroom", "toilet", "kitchen", "lab",
            "laboratory", "utility room"
        ],
        "cues": [
            "tiles", "mirror", "soap dispenser", "countertop", "pipes",
            "toilet", "radiator", "water fixture"
        ],
        "unlikely_contexts": [
            "outdoor courtyard", "parking area", "street", "lecture hall"
        ],
        "strategy": (
            "first find a bathroom, kitchen, lab, or utility room; "
            "then inspect that room for the sink or faucet"
        ),
    },
    {
        "terms": ["coffee machine", "coffee maker", "espresso machine", "coffee"],
        "synonyms": ["coffee machine", "coffee maker", "espresso machine", "coffee dispenser"],
        "likely_contexts": [
            "kitchen", "break room", "common area", "office pantry",
            "cafeteria", "lobby", "public indoor area"
        ],
        "cues": [
            "counter", "cups", "trash cans", "vending machine", "sink",
            "posters", "tables", "chairs"
        ],
        "unlikely_contexts": [
            "outdoor courtyard", "parking area", "street", "bathroom",
            "classroom podium"
        ],
        "strategy": (
            "first find a kitchen, break room, cafeteria, lobby, or common area; "
            "then look for a coffee machine"
        ),
    },
    {
        "terms": ["printer", "copier", "copy machine", "scanner"],
        "synonyms": ["printer", "copier", "copy machine", "scanner", "multifunction printer"],
        "likely_contexts": ["office", "copy room", "reception", "library", "lab", "hallway"],
        "cues": ["desk", "paper", "shelves", "office equipment", "counter", "sign"],
        "unlikely_contexts": ["outdoor courtyard", "parking area", "bathroom"],
        "strategy": "first find an office, copy room, reception, library, or lab; then look for the machine",
    },
    {
        "terms": ["microwave", "fridge", "refrigerator", "water cooler"],
        "synonyms": ["microwave", "fridge", "refrigerator", "water cooler"],
        "likely_contexts": ["kitchen", "break room", "cafeteria", "common area", "office pantry"],
        "cues": ["counter", "cups", "tables", "chairs", "sink", "vending machine"],
        "unlikely_contexts": ["outdoor courtyard", "parking area", "lecture hall"],
        "strategy": "first find a kitchen, break room, cafeteria, or common area; then look for the appliance",
    },
    {
        "terms": ["car", "vehicle", "van", "truck", "bicycle", "bike", "motorcycle"],
        "synonyms": ["car", "vehicle", "van", "truck", "bicycle", "bike", "motorcycle"],
        "target_environment": "outdoor",
        "likely_contexts": ["parking area", "street", "driveway", "garage", "outdoor entrance", "road"],
        "cues": ["parked cars", "parking lot", "curb", "road", "street", "sidewalk", "gate"],
        "unlikely_contexts": ["classroom", "bathroom", "lecture hall", "office", "kitchen"],
        "strategy": "first find an outdoor parking, street, driveway, or garage area; then look for the vehicle",
    },
]


def extract_requested_target(task):
    task_l = normalize_for_match(task).strip()
    task_l = re.sub(r"^(please\s+)?", "", task_l)

    patterns = [
        r"^(find|locate|look for|search for|go to|navigate to)\s+(a|an|the|any)?\s*(.+)$",
        r"^(where is|where are)\s+(a|an|the|any)?\s*(.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, task_l)
        if match:
            target = match.group(match.lastindex or 0)
            target = re.split(r"\s+(in|inside|near|beside|next to|through|via)\s+", target)[0]
            target = re.sub(r"[^a-z0-9 _/-]+$", "", target).strip()
            target = re.sub(r"^(nearest|closest)\s+", "", target).strip()
            return target or ""

    return ""


def is_place_target(task_l, requested_target):
    text = normalize_for_match(" ".join([task_l, requested_target or ""]))
    return any(term_matches_task_text(term, text) for term in PLACE_TARGET_TERMS)


def find_object_container_prior(task_l, requested_target):
    text = normalize_for_match(" ".join([task_l, requested_target or ""]))

    for object_prior in OBJECT_CONTAINER_PRIORS:
        if any(term_matches_task_text(term, text) for term in object_prior["terms"]):
            return object_prior

    return None


def term_matches_task_text(term, text):
    term = normalize_for_match(term)
    text = normalize_for_match(text)
    if not term:
        return False

    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def clean_object_synonyms(existing_synonyms, object_synonyms, requested_target):
    object_terms = [
        normalize_for_match(x)
        for x in add_unique(object_synonyms, [requested_target] if requested_target else [])
        if normalize_for_match(x)
    ]
    object_term_set = set(object_terms)
    cleaned = []

    for synonym in existing_synonyms or []:
        key = normalize_for_match(str(synonym))
        if not key or key in GENERIC_PLACE_TERMS:
            continue
        if key in object_term_set or any(key in term or term in key for term in object_terms):
            cleaned.append(str(synonym))

    return add_unique(cleaned, object_terms)


def apply_object_container_prior(prior, requested_target, object_prior=None):
    likely = prior.get("likely_contexts", [])
    unlikely = prior.get("unlikely_contexts", [])
    cues = prior.get("useful_visual_cues", [])
    synonyms = prior.get("target_synonyms", [])

    prior["target_environment"] = object_prior.get("target_environment", "indoor") if object_prior else "indoor"
    prior["target_place_type"] = "object_location"

    if object_prior:
        synonyms = clean_object_synonyms(
            synonyms,
            object_prior.get("synonyms", []),
            requested_target,
        )
        likely = add_unique(likely, object_prior.get("likely_contexts", []))
        unlikely = add_unique(unlikely, object_prior.get("unlikely_contexts", []))
        cues = add_unique(cues, object_prior.get("cues", []))
        prior["search_strategy"] = object_prior.get("strategy", prior.get("search_strategy", ""))
        prior["target_object"] = requested_target or object_prior.get("synonyms", ["object"])[0]
        prior["container_place_priors"] = object_prior.get("likely_contexts", [])
    else:
        synonyms = clean_object_synonyms(synonyms, [requested_target], requested_target)
        likely = add_unique(likely, [
            "indoor room", "hallway", "corridor", "common area",
            "lobby", "public indoor area", "room sign"
        ])
        unlikely = add_unique(unlikely, [
            "outdoor courtyard", "parking area", "street", "garden", "grass"
        ])
        cues = add_unique(cues, [
            requested_target, "open door", "room sign", "counter",
            "table", "shelf", "desk"
        ])
        prior["search_strategy"] = (
            f"first find a plausible room or public indoor area for {requested_target}; "
            f"then look for {requested_target} itself"
        )
        prior["target_object"] = requested_target
        prior["container_place_priors"] = likely[:8]

    prior["target_synonyms"] = synonyms
    prior["likely_contexts"] = likely
    prior["unlikely_contexts"] = unlikely
    prior["useful_visual_cues"] = cues

    return prior


def enrich_task_prior(task, prior):
    task_l = task.lower()
    prior = dict(prior or {})
    requested_target = extract_requested_target(task)
    object_prior = find_object_container_prior(task_l, requested_target)

    likely = prior.get("likely_contexts", [])
    unlikely = prior.get("unlikely_contexts", [])
    cues = prior.get("useful_visual_cues", [])
    synonyms = prior.get("target_synonyms", [])

    if "classroom" in task_l or "lecture room" in task_l:
        prior["target_environment"] = "indoor"
        prior["target_place_type"] = "classroom"

        synonyms = add_unique(synonyms, [
            "classroom", "lecture hall", "lecture room", "seminar room",
            "teaching room", "study room", "auditorium", "auditorium seating"
        ])

        likely = add_unique(likely, [
            "corridor", "hallway", "indoor public area",
            "classroom", "lecture hall", "room with desks",
            "room with chairs", "auditorium", "auditorium seating",
            "blackboard", "whiteboard"
        ])

        unlikely = add_unique(unlikely, [
            "outdoor courtyard", "parking area", "street",
            "garden", "exterior path", "grass", "trees"
        ])

        cues = add_unique(cues, [
            "desks", "chairs", "whiteboard", "blackboard",
            "projector", "podium", "lecture hall", "auditorium",
            "auditorium seating", "classroom door", "room sign"
        ])

    if "bathroom" in task_l or "restroom" in task_l or "toilet" in task_l:
        prior["target_environment"] = "indoor"
        prior["target_place_type"] = "bathroom"

        synonyms = add_unique(synonyms, [
            "bathroom", "restroom", "toilet", "wc"
        ])

        likely = add_unique(likely, [
            "corridor", "hallway", "public indoor area",
            "door", "sign", "tiled room", "sink area"
        ])

        unlikely = add_unique(unlikely, [
            "outdoor courtyard", "parking area", "street",
            "garden", "exterior path"
        ])

        cues = add_unique(cues, [
            "bathroom sign", "restroom sign", "toilet sign",
            "sink", "tiles", "washbasin", "door label"
        ])

    if "vending" in task_l:
        prior["target_environment"] = "indoor"
        prior["target_place_type"] = "object_location"

        synonyms = add_unique(synonyms, [
            "vending machine", "snack machine", "drink machine"
        ])

        likely = add_unique(likely, [
            "hallway", "corridor", "lobby", "public indoor area",
            "common area", "near entrance"
        ])

        unlikely = add_unique(unlikely, [
            "outdoor courtyard", "parking area", "garden", "grass"
        ])

        cues = add_unique(cues, [
            "vending machine", "drink machine", "snack machine",
            "bright machine", "public hallway"
        ])

    outdoor_exit_terms = [
        "outside", "outdoor", "exit", "yard", "barrier",
        "gate", "fence", "street", "sidewalk", "parking"
    ]

    if any(term in task_l for term in outdoor_exit_terms):
        prior["target_environment"] = "outdoor"
        prior["target_place_type"] = "entrance_exit"

        likely = add_unique(likely, [
            "exit", "glass door", "entrance", "outdoor view",
            "courtyard", "exterior path", "sky", "gate",
            "fence", "barrier", "street", "sidewalk",
            "parking area", "yard boundary"
        ])

        unlikely = add_unique(unlikely, [
            "classroom", "lecture hall", "closed room", "interior wall"
        ])

        cues = add_unique(cues, [
            "exit sign", "glass doors", "open door", "outdoor area",
            "courtyard", "sky", "exterior", "gate", "fence",
            "barrier", "concrete barrier", "street", "sidewalk",
            "parking lot", "parked cars", "yard boundary"
        ])

        synonyms = add_unique(synonyms, [
            "outside", "outdoor area", "exit", "gate", "fence",
            "barrier", "street", "sidewalk", "parking lot",
            "yard boundary"
        ])

    if object_prior:
        prior["target_synonyms"] = synonyms
        prior["likely_contexts"] = likely
        prior["unlikely_contexts"] = unlikely
        prior["useful_visual_cues"] = cues
        prior = apply_object_container_prior(prior, requested_target, object_prior)
        synonyms = prior.get("target_synonyms", [])
        likely = prior.get("likely_contexts", [])
        unlikely = prior.get("unlikely_contexts", [])
        cues = prior.get("useful_visual_cues", [])
    elif (
        requested_target
        and any(verb in task_l for verb in ["find", "locate", "look for", "search for"])
        and not is_place_target(task_l, requested_target)
    ):
        prior["target_synonyms"] = synonyms
        prior["likely_contexts"] = likely
        prior["unlikely_contexts"] = unlikely
        prior["useful_visual_cues"] = cues
        prior = apply_object_container_prior(prior, requested_target)
        synonyms = prior.get("target_synonyms", [])
        likely = prior.get("likely_contexts", [])
        unlikely = prior.get("unlikely_contexts", [])
        cues = prior.get("useful_visual_cues", [])

    prior["target_synonyms"] = synonyms
    prior["likely_contexts"] = likely
    prior["unlikely_contexts"] = unlikely
    prior["useful_visual_cues"] = cues

    return prior


def build_planner_prompt(
    task,
    current_node,
    observations,
    candidates,
    visited_memory,
    trajectory,
    max_steps,
    step_idx,
    task_prior=None,
    navigation_memory=None,
):
    compact_observations = []

    for obs in observations:
        compact_observations.append({
            "camera": obs.get("camera_id"),
            "relative_yaw_deg": obs.get("relative_yaw_deg"),
            "heading_abs_deg": obs.get("heading_abs_deg"),
            "description": short_description(obs.get("description", ""), 260),
        })

    compact_candidates = []
    
    for c in candidates:
        views = c.get("views_toward_neighbor", [])

        compact_candidates.append({
            "node_id": c["node_id"],
            "direction": c["direction"],
            "bearing_from_current_deg": c["bearing_from_current_deg"],
            "distance_m": c["distance_m"],
            "visited_count": c["visited_count"],
            "views_toward_neighbor": [
                {
                    "camera": view.get("camera_id"),
                    "relative_yaw_deg": view.get("relative_yaw_deg"),
                    "angular_error_deg": view.get("angular_error_deg"),

                    "scene": short_description(view.get("scene", ""), 120),
                    "scene_type": view.get("scene_type", "unknown"),
                    "navigation_cues": short_description(view.get("navigation_cues", ""), 160),
                    "traversable_openings": short_description(view.get("traversable_openings", ""), 160),
                    "visible_objects": short_description(view.get("visible_objects", ""), 120),
                }
                for view in views
            ],
            "neighbor_summary_if_known": short_description(c.get("neighbor_summary", ""), 250),
            "visual_audit": c.get("visual_audit", {
                "scene_environment": "unknown",
                "target_environment_match": c.get("task_prior_alignment", "unknown"),
                "edge_affordance": "unknown",
                "open_route_quote": "none",
                "blocked_quote": "none",
                "target_cue": "none",
                "target_cue_quote": "none",
                "visual_score": 0,
                "selection_score": 0,
                "should_consider": True,
            }),
            "edge_visit_count": c.get("edge_visit_count", 0),
            "edge_bad_count": c.get("edge_bad_count", 0),
            "node_bad_count": c.get("node_bad_count", 0),
            "is_dead_end_memory": c.get("is_dead_end_memory", False),
            "task_prior_alignment": c.get("task_prior_alignment", "unknown"),
            "branch_memory_status": c.get("branch_memory_status", "untried"),
            "branch_good_score": c.get("branch_good_score", 0),
            "branch_bad_score": c.get("branch_bad_score", 0),
            "branch_memory_summary": c.get("branch_memory_summary", "No branch memory yet."),
        })

    compact_memory = []

    for m in visited_memory[-8:]:
        compact_memory.append({
            "node_id": m["node_id"],
            "visited_count": m["visited_count"],
            "summary": short_description(m.get("semantic_summary", ""), 250),
        })

    payload = {
        "task": task,
        "task_prior": task_prior or {},
        "navigation_memory": navigation_memory or {},
        "current_node_id": current_node["id"],
        "step": step_idx,
        "max_steps": max_steps,
        "trajectory_so_far": trajectory,
        "visited_memory": compact_memory,
        "current_observations": compact_observations,
        "candidate_neighbors": compact_candidates,
        "allowed_next_node_ids": [c["node_id"] for c in compact_candidates],
    }

    planner_rules = """
Decision rules:
- Choose exactly one node from allowed_next_node_ids.
- These are the only nodes you are allowed to choose.
- You are choosing the next feasible movement for a real mobile robot.
- Do not rely on exhaustive graph search, DFS, or future backtracking.
- Choose a candidate only if the current visual information supports moving that way.
- If no candidate is visually justified, choose the least-bad candidate but mark move_is_visually_justified=false.
- You will be evaluated on visual grounding, not just eventual success.
- First inspect each candidate's visual_audit.
- Prefer high selection_score candidates. This score combines visual evidence, target context, branch memory, and loop avoidance.
- Prefer high visual_score candidates with edge_affordance=open/unclear and target_environment_match=supports_task_prior or direct_target_hint.
- A candidate with edge_affordance=closed is unavailable unless it has direct target evidence and no better movement exists.
- Do not treat "closed door" as a useful classroom/indoor cue. It is blocked evidence.
- A generic outdoor clear path is only useful for an indoor target if it visibly leads to an open entrance, open doorway, hallway, corridor, lobby, or ramp into the building.

Action rules:
- First decide whether the robot has already reached the target at the current node.
- If the target is clearly visible in current_observations, choose action STOP_TARGET_REACHED.
- STOP_TARGET_REACHED means the robot should stop because it believes the task is complete.
- Use STOP_TARGET_REACHED only from current_observations, not from candidate_neighbors, not from neighbor summaries, and not from task_prior.
- Do not stop only because the target might be nearby, might be behind a door, or might be in the next node.
- If the target is not clearly visible or explicitly described at the current node, choose MOVE.
- If the robot is looping, all useful branches are avoid/dead_end, or there is no safe useful movement left, choose CANCELLED.
- For MOVE, next_node_id must be one of allowed_next_node_ids.
- For STOP_TARGET_REACHED or CANCELLED, next_node_id must be "none".

Grounding rules:
- The task describes what the robot is searching for.
- The task itself is not visual evidence.
- Use only the provided current observations, candidate view descriptions, known neighbor summaries, and visited memory.
- Do not invent objects, rooms, signs, doors, paths, labels, or destinations.
- Do not guess what is behind a door, around a corner, inside a building, or farther down a path.
- Do not claim visual evidence unless you can quote it from the provided descriptions.

Stop rules:
- stop_evidence_quote must be copied only from current_observations.
- Candidate neighbor descriptions cannot justify stopping.
- Visited memory cannot justify stopping.
- Task prior cannot justify stopping.
- If stopping, decision_basis should usually be direct_visual_evidence.
- For object targets, stop only if the object or a close synonym is visible.
- For place/environment targets, stop only if the current scene explicitly matches the requested place/environment.

Navigation affordance:
- Prefer candidates whose view contains a clear traversable route, passage, corridor, doorway, entrance, hallway, path, walkway, stairs, ramp, or opening.
- Do not treat a generic building facade, wall, closed doors, courtyard, trees, benches, or parked objects as strong evidence by themselves.
- Treat closed doors and closed doorways as unavailable routes, not as promising entrances.
- Treat "could potentially traverse" or "possible entrance" as unclear unless an open path is explicitly described.
- If two candidates are semantically similar, prefer the one with a clearer traversable transition in the direction of travel.
- If a candidate only shows a large exterior area with no clear transition, mark it as exploration unless it directly matches the task.
- If the task is probably indoor, and a node shows a clear entrance, doorway, corridor, or transition to an indoor space, mark it as a strong candidate rather than continue outdoor exploration.
- If the task is probably outdoor, and a node shows a clear exit, doorway, corridor, or transition to an outdoor space, mark it as a strong candidate rather than continue indoor exploration.

Decision basis:
- direct_visual_evidence:
  The requested target, destination, object, place, sign, label, or very close synonym is explicitly visible or named in the descriptions.

- contextual_visual_evidence:
  The target is not directly visible, but the visible scene itself is concretely related to the task.
  This must still be supported by a quote from the descriptions.

- commonsense_prior:
  The descriptions do not contain concrete task evidence, but general world knowledge makes one option more plausible.
  Example: choosing a building entrance for an indoor object, or choosing an exterior path for an outdoor destination.
  This is allowed, but it must not be presented as visual evidence.

- exploration:
  There is no useful visual evidence and no strong commonsense preference, so choose an unvisited node.

- uncertain:
  The choice is weak or ambiguous.
  
- visual_evidence_quote and affordance_quote must be short exact phrases, maximum 12 words.
- Do not copy the full description.
- The reason must be one short sentence.

Direct visual evidence:
- Use direct_visual_evidence only when the target itself is visible or explicitly named.
- For environment tasks such as "go indoors" or "go outside", direct evidence requires the candidate view to explicitly show the target environment type, such as Scene type: indoor, Scene type: outdoor, or Scene type: transition.
- A walkway, paved area, building, wall, door, or courtyard alone is not direct evidence.

Task-prior reasoning:
- Use task_prior as commonsense guidance only.
- task_prior is not visual evidence.
- Prefer candidates whose scene_type, visible objects, traversable openings, or navigation cues match likely_contexts or useful_visual_cues.
- Avoid candidates whose visible scene strongly matches unlikely_contexts, unless they contain direct visual evidence of the target.
- If the target is probably indoor, outdoor-only candidates should be weak unless they show an entrance, doorway, corridor, or transition.
- If the target is probably outdoor, indoor-only candidates should be weak unless they show an exit, outdoor view, doorway, or transition.
- For object_location tasks, treat target_synonyms as the specific object being searched for.
- For object_location tasks, treat likely_contexts/container_place_priors as likely rooms or areas for routing toward the object.
- For object_location tasks, first move toward the best visually supported likely room or area, then stop only when the object itself is visible.
- Do not treat the likely room as direct target evidence unless the task asks for that room.
- For hidden targets, such as bathrooms, classrooms, kitchens, or coffee, reason about plausible intermediate places, but mark this as commonsense_prior, not direct_visual_evidence.
- For indoor targets such as classrooms, bathrooms, whiteboards, coffee, or vending machines, outdoor branches are useful only if they visibly lead to an indoor entrance or corridor.
- A path going outside is not aligned with finding a classroom unless there is direct evidence of classroom-like objects or a building entrance.

Branch-memory reasoning:
- Use branch_memory_status to avoid repeating bad branches.
- dead_end means this branch was already explored and produced no useful progress; avoid it unless direct target evidence is visible.
- avoid means this branch repeatedly produced unsupported, contradictory, or looping moves; choose another candidate if available.
- promising means previous movement in that branch produced useful visual or task-aligned evidence.
- untried means there is no evidence yet.
- Do not choose an avoid/dead_end branch only because it is traversable.
- For hidden targets, prefer branches that remain consistent with task_prior over time.
- In branch_memory_reasoning, explain whether this is a new, promising, neutral, avoid, or dead-end branch.

Quotes:
- visual_evidence_quote must be copied from the provided descriptions.
- If there is no visual evidence, write "none".
- commonsense_assumption should explain any world-knowledge assumption, or "none".
- Do not use speculative phrases like "could lead to", "might contain", "likely to have", or "probably" as visual evidence.

Confidence:
- 0.8 to 1.0: direct visual evidence.
- 0.5 to 0.7: contextual visual evidence.
- 0.3 to 0.5: commonsense prior.
- 0.1 to 0.3: exploration.
- 0.0 to 0.1: very uncertain.
- confidence must be between 0.0 and 1.0. Do not copy visual_score or selection_score into confidence.
""".strip()

    example_output = {
        "action": "MOVE/STOP_TARGET_REACHED/CANCELLED",
        "next_node_id": "valid neighbor id, or none",
        "target_reached": False,
        "stop_confidence": 0.0,
        "stop_evidence_quote": "exact quote from current_observations, or none",
        "stop_reason": "why the robot should stop, or none",
        "cancellation_reason": "why the robot cancelled, or none",
        "confidence": 0.0,
        "decision_basis": "direct_visual_evidence/contextual_visual_evidence/navigation_affordance/commonsense_prior/insufficient_visual_evidence/exploration/backtrack/uncertain",
        "move_is_visually_justified": False,
        "navigation_affordance": "clear_transition/open_path/weak_path/blocked_or_unclear/unknown",
        "visual_evidence_quote": "exact quote from descriptions, or none",
        "affordance_quote": "exact quote about path/door/entrance/corridor/walkway, or none",
        "commonsense_assumption": "general assumption used, or none",
        "target_context_reasoning": "why this direction fits the likely target context",
        "branch_memory_reasoning": "how previous branch memory affected this choice",
        "reason": "short grounded reason"
    }

    return dedent(f"""
    Choose the next node for a mobile robot.

    Task:
    {task}

    {planner_rules}

    Input:
    {json.dumps(payload, indent=2, ensure_ascii=False)}

    Return only this JSON object:
    {json.dumps(example_output, indent=2, ensure_ascii=False)}
    """).strip()
        
        
def make_decision_schema(valid_neighbors):
    valid_next_ids = list(valid_neighbors) + ["none"]

    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [ACTION_MOVE]
            },
            "next_node_id": {
                "type": "string",
                "enum": valid_next_ids
            },
            "target_reached": {
                "type": "boolean"
            },
            "stop_confidence": {
                "type": "number"
            },
            "stop_evidence_quote": {
                "type": "string"
            },
            "stop_reason": {
                "type": "string"
            },
            "cancellation_reason": {
                "type": "string"
            },
            "confidence": {
                "type": "number"
            },
            "decision_basis": {
                "type": "string",
                "enum": [
                    "direct_visual_evidence",
                    "contextual_visual_evidence",
                    "navigation_affordance",
                    "commonsense_prior",
                    "insufficient_visual_evidence",
                    "exploration",
                    "backtrack",
                    "uncertain"
                ]
            },
            "navigation_affordance": {
                "type": "string",
                "enum": [
                    "clear_transition",
                    "open_path",
                    "weak_path",
                    "blocked_or_unclear",
                    "unknown"
                ]
            },
            "affordance_quote": {
                "type": "string"
            },
            "visual_evidence_quote": {
                "type": "string"
            },
            "commonsense_assumption": {
                "type": "string"
            },
            "target_context_reasoning": {
                "type": "string"
            },
            "branch_memory_reasoning": {
                "type": "string"
            },
            "reason": {
                "type": "string"
            },
            "move_is_visually_justified": {
                "type": "boolean"
            }
        },
        "required": [
            "action",
            "next_node_id",
            "target_reached",
            "stop_confidence",
            "stop_evidence_quote",
            "stop_reason",
            "cancellation_reason",
            "confidence",
            "decision_basis",
            "navigation_affordance",
            "affordance_quote",
            "visual_evidence_quote",
            "commonsense_assumption",
            "target_context_reasoning",
            "branch_memory_reasoning",
            "reason",
            "move_is_visually_justified"
        ]
    }
    
    
def term_in_text(term, text):
    term = str(term).lower().strip()
    if not term:
        return False

    variants = {
        term,
        term.rstrip("s"),
        term.replace("_", " "),
    }

    return any(v and v in text for v in variants)


def any_term_in_text(terms, text):
    return any(term_in_text(term, text) for term in terms)


def target_visibility_at_node(node, task_prior):
    observations = current_observations(node)
    evidence_text = "\n".join(
        str(obs.get("description", ""))
        for obs in observations
    )
    evidence_l = normalize_for_match(evidence_text)

    terms = []
    terms.extend(str(x) for x in (task_prior or {}).get("target_synonyms", []))

    target_object = (task_prior or {}).get("target_object")
    if target_object:
        terms.append(str(target_object))

    matched_terms = [
        term for term in terms
        if term_in_text(term, evidence_l)
    ]

    return {
        "visible": bool(matched_terms),
        "matched_terms": sorted(set(matched_terms)),
        "evidence_quote": (
            first_matching_quote(evidence_text, matched_terms)
            if matched_terms else "none"
        ),
    }


OPEN_ROUTE_PHRASES = [
    "open doorway",
    "open door",
    "open passage",
    "open entrance",
    "clear path",
    "clear route",
    "clear corridor",
    "clear hallway",
    "can walk through",
    "walk through",
    "can follow",
    "path through",
    "through or around",
    "ramp leading",
    "covered entrance",
]

WEAK_ROUTE_PHRASES = [
    "hallway",
    "corridor",
    "walkway",
    "path",
    "ramp",
    "stairs",
    "passage",
]

CLOSED_ROUTE_PHRASES = [
    "closed door",
    "closed doorway",
    "closed doors",
    "closed wooden door",
    "closed entrance",
    "only see a closed door",
    "can only see a closed door",
    "only traverse a closed doorway",
]

NO_ROUTE_PHRASES = [
    "no visible traversable",
    "no visible routes",
    "no visible traversable openings",
    "no visible traversable routes",
    "no clear open passage",
    "no clear visible route",
    "no clear traversable",
    "no clear path",
]


OUTDOOR_BOUNDARY_TERMS = [
    "barrier",
    "concrete barrier",
    "gate",
    "fence",
    "street",
    "sidewalk",
    "parking lot",
    "parking area",
    "parked car",
    "parked cars",
    "car",
    "cars",
    "outside",
    "outdoor",
    "exterior",
    "yard",
    "boundary",
]

STRONG_OUTDOOR_BOUNDARY_TERMS = [
    "barrier",
    "concrete barrier",
    "gate",
    "street",
    "sidewalk",
    "parking lot",
    "parking area",
]


def candidate_visual_text(candidate):
    parts = []

    for view in candidate.get("views_toward_neighbor", []):
        parts.extend([
            view.get("scene", ""),
            view.get("scene_type", ""),
            view.get("navigation_cues", ""),
            view.get("traversable_openings", ""),
            view.get("visible_objects", ""),
            view.get("distinctive_landmarks", ""),
            view.get("description", ""),
        ])

    return "\n".join(str(p) for p in parts if p)


def first_matching_quote(text, phrases, max_words=12):
    text = str(text)
    text_l = text.lower()

    for phrase in phrases:
        idx = text_l.find(phrase)
        if idx < 0:
            continue

        end = min(len(text), idx + len(phrase))
        words = text[idx:end].strip().split()
        return " ".join(words[:max_words])

    return "none"


def matching_terms(text, terms):
    text_l = normalize_for_match(text)
    return [term for term in terms if term in text_l]


def scene_environment_from_views(candidate):
    scene_types = {
        str(v.get("scene_type", "unknown")).lower()
        for v in candidate.get("views_toward_neighbor", [])
        if v.get("scene_type")
    }

    if "transition" in scene_types:
        return "transition"
    if "indoor" in scene_types and "outdoor" in scene_types:
        return "mixed"
    if "indoor" in scene_types:
        return "indoor"
    if "outdoor" in scene_types:
        return "outdoor"
    return "unknown"


def closest_view_text(candidate):
    views = candidate.get("views_toward_neighbor", [])
    if not views:
        return ""

    view = min(
        views,
        key=lambda v: float(v.get("angular_error_deg", 999.0) or 999.0)
    )

    return " ".join([
        str(view.get("scene", "")),
        str(view.get("scene_type", "")),
        str(view.get("navigation_cues", "")),
        str(view.get("traversable_openings", "")),
        str(view.get("visible_objects", "")),
        str(view.get("distinctive_landmarks", "")),
    ])


def phrase_has_negating_prefix(text_l, start_idx):
    prefix = text_l[max(0, start_idx - 45):start_idx]
    return re.search(
        r"(no|not|without|cannot|can't|only)\s+[^.;,\n]{0,35}$",
        prefix,
    ) is not None


def positive_phrase_in_text(phrases, text_l):
    for phrase in phrases:
        phrase_l = normalize_for_match(phrase)
        if not phrase_l:
            continue

        for match in re.finditer(re.escape(phrase_l), text_l):
            if not phrase_has_negating_prefix(text_l, match.start()):
                return True

    return False


def classify_edge_affordance(text):
    text_l = normalize_for_match(text)

    has_open = positive_phrase_in_text(OPEN_ROUTE_PHRASES, text_l)
    has_weak_route = any(p in text_l for p in WEAK_ROUTE_PHRASES)
    has_closed = any(p in text_l for p in CLOSED_ROUTE_PHRASES)
    has_no_route = any(p in text_l for p in NO_ROUTE_PHRASES)

    if has_open:
        return "open"

    if has_weak_route:
        return "unclear"

    if has_closed or has_no_route:
        return "closed"

    return "unknown"


def audit_candidate_visual(candidate, task_prior):
    text = candidate_visual_text(candidate)
    text_l = normalize_for_match(text)
    closest_l = normalize_for_match(closest_view_text(candidate))
    alignment = candidate.get("task_prior_alignment", "unknown")
    edge_affordance = classify_edge_affordance(text)
    target_env = (task_prior or {}).get("target_environment", "unknown")
    target_place_type = (task_prior or {}).get("target_place_type", "unknown")
    closest_outdoor_boundary_terms = matching_terms(closest_l, OUTDOOR_BOUNDARY_TERMS)
    closest_strong_boundary_terms = matching_terms(closest_l, STRONG_OUTDOOR_BOUNDARY_TERMS)
    has_outdoor_boundary_cue = bool(closest_outdoor_boundary_terms)

    if (
        target_env == "outdoor"
        and target_place_type == "entrance_exit"
        and has_outdoor_boundary_cue
        and edge_affordance == "closed"
    ):
        edge_affordance = "unclear"

    target_synonyms = [
        str(x).lower()
        for x in (task_prior or {}).get("target_synonyms", [])
    ]
    container_place_priors = [
        str(x).lower()
        for x in (task_prior or {}).get("container_place_priors", [])
    ]
    likely_contexts = [
        str(x).lower()
        for x in (task_prior or {}).get("likely_contexts", [])
    ]
    useful_cues = [
        str(x).lower()
        for x in (task_prior or {}).get("useful_visual_cues", [])
    ]

    object_container_contexts = container_place_priors or likely_contexts
    has_direct_target = any_term_in_text(target_synonyms, text_l)
    has_container_context = any_term_in_text(object_container_contexts, text_l)
    has_likely_context = any_term_in_text(likely_contexts, text_l)
    has_useful_cue = any_term_in_text(useful_cues, text_l)
    context_strength = "none"

    if has_direct_target:
        target_cue = "direct"
        target_quote = first_matching_quote(text, target_synonyms)
        context_strength = "target"
    elif target_place_type == "object_location" and has_container_context:
        target_cue = "contextual"
        target_quote = first_matching_quote(text, object_container_contexts)
        context_strength = "container_place"
    elif target_place_type != "object_location" and has_likely_context:
        target_cue = "contextual"
        target_quote = first_matching_quote(text, likely_contexts)
        context_strength = "container_place"
    elif has_likely_context or has_useful_cue:
        target_cue = "contextual"
        target_quote = first_matching_quote(text, likely_contexts + useful_cues)
        context_strength = "supporting_cue"
    else:
        target_cue = "none"
        target_quote = "none"

    open_quote = first_matching_quote(text, OPEN_ROUTE_PHRASES + WEAK_ROUTE_PHRASES)
    blocked_quote = first_matching_quote(text, CLOSED_ROUTE_PHRASES + NO_ROUTE_PHRASES)

    if edge_affordance == "open":
        blocked_quote = "none"
    elif edge_affordance == "closed":
        open_quote = "none"

    score = 0
    if target_cue == "direct":
        score += 4
    elif context_strength == "container_place":
        score += 3
    elif context_strength == "supporting_cue":
        score += 2

    if edge_affordance == "open":
        score += 3
    elif edge_affordance == "unclear":
        score += 0
    elif edge_affordance == "closed":
        score -= 4

    if alignment == "direct_target_hint":
        score += 4
    elif alignment == "supports_task_prior":
        if target_place_type == "object_location":
            if context_strength == "container_place":
                score += 4
            elif context_strength == "supporting_cue":
                score += 1
            else:
                score += 2
        else:
            score += 4
    elif alignment == "contradicts_task_prior":
        score -= 6

    outdoor_distractors = [
        "parked car", "parked cars", "parking", "fence", "gate",
        "trees", "park", "courtyard", "street", "garden"
    ]
    indoor_transition_terms = [
        "covered entrance", "open doorway", "open door", "hallway",
        "corridor", "lobby", "ramp", "walk through"
    ]

    if target_env == "indoor":
        if (
            any(p in closest_l for p in outdoor_distractors)
            and not any(p in closest_l for p in indoor_transition_terms)
            and target_cue != "direct"
        ):
            score -= 4

    if target_env == "outdoor":
        if (
            any(p in closest_l for p in ["classroom", "lecture hall", "office", "interior wall"])
            and not any(p in closest_l for p in ["exit", "outside", "outdoor", "courtyard"])
            and target_cue != "direct"
        ):
            score -= 4

        if target_place_type == "entrance_exit" and has_outdoor_boundary_cue:
            score += min(6, 2 + len(closest_outdoor_boundary_terms))

            if closest_strong_boundary_terms:
                score += 2

            if (
                "covered entrance" in normalize_for_match(open_quote)
                and not closest_strong_boundary_terms
            ):
                score -= 3

    return {
        "scene_environment": scene_environment_from_views(candidate),
        "target_environment_match": alignment,
        "edge_affordance": edge_affordance,
        "open_route_quote": open_quote,
        "blocked_quote": blocked_quote,
        "target_cue": target_cue,
        "context_strength": context_strength,
        "target_cue_quote": target_quote,
        "outdoor_boundary_cues": closest_outdoor_boundary_terms[:6],
        "strong_outdoor_boundary_cues": closest_strong_boundary_terms[:4],
        "visual_score": score,
        "should_consider": edge_affordance != "closed" or target_cue == "direct",
    }


def candidate_is_closed_by_audit(candidate):
    audit = candidate.get("visual_audit", {})
    return (
        audit.get("edge_affordance") == "closed"
        and audit.get("target_cue") != "direct"
    )


def add_candidate_selection_scores(candidates, recent_route):
    recent_route = list(recent_route or [])
    very_recent = set(recent_route[-4:])

    for candidate in candidates:
        audit = candidate.get("visual_audit", {})
        score = float(audit.get("visual_score", 0))

        visited_count = int(candidate.get("visited_count", 0) or 0)
        edge_visit_count = int(candidate.get("edge_visit_count", 0) or 0)
        node_bad_count = int(candidate.get("node_bad_count", 0) or 0)
        edge_bad_count = int(candidate.get("edge_bad_count", 0) or 0)
        node_id = candidate.get("node_id")
        branch = candidate.get("branch_memory_status", "untried")

        if visited_count == 0:
            score += 3
        else:
            score -= min(8, 3 * visited_count)

        if node_id in very_recent:
            score -= 5

        if edge_visit_count > 0:
            score -= 6

        score -= min(6, 2 * edge_bad_count + node_bad_count)

        if branch == "promising":
            score += 2
        elif branch == "avoid":
            score -= 5
        elif branch == "dead_end":
            score -= 8

        audit["selection_score"] = round(score, 2)
        candidate["visual_audit"] = audit

    return candidates


def candidate_selection_score(candidate):
    return float(candidate.get("visual_audit", {}).get("selection_score", 0.0))


def compact_candidate_audits(candidates):
    audits = []

    for candidate in candidates:
        audit = candidate.get("visual_audit", {})
        audits.append({
            "node_id": candidate.get("node_id"),
            "direction": candidate.get("direction"),
            "visited_count": candidate.get("visited_count", 0),
            "task_prior_alignment": candidate.get("task_prior_alignment", "unknown"),
            "edge_affordance": audit.get("edge_affordance", "unknown"),
            "target_cue": audit.get("target_cue", "none"),
            "context_strength": audit.get("context_strength", "none"),
            "outdoor_boundary_cues": audit.get("outdoor_boundary_cues", []),
            "strong_outdoor_boundary_cues": audit.get("strong_outdoor_boundary_cues", []),
            "open_route_quote": audit.get("open_route_quote", "none"),
            "blocked_quote": audit.get("blocked_quote", "none"),
            "visual_score": audit.get("visual_score", 0),
            "selection_score": audit.get("selection_score", 0),
        })

    return audits


def candidate_prior_alignment(candidate, task_prior):
    if not task_prior:
        return "unknown"

    target_env = task_prior.get("target_environment", "unknown")
    views = candidate.get("views_toward_neighbor", [])

    scene_types = {
        str(v.get("scene_type", "unknown")).lower()
        for v in views
        if v.get("scene_type")
    }

    text = " ".join(
        [
            str(v.get("scene", "")) + " " +
            str(v.get("scene_type", "")) + " " +
            str(v.get("navigation_cues", "")) + " " +
            str(v.get("traversable_openings", "")) + " " +
            str(v.get("visible_objects", "")) + " " +
            str(v.get("distinctive_landmarks", ""))
            for v in views
        ]
    ).lower()

    likely_contexts = [str(x).lower() for x in task_prior.get("likely_contexts", [])]
    unlikely_contexts = [str(x).lower() for x in task_prior.get("unlikely_contexts", [])]
    useful_cues = [str(x).lower() for x in task_prior.get("useful_visual_cues", [])]
    target_synonyms = [str(x).lower() for x in task_prior.get("target_synonyms", [])]

    has_target = any_term_in_text(target_synonyms, text)
    has_likely_context = any_term_in_text(likely_contexts, text)
    has_unlikely_context = any_term_in_text(unlikely_contexts, text)
    has_useful_cue = any_term_in_text(useful_cues, text)

    indoor_terms = [
        "indoor", "hallway", "corridor", "room", "classroom",
        "lecture hall", "office", "lab", "lobby"
    ]

    outdoor_terms = [
        "outdoor", "outside", "courtyard", "parking", "street",
        "garden", "grass", "trees", "exterior", "sky"
    ]

    transition_to_indoor_terms = [
        "entrance", "doorway", "open door", "glass door",
        "hallway", "corridor", "lobby"
    ]

    transition_to_outdoor_terms = [
        "exit", "outside", "outdoor", "courtyard", "exterior", "sky"
    ]

    if has_target:
        return "direct_target_hint"

    if target_env == "indoor":
        if has_unlikely_context and not has_likely_context and not has_useful_cue:
            return "contradicts_task_prior"

        if "outdoor" in scene_types and not any_term_in_text(transition_to_indoor_terms, text):
            return "contradicts_task_prior"

        if "transition" in scene_types:
            if any_term_in_text(transition_to_outdoor_terms, text) and not any_term_in_text(transition_to_indoor_terms, text):
                return "contradicts_task_prior"
            if any_term_in_text(transition_to_indoor_terms, text):
                return "supports_task_prior"
            return "neutral"

        if "indoor" in scene_types:
            return "supports_task_prior"

    if target_env == "outdoor":
        if has_unlikely_context and not has_likely_context and not has_useful_cue:
            return "contradicts_task_prior"

        if "indoor" in scene_types and not any_term_in_text(transition_to_outdoor_terms, text):
            return "contradicts_task_prior"

        if "transition" in scene_types:
            if any_term_in_text(transition_to_outdoor_terms, text):
                return "supports_task_prior"
            return "neutral"

        if "outdoor" in scene_types:
            return "supports_task_prior"

    if has_useful_cue or has_likely_context:
        return "supports_task_prior"

    if has_unlikely_context:
        return "contradicts_task_prior"

    return "neutral"


def apply_candidate_constraints(decision, chosen, candidate_by_id):
    candidate = candidate_by_id.get(chosen)

    if not candidate:
        return decision

    alignment = candidate.get("task_prior_alignment", "unknown")
    audit = candidate.get("visual_audit", {})

    if audit.get("edge_affordance") == "closed":
        decision["navigation_affordance"] = "blocked_or_unclear"
        decision["decision_basis"] = "insufficient_visual_evidence"
        decision["move_is_visually_justified"] = False
        decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.15)
        decision["reason"] = (
            "Rejected as visually unsupported because the candidate audit marks the route closed."
        )

    if alignment == "contradicts_task_prior":
        if decision.get("decision_basis") != "direct_visual_evidence":
            decision["decision_basis"] = "insufficient_visual_evidence"
            decision["move_is_visually_justified"] = False
            decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.2)
            decision["reason"] = (
                "Rejected as visually unsupported because the candidate contradicts the task prior "
                "and does not contain direct target evidence."
            )

    return decision


def call_ollama_planner(model, prompt, valid_neighbors, temperature=0.0, num_predict=QWEN_PLANNER_NUM_PREDICT):
    schema = make_decision_schema(valid_neighbors)

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a robot graph-navigation planner. "
                    "You must choose exactly one valid neighbor node. "
                    "Return only the requested JSON object."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
        format=schema,
    )

    if isinstance(response, dict):
        return response["message"]["content"]

    return response.message.content


def choose_fallback_neighbor(current_node, visited_counts, previous_node=None):
    neighbors = current_node.get("neighbors", [])

    if not neighbors:
        return None

    unvisited = [n for n in neighbors if visited_counts[n] == 0]

    if unvisited:
        # Prefer unvisited, but avoid simply taking the first one every time.
        return unvisited[-1]

    if previous_node and len(neighbors) > 1:
        non_backtrack = [n for n in neighbors if n != previous_node]
        if non_backtrack:
            return min(non_backtrack, key=lambda n: visited_counts[n])

    return min(neighbors, key=lambda n: visited_counts[n])


def make_visited_memory(nodes, visited_counts, max_items=12):
    visited = [
        node_id for node_id, count in visited_counts.items()
        if count > 0
    ]

    memory = []

    for node_id in visited[-max_items:]:
        node = nodes[node_id]
        memory.append({
            "node_id": node_id,
            "visited_count": visited_counts[node_id],
            "pose": node_pose(node),
            "semantic_summary": node_semantic_summary(node),
        })

    return memory


def should_reprompt_for_loop(choice, current_node, visited_counts):
    neighbors = current_node.get("neighbors", [])

    if choice not in neighbors:
        return True

    unvisited_neighbors = [n for n in neighbors if visited_counts[n] == 0]

    if unvisited_neighbors and visited_counts[choice] >= 2:
        return True

    return False

def edge_key(a, b):
    return f"{a}->{b}"


def make_branch_record():
    return {
        "visits": 0,
        "supported": 0,
        "unsupported": 0,
        "contradictions": 0,
        "direct_hits": 0,
        "loops": 0,
        "dead_end_marks": 0,
        "last_reason": "",
        "last_step": None,
    }


def branch_good_score(record):
    return (
        record.get("supported", 0)
        + 3 * record.get("direct_hits", 0)
    )


def branch_bad_score(record):
    return (
        record.get("unsupported", 0)
        + 2 * record.get("contradictions", 0)
        + 2 * record.get("loops", 0)
        + 2 * record.get("dead_end_marks", 0)
    )


def branch_status(record):
    if not record or record.get("visits", 0) == 0:
        return "untried"

    good = branch_good_score(record)
    bad = branch_bad_score(record)

    if bad >= BRANCH_DEAD_END_BAD_SCORE and good == 0:
        return "dead_end"

    if bad >= BRANCH_AVOID_BAD_SCORE and good == 0:
        return "avoid"

    if good >= BRANCH_PROMISING_SCORE:
        return "promising"

    return "neutral"


def branch_summary(record):
    if not record or record.get("visits", 0) == 0:
        return "No branch memory yet."

    return (
        f"visits={record.get('visits', 0)}, "
        f"supported={record.get('supported', 0)}, "
        f"unsupported={record.get('unsupported', 0)}, "
        f"contradictions={record.get('contradictions', 0)}, "
        f"direct_hits={record.get('direct_hits', 0)}, "
        f"loops={record.get('loops', 0)}, "
        f"status={branch_status(record)}, "
        f"last_reason={record.get('last_reason', '')}"
    )


def get_branch_record(branch_memory, key):
    if key not in branch_memory:
        branch_memory[key] = make_branch_record()
    return branch_memory[key]


def update_branch_memory_for_edge(
    branch_memory,
    edge,
    decision,
    candidate,
    step,
    reached_goal=False,
    loop_rejected=False,
):
    record = get_branch_record(branch_memory, edge)

    record["visits"] += 1
    record["last_step"] = step

    basis = decision.get("decision_basis", "unknown")
    visual = decision.get("move_is_visually_justified") is True
    alignment = candidate.get("task_prior_alignment", "unknown") if candidate else "unknown"

    if reached_goal:
        record["direct_hits"] += 1
        record["supported"] += 2
        record["last_reason"] = "This branch reached the goal."
        return

    if loop_rejected:
        record["loops"] += 1
        record["last_reason"] = "This branch caused a repeated-edge loop."
        return

    if alignment == "contradicts_task_prior":
        record["contradictions"] += 1

    if visual:
        record["supported"] += 1
        record["last_reason"] = decision.get("reason", "Visually supported move.")
    else:
        record["unsupported"] += 1
        record["last_reason"] = decision.get("reason", "Unsupported move.")

    if basis == "direct_visual_evidence":
        record["direct_hits"] += 1


def punish_recent_branches(branch_memory, recent_branch_edges, reason, step, amount=1):
    for edge in list(recent_branch_edges):
        record = get_branch_record(branch_memory, edge)
        record["unsupported"] += amount
        record["last_reason"] = reason
        record["last_step"] = step


def reward_recent_branches(branch_memory, recent_branch_edges, reason, step, amount=1):
    for edge in list(recent_branch_edges):
        record = get_branch_record(branch_memory, edge)
        record["supported"] += amount
        record["last_reason"] = reason
        record["last_step"] = step


def build_branch_memory_payload(branch_memory, max_items=12):
    items = []

    for edge, record in branch_memory.items():
        status = branch_status(record)

        if status in {"avoid", "dead_end", "promising"}:
            items.append({
                "edge": edge,
                "status": status,
                "good_score": branch_good_score(record),
                "bad_score": branch_bad_score(record),
                "summary": branch_summary(record),
            })

    items = sorted(
        items,
        key=lambda x: (
            0 if x["status"] == "dead_end" else
            1 if x["status"] == "avoid" else
            2
        )
    )

    return items[:max_items]


def build_current_evidence_text(observations):
    return "\n".join(
        str(obs.get("description", ""))
        for obs in observations
    )


def validate_stop_decision(decision, current_evidence_text, min_confidence=STOP_CONFIDENCE_THRESHOLD):
    action = decision.get("action", ACTION_MOVE)

    if action != ACTION_STOP:
        return False, "Action is not STOP_TARGET_REACHED."

    if decision.get("target_reached") is not True:
        return False, "target_reached is not true."

    try:
        stop_conf = float(decision.get("stop_confidence", 0.0))
    except Exception:
        stop_conf = 0.0

    if stop_conf < min_confidence:
        return False, f"stop_confidence {stop_conf} is below threshold {min_confidence}."

    quote = str(decision.get("stop_evidence_quote", "")).strip()
    quote_l = normalize_for_match(quote)
    evidence_l = normalize_for_match(current_evidence_text)

    if not quote_l or quote_l == "none":
        return False, "Missing stop_evidence_quote."

    if quote_l not in evidence_l:
        return False, "stop_evidence_quote was not found in current observations."

    if decision.get("decision_basis") != "direct_visual_evidence":
        return False, "Stopping requires direct_visual_evidence."

    return True, "Valid stop decision."

def make_target_reached_schema():
    return {
        "type": "object",
        "properties": {
            "target_reached": {
                "type": "boolean"
            },
            "confidence": {
                "type": "number"
            },
            "evidence_quote": {
                "type": "string"
            },
            "matched_target": {
                "type": "string"
            },
            "reason": {
                "type": "string"
            }
        },
        "required": [
            "target_reached",
            "confidence",
            "evidence_quote",
            "matched_target",
            "reason"
        ]
    }


def build_target_reached_prompt(task, task_prior, current_node, observations):
    compact_observations = []

    for obs in observations:
        compact_observations.append({
            "camera": obs.get("camera_id"),
            "relative_yaw_deg": obs.get("relative_yaw_deg"),
            "description": short_description(obs.get("description", ""), 500),
        })

    payload = {
        "task": task,
        "task_prior": task_prior or {},
        "current_node_id": current_node["id"],
        "current_observations": compact_observations,
    }

    rules = """
You decide whether a mobile robot has already reached the target.

Rules:
- Use ONLY current_observations.
- Do NOT use candidate neighbors.
- Do NOT use visited memory.
- Do NOT use future possibilities.
- The task itself is not evidence.
- Stop only if the current scene clearly satisfies the task.
- For object targets, the object or a close synonym must be visible or explicitly named in the robot's current reachable area.
- For object targets, likely rooms or container places in task_prior are routing hints only; they are not enough to stop.
- Do not stop if the target is only visible through/inside another room, behind a closed door, or beyond an unavailable doorway.
- If the same observation mentions the target but also says closed door, closed doorway, no visible traversable route, or no open doorway, return target_reached=false unless the target is clearly in the same reachable space as the robot.
- For place targets, the current scene must explicitly match the requested place type.
- For outdoor/exit/barrier tasks, stop if the current reachable scene clearly shows the robot is outside or at an outdoor boundary/exit area, such as a gate, fence, barrier, street, sidewalk, parking lot, parked cars, exterior yard boundary, or open outdoor area.
- For "classroom", accept classroom, lecture hall, seminar room, teaching room, room with desks/chairs, chalkboard, blackboard, whiteboard, or podium if clearly described.
- For "vending machine", accept only vending machine, snack machine, drink machine, or a clearly described machine matching that object.
- Do not stop because the target might be nearby.
- Do not stop because a doorway/corridor could lead to the target.
- evidence_quote must be copied exactly from current_observations.
- If there is no exact evidence, return target_reached=false and evidence_quote="none".
""".strip()

    return dedent(f"""
    Task:
    {task}

    {rules}

    Input:
    {json.dumps(payload, indent=2, ensure_ascii=False)}

    Return only JSON:
    {{
      "target_reached": false,
      "confidence": 0.0,
      "evidence_quote": "exact quote from current_observations, or none",
      "matched_target": "what target was matched, or none",
      "reason": "short reason"
    }}
    """).strip()


def call_target_reached_checker(model, prompt, temperature=0.0, num_predict=QWEN_TARGET_REACHED_NUM_PREDICT):
    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a target-reached checker for a mobile robot. "
                    "Return only valid JSON."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": temperature,
            "num_predict": num_predict,
        },
        format=make_target_reached_schema(),
    )

    if isinstance(response, dict):
        return response["message"]["content"]

    return response.message.content


def validate_target_reached(stop_decision, current_observations, threshold=0.75):
    evidence_text = "\n".join(
        str(obs.get("description", ""))
        for obs in current_observations
    )

    evidence_l = normalize_for_match(evidence_text)
    quote = str(stop_decision.get("evidence_quote", "")).strip()
    quote_l = normalize_for_match(quote)

    try:
        confidence = float(stop_decision.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if stop_decision.get("target_reached") is not True:
        return False, "target_reached is false"

    if confidence < threshold:
        return False, f"confidence {confidence} below threshold {threshold}"

    if not quote_l or quote_l == "none":
        return False, "missing evidence_quote"

    if quote_l not in evidence_l:
        return False, "evidence_quote not found in current observations"

    return True, "valid target reached decision"


def make_cancel_result(
    current,
    step,
    total_distance,
    trajectory,
    decisions,
    visited_counts,
    invalid_decisions,
    fallback_used,
    branch_memory,
    reason,
    visited_goal_nodes=None,
):
    return {
        "success": False,
        "failure_reason": reason,
        "final_state": STATE_CANCELLED,
        "model_stopped": False,
        "stop_correct": False,
        "final_node": current,
        "steps": step,
        "total_distance_m": round(total_distance, 2),
        "trajectory": trajectory,
        "decisions": decisions,
        "visited_counts": dict(visited_counts),
        "invalid_decisions": invalid_decisions,
        "reprompts": 0,
        "fallback_used": fallback_used,
        "branch_memory": branch_memory,
        "visited_goal_nodes": sorted(visited_goal_nodes or []),
    }
    

def run_episode_dfs(nodes, model, task, start_node, goal_nodes, max_steps, memory_mode, temperature=0.0):
    current = start_node
    goal_nodes = set(goal_nodes)

    visited_counts = Counter()
    stack = [start_node]

    trajectory = []
    decisions = []

    invalid_decisions = 0
    fallback_used = 0
    total_distance = 0.0
    low_evidence_streak = 0

    for step in range(max_steps + 1):
        visited_counts[current] += 1

        if current in goal_nodes:
            return {
                "success": True,
                "final_node": current,
                "steps": step,
                "total_distance_m": round(total_distance, 2),
                "trajectory": trajectory,
                "decisions": decisions,
                "visited_counts": dict(visited_counts),
                "invalid_decisions": invalid_decisions,
                "reprompts": 0,
                "fallback_used": fallback_used,
            }

        if step == max_steps:
            break

        current_node = nodes[current]
        all_neighbors = current_node.get("neighbors", [])

        # True DFS behavior: only ask the LLM about never-visited neighbors.
        allowed_neighbors = [
            n for n in all_neighbors
            if visited_counts[n] == 0
        ]

        if low_evidence_streak >= LOW_EVIDENCE_BUDGET and not allowed_neighbors and len(stack) > 1:
            mode = "low_evidence_forced_backtrack"

            stack.pop()
            chosen = stack[-1]

            decision = {
                "next_node_id": chosen,
                "confidence": 1.0,
                "decision_basis": "backtrack",
                "navigation_affordance": "unknown",
                "visual_evidence_quote": "none",
                "affordance_quote": "none",
                "commonsense_assumption": "none",
                "reason": (
                    f"Forced backtrack after {LOW_EVIDENCE_BUDGET} consecutive "
                    "low-evidence decisions."
                ),
            }

            low_evidence_streak = 0

            d = path_distance(nodes, [current, chosen])
            total_distance += d

            step_log = {
                "step": step,
                "from": current,
                "to": chosen,
                "distance_m": round(d, 2),
                "mode": mode,
                "low_evidence_streak": low_evidence_streak,
                "decision": decision,
            }

            trajectory.append(chosen)
            decisions.append(step_log)

            print(
                f"[step {step}] {current} -> {chosen} | "
                f"{decision.get('decision_basis')} | conf={decision.get('confidence')} | "
                f"{decision.get('reason')}"
            )

            current = chosen
            continue
        
        if allowed_neighbors:
            mode = "llm_rank_unvisited_neighbors"

            observations = current_observations(current_node)

            candidates = build_candidate_neighbors(
                current_node=current_node,
                nodes=nodes,
                visited_counts=visited_counts,
                memory_mode=memory_mode,
                allowed_neighbor_ids=allowed_neighbors,
            )

            visited_memory = make_visited_memory(nodes, visited_counts)

            prompt = build_planner_prompt(
                task=task,
                current_node=current_node,
                observations=observations,
                candidates=candidates,
                visited_memory=visited_memory,
                trajectory=trajectory,
                max_steps=max_steps,
                step_idx=step,
            )

            raw = call_ollama_planner(
                model=model,
                prompt=prompt,
                valid_neighbors=allowed_neighbors,
                temperature=temperature,
            )

            evidence_source_text = build_evidence_source_text(
                observations=observations,
                candidates=candidates,
                visited_memory=visited_memory,
            )

            try:
                decision = extract_json_object(raw)
                decision = sanitize_decision(decision, evidence_source_text)
                chosen = decision.get("next_node_id")
            except Exception as e:
                invalid_decisions += 1
                chosen = allowed_neighbors[0]
                decision = {
                    "next_node_id": chosen,
                    "confidence": 0.0,
                    "decision_basis": "exploration",
                    "navigation_affordance": "unknown",
                    "visual_evidence_quote": "none",
                    "affordance_quote": "none",
                    "commonsense_assumption": "none",
                    "reason": f"Model returned invalid JSON. Chose first unvisited neighbor. Error: {e}",
                    "raw_response": raw,
                }
                fallback_used += 1

            if decision.get("action", ACTION_MOVE) == ACTION_MOVE and chosen not in allowed_neighbors:
                invalid_decisions += 1
                old_choice = chosen
                chosen = allowed_neighbors[0]
                decision["invalid_choice"] = old_choice
                decision["next_node_id"] = chosen
                decision["decision_basis"] = "exploration"
                decision["reason"] = (
                    f"Model chose an invalid or disallowed neighbor '{old_choice}'. "
                    f"Fallback selected '{chosen}'."
                )
                fallback_used += 1

            if is_low_evidence_decision(decision):
                low_evidence_streak += 1
            else:
                low_evidence_streak = 0
    
            stack.append(chosen)

        else:
            mode = "deterministic_backtrack"
            low_evidence_streak = 0
            
            if len(stack) <= 1:
                return {
                    "success": False,
                    "failure_reason": "all_reachable_explored",
                    "final_node": current,
                    "steps": step,
                    "total_distance_m": round(total_distance, 2),
                    "trajectory": trajectory,
                    "decisions": decisions,
                    "visited_counts": dict(visited_counts),
                    "invalid_decisions": invalid_decisions,
                    "reprompts": 0,
                    "fallback_used": fallback_used,
                }

            # Pop current node and go back to the previous node on the DFS stack.
            stack.pop()
            chosen = stack[-1]

            decision = {
                "next_node_id": chosen,
                "confidence": 1.0,
                "decision_basis": "backtrack",
                "navigation_affordance": "unknown",
                "visual_evidence_quote": "none",
                "affordance_quote": "none",
                "commonsense_assumption": "none",
                "reason": "No unvisited neighbors remain at the current node, so the controller backtracks along the DFS stack."
            }

        d = path_distance(nodes, [current, chosen])
        total_distance += d

        step_log = {
            "step": step,
            "from": current,
            "to": chosen,
            "distance_m": round(d, 2),
            "mode": mode,
            "state": STATE_EXPLORING,
            "action": ACTION_MOVE,
            "decision": decision,
            "executed": True,
            "low_evidence_streak": low_evidence_streak,
        }

        trajectory.append(chosen)
        decisions.append(step_log)

        print(
            f"[step {step}] {current} -> {chosen} | "
            f"{decision.get('decision_basis')} | conf={decision.get('confidence')} | "
            f"{decision.get('reason')}"
        )

        current = chosen
        rejected_move_streak = 0
        state = STATE_EXPLORING

    return {
        "success": current in goal_nodes,
        "failure_reason": "max_steps",
        "final_node": current,
        "steps": max_steps,
        "total_distance_m": round(total_distance, 2),
        "trajectory": trajectory,
        "decisions": decisions,
        "visited_counts": dict(visited_counts),
        "invalid_decisions": invalid_decisions,
        "reprompts": 0,
        "fallback_used": fallback_used,
    }
    

def run_episode_visual(
    nodes,
    model,
    task,
    start_node,
    goal_nodes,
    max_steps,
    memory_mode,
    temperature=0.0,
    stop_on_unsupported=False,
    max_unsupported_streak=3,
    max_edge_repeats=2,
    max_node_visits=4,
    task_prior=None,
    model_stop_on_target=False,
    stop_confidence_threshold=STOP_CONFIDENCE_THRESHOLD,
    max_invalid_stop_attempts=MAX_INVALID_STOP_ATTEMPTS,
):
    current = start_node
    goal_nodes = set(goal_nodes)

    visited_counts = Counter()
    trajectory = []
    decisions = []
    previous_node = None
    invalid_decisions = 0
    fallback_used = 0
    total_distance = 0.0

    state = STATE_START
    visited_goal_nodes = set()
    invalid_stop_attempts = 0
    rejected_move_streak = 0

    edge_counts = Counter()
    edge_bad_counts = Counter()
    node_bad_counts = Counter()
    blocked_edges = set()
    unsupported_streak = 0

    branch_memory = {}
    recent_branch_edges = deque(maxlen=BRANCH_MEMORY_WINDOW)

    for step in range(max_steps + 1):
        visited_counts[current] += 1

        if current in goal_nodes:
            visited_goal_nodes.add(current)

            # Old evaluation mode: goal node terminates the run.
            if not model_stop_on_target:
                return {
                    "success": True,
                    "failure_reason": None,
                    "final_state": STATE_STOP,
                    "model_stopped": False,
                    "stop_correct": True,
                    "final_node": current,
                    "steps": step,
                    "total_distance_m": round(total_distance, 2),
                    "trajectory": trajectory,
                    "decisions": decisions,
                    "visited_counts": dict(visited_counts),
                    "invalid_decisions": invalid_decisions,
                    "reprompts": 0,
                    "fallback_used": fallback_used,
                    "branch_memory": branch_memory,
                    "visited_goal_nodes": sorted(visited_goal_nodes),
                }

        if step == max_steps:
            break

        current_node = nodes[current]
        all_neighbors = list(current_node.get("neighbors", []))

        if not all_neighbors:
            return {
                "success": False,
                "failure_reason": "dead_end",
                "final_node": current,
                "steps": step,
                "total_distance_m": round(total_distance, 2),
                "trajectory": trajectory,
                "decisions": decisions,
                "visited_counts": dict(visited_counts),
                "invalid_decisions": invalid_decisions,
                "reprompts": 0,
                "fallback_used": fallback_used,
            }

        # Visual mode: consider all physically feasible neighbors.
        # Do not force unvisited-only DFS behavior.
        allowed_neighbors = [
            n for n in all_neighbors
            if (current, n) not in blocked_edges
        ]

        # Avoid known dead-end branches if alternatives exist.
        non_dead_end_neighbors = [
            n for n in allowed_neighbors
            if branch_status(branch_memory.get(edge_key(current, n))) != "dead_end"
        ]

        if non_dead_end_neighbors:
            allowed_neighbors = non_dead_end_neighbors

        # Avoid bad branches if alternatives exist.
        non_avoid_neighbors = [
            n for n in allowed_neighbors
            if branch_status(branch_memory.get(edge_key(current, n))) != "avoid"
        ]

        if non_avoid_neighbors:
            allowed_neighbors = non_avoid_neighbors

        if not allowed_neighbors:
            allowed_neighbors = [
                n for n in all_neighbors
                if (current, n) not in blocked_edges
            ]

        if not allowed_neighbors:
            return {
                "success": False,
                "failure_reason": "no_allowed_visual_moves_after_memory_filter",
                "final_node": current,
                "steps": step,
                "total_distance_m": round(total_distance, 2),
                "trajectory": trajectory,
                "decisions": decisions,
                "visited_counts": dict(visited_counts),
                "invalid_decisions": invalid_decisions,
                "reprompts": 0,
                "fallback_used": fallback_used,
            }

        # Avoid immediate oscillation unless it is the only possible move.
        if previous_node in allowed_neighbors and len(allowed_neighbors) > 1:
            allowed_neighbors = [
                n for n in allowed_neighbors
                if n != previous_node
            ]

        # Soft node-visit safety filter.
        # This is not semantic memory; it only prevents local looping.
        if max_node_visits is not None and max_node_visits > 0:
            under_visit_limit = [
                n for n in allowed_neighbors
                if visited_counts[n] < max_node_visits
            ]

            # Only apply the filter if it does not remove every option.
            if under_visit_limit:
                allowed_neighbors = under_visit_limit

        mode = "visual_policy"

        observations = current_observations(current_node)
        
        if model_stop_on_target:
            stop_prompt = build_target_reached_prompt(
                task=task,
                task_prior=task_prior,
                current_node=current_node,
                observations=observations,
            )

            try:
                raw_stop = call_target_reached_checker(
                    model=model,
                    prompt=stop_prompt,
                    temperature=temperature,
                    num_predict=QWEN_TARGET_REACHED_NUM_PREDICT,
                )

                stop_decision = extract_json_object(raw_stop)

                stop_ok, stop_validation_reason = validate_target_reached(
                    stop_decision=stop_decision,
                    current_observations=observations,
                    threshold=stop_confidence_threshold,
                )

            except Exception as e:
                stop_decision = {
                    "target_reached": False,
                    "confidence": 0.0,
                    "evidence_quote": "none",
                    "matched_target": "none",
                    "reason": f"Invalid target-reached checker output: {e}",
                }
                stop_ok = False
                stop_validation_reason = str(e)

            if stop_ok:
                stop_correct = current in goal_nodes

                step_log = {
                    "step": step,
                    "from": current,
                    "to": None,
                    "distance_m": 0.0,
                    "mode": "target_reached_checker",
                    "state": STATE_STOP if stop_correct else STATE_EXPLORING,
                    "action": ACTION_STOP,
                    "executed": False,
                    "stop_correct": stop_correct,
                    "stop_validation_reason": stop_validation_reason,
                    "decision": stop_decision,
                }

                decisions.append(step_log)

                print(
                    f"[step {step}] STOP at {current} | "
                    f"target_reached=True | "
                    f"conf={stop_decision.get('confidence')} | "
                    f"{stop_decision.get('reason')}"
                )

                return {
                    "success": stop_correct,
                    "failure_reason": None if stop_correct else "model_stopped_wrong_node",
                    "final_state": STATE_STOP,
                    "model_stopped": True,
                    "stop_correct": stop_correct,
                    "stop_node": current,
                    "stop_confidence": stop_decision.get("confidence", 0.0),
                    "stop_reason": stop_decision.get("reason", ""),
                    "stop_evidence_quote": stop_decision.get("evidence_quote", ""),
                    "final_node": current,
                    "steps": step,
                    "total_distance_m": round(total_distance, 2),
                    "trajectory": trajectory,
                    "decisions": decisions,
                    "visited_counts": dict(visited_counts),
                    "invalid_decisions": invalid_decisions,
                    "reprompts": 0,
                    "fallback_used": fallback_used,
                    "branch_memory": branch_memory,
                    "visited_goal_nodes": sorted(visited_goal_nodes),
                }

        candidates = build_candidate_neighbors(
            current_node=current_node,
            nodes=nodes,
            visited_counts=visited_counts,
            memory_mode=memory_mode,
            allowed_neighbor_ids=allowed_neighbors,
        )
        for c in candidates:
            nb = c["node_id"]
            key = edge_key(current, nb)
            record = branch_memory.get(key)

            c["edge_visit_count"] = edge_counts[(current, nb)]
            c["edge_bad_count"] = edge_bad_counts[(current, nb)]
            c["node_bad_count"] = node_bad_counts[nb]
            c["task_prior_alignment"] = candidate_prior_alignment(c, task_prior)
            c["visual_audit"] = audit_candidate_visual(c, task_prior)

            c["branch_memory_status"] = branch_status(record)
            c["branch_good_score"] = branch_good_score(record or {})
            c["branch_bad_score"] = branch_bad_score(record or {})
            c["branch_memory_summary"] = branch_summary(record)

        recent_route = [start_node] + trajectory
        candidates = add_candidate_selection_scores(candidates, recent_route)

        visually_available_candidates = [
            c for c in candidates
            if not candidate_is_closed_by_audit(c)
        ]

        if visually_available_candidates:
            candidates = sorted(
                visually_available_candidates,
                key=lambda c: c.get("visual_audit", {}).get("selection_score", 0),
                reverse=True,
            )
            allowed_neighbors = [c["node_id"] for c in candidates]
        else:
            candidates = sorted(
                candidates,
                key=lambda c: c.get("visual_audit", {}).get("selection_score", 0),
                reverse=True,
            )

        visited_memory = make_visited_memory(nodes, visited_counts)
        navigation_memory = {
            "recent_route": ([start_node] + trajectory)[-12:],
            "recent_branch_edges": list(recent_branch_edges),
            "blocked_edges": [f"{a}->{b}" for a, b in sorted(blocked_edges)],
            "branch_memory": build_branch_memory_payload(branch_memory),
            "note": (
                "Avoid avoid/dead_end branches unless there is direct target evidence. "
                "Prefer promising branches when they match the task prior."
            ),
        }
        
        prompt = build_planner_prompt(
            task=task,
            current_node=current_node,
            observations=observations,
            candidates=candidates,
            visited_memory=visited_memory,
            trajectory=trajectory,
            max_steps=max_steps,
            step_idx=step,
            task_prior=task_prior,
            navigation_memory=navigation_memory,
        )

        raw = call_ollama_planner(
            model=model,
            prompt=prompt,
            valid_neighbors=allowed_neighbors,
            temperature=temperature,
        )

        evidence_source_text = build_evidence_source_text(
            observations=observations,
            candidates=candidates,
            visited_memory=visited_memory,
        )

        try:
            decision = extract_json_object(raw)
            decision = sanitize_decision(decision, evidence_source_text)
            chosen = decision.get("next_node_id")
            candidate_by_id = {c["node_id"]: c for c in candidates}
            decision = apply_candidate_constraints(decision, chosen, candidate_by_id)

            if candidates and chosen in candidate_by_id:
                best_candidate = max(candidates, key=candidate_selection_score)
                chosen_candidate = candidate_by_id[chosen]
                best_score = candidate_selection_score(best_candidate)
                chosen_score = candidate_selection_score(chosen_candidate)
                chosen_direct = (
                    chosen_candidate.get("visual_audit", {}).get("target_cue") == "direct"
                )

                if (
                    best_candidate["node_id"] != chosen
                    and not chosen_direct
                    and best_score - chosen_score >= 3
                ):
                    old_choice = chosen
                    chosen = best_candidate["node_id"]
                    decision["planner_choice_overridden"] = old_choice
                    decision["next_node_id"] = chosen
                    decision["decision_basis"] = "navigation_affordance"
                    decision["navigation_affordance"] = (
                        "open_path"
                        if best_candidate.get("visual_audit", {}).get("edge_affordance") == "open"
                        else "weak_path"
                    )
                    decision["move_is_visually_justified"] = (
                        best_candidate.get("visual_audit", {}).get("edge_affordance")
                        in {"open", "unclear"}
                    )
                    decision["confidence"] = min(
                        max(float(decision.get("confidence", 0.0)), 0.45),
                        0.7,
                    )
                    decision["reason"] = (
                        "Overrode lower-scored model choice using visual audit selection_score."
                    )

            action = decision.get("action", ACTION_MOVE)

            current_evidence_text = build_current_evidence_text(observations)

            # ------------------------------------------------------------
            # Model decided to STOP
            # ------------------------------------------------------------
            if model_stop_on_target and action == ACTION_STOP:
                stop_ok, stop_validation_reason = validate_stop_decision(
                    decision=decision,
                    current_evidence_text=current_evidence_text,
                    min_confidence=stop_confidence_threshold,
                )

                step_log = {
                    "step": step,
                    "from": current,
                    "to": None,
                    "distance_m": 0.0,
                    "mode": "visual_policy_stop",
                    "state": STATE_STOP if stop_ok else STATE_EXPLORING,
                    "executed": False,
                    "stop_validation_reason": stop_validation_reason,
                    "decision": decision,
                }

                decisions.append(step_log)
                
                rejected_move_streak += 1

                if rejected_move_streak >= MAX_REJECTED_MOVE_STREAK:
                    return make_cancel_result(
                        current=current,
                        step=step,
                        total_distance=total_distance,
                        trajectory=trajectory,
                        decisions=decisions,
                        visited_counts=visited_counts,
                        invalid_decisions=invalid_decisions,
                        fallback_used=fallback_used,
                        branch_memory=branch_memory,
                        reason="cancelled_due_to_repeated_rejected_moves",
                        visited_goal_nodes=visited_goal_nodes,
                    )

                continue

                if stop_ok:
                    stop_correct = current in goal_nodes

                    return {
                        "success": stop_correct,
                        "failure_reason": None if stop_correct else "model_stopped_wrong_node",
                        "final_state": STATE_STOP,
                        "model_stopped": True,
                        "stop_correct": stop_correct,
                        "stop_node": current,
                        "stop_confidence": decision.get("stop_confidence", 0.0),
                        "stop_reason": decision.get("stop_reason", ""),
                        "final_node": current,
                        "steps": step,
                        "total_distance_m": round(total_distance, 2),
                        "trajectory": trajectory,
                        "decisions": decisions,
                        "visited_counts": dict(visited_counts),
                        "invalid_decisions": invalid_decisions,
                        "reprompts": 0,
                        "fallback_used": fallback_used,
                        "branch_memory": branch_memory,
                        "visited_goal_nodes": sorted(visited_goal_nodes),
                    }

                # Invalid stop: reject it and force a move.
                invalid_stop_attempts += 1

                if invalid_stop_attempts >= max_invalid_stop_attempts:
                    return make_cancel_result(
                        current=current,
                        step=step,
                        total_distance=total_distance,
                        trajectory=trajectory,
                        decisions=decisions,
                        visited_counts=visited_counts,
                        invalid_decisions=invalid_decisions,
                        fallback_used=fallback_used,
                        branch_memory=branch_memory,
                        reason="too_many_invalid_stop_attempts",
                        visited_goal_nodes=visited_goal_nodes,
                    )

                decision["action"] = ACTION_MOVE
                decision["target_reached"] = False
                decision["stop_confidence"] = 0.0
                decision["stop_evidence_quote"] = "none"
                decision["stop_reason"] = "Rejected invalid stop; continuing exploration."
                decision["reason"] = f"Rejected invalid stop: {stop_validation_reason}"

                chosen = min(allowed_neighbors, key=lambda n: visited_counts[n])
                decision["next_node_id"] = chosen

            # ------------------------------------------------------------
            # Model decided to CANCEL
            # ------------------------------------------------------------
            elif model_stop_on_target and action == ACTION_CANCEL:
                step_log = {
                    "step": step,
                    "from": current,
                    "to": None,
                    "distance_m": 0.0,
                    "mode": "visual_policy_cancelled",
                    "state": STATE_CANCELLED,
                    "executed": False,
                    "decision": decision,
                }

                decisions.append(step_log)

                return make_cancel_result(
                    current=current,
                    step=step,
                    total_distance=total_distance,
                    trajectory=trajectory,
                    decisions=decisions,
                    visited_counts=visited_counts,
                    invalid_decisions=invalid_decisions,
                    fallback_used=fallback_used,
                    branch_memory=branch_memory,
                    reason=decision.get("cancellation_reason", "model_cancelled"),
                    visited_goal_nodes=visited_goal_nodes,
                )

            # ------------------------------------------------------------
            # Normal MOVE
            # ------------------------------------------------------------
            else:
                decision["action"] = ACTION_MOVE
                action = ACTION_MOVE
                chosen = decision.get("next_node_id")
    
        except Exception as e:
            invalid_decisions += 1
            chosen = min(allowed_neighbors, key=lambda n: visited_counts[n])
            decision = {
                "next_node_id": chosen,
                "confidence": 0.0,
                "decision_basis": "insufficient_visual_evidence",
                "move_is_visually_justified": False,
                "navigation_affordance": "unknown",
                "visual_evidence_quote": "none",
                "affordance_quote": "none",
                "commonsense_assumption": "none",
                "reason": f"Model returned invalid JSON. Chose least-visited feasible neighbor. Error: {e}",
                "raw_response": raw,
            }
            fallback_used += 1

        if chosen not in allowed_neighbors:
            invalid_decisions += 1
            old_choice = chosen
            chosen = min(allowed_neighbors, key=lambda n: visited_counts[n])

            decision["invalid_choice"] = old_choice
            decision["next_node_id"] = chosen
            decision["decision_basis"] = "insufficient_visual_evidence"
            decision["move_is_visually_justified"] = False
            decision["visual_evidence_quote"] = "none"
            decision["affordance_quote"] = "none"
            decision["reason"] = (
                f"Model chose invalid neighbor '{old_choice}'. "
                f"Fallback selected least-visited feasible neighbor '{chosen}'."
            )
            fallback_used += 1

        edge = (current, chosen)
        
        edge_str = edge_key(current, chosen)
        candidate_by_id = {c["node_id"]: c for c in candidates}
        chosen_candidate = candidate_by_id.get(chosen, {})

        # Reject repeated directed edges before moving.
        if edge_counts[edge] >= max_edge_repeats:
            blocked_edges.add(edge)

            update_branch_memory_for_edge(
                branch_memory=branch_memory,
                edge=edge_str,
                decision=decision,
                candidate=chosen_candidate,
                step=step,
                loop_rejected=True,
            )

            punish_recent_branches(
                branch_memory=branch_memory,
                recent_branch_edges=recent_branch_edges,
                reason=f"Recent branch led to repeated edge {current}->{chosen}.",
                step=step,
                amount=1,
            )

            step_log = {
                "step": step,
                "from": current,
                "to": None,
                "proposed_to": chosen,
                "distance_m": 0.0,
                "mode": "visual_policy_rejected_repeated_edge",
                "executed": False,
                "unsupported_streak": unsupported_streak,
                "candidate_audits": compact_candidate_audits(candidates),
                "decision": {
                    **decision,
                    "reason": (
                        f"Rejected repeated edge {current}->{chosen}; "
                        "added it to branch memory."
                    ),
                },
            }

            decisions.append(step_log)
            continue

        # Update support / bad-edge memory exactly once.
        if decision.get("move_is_visually_justified") is True:
            unsupported_streak = 0
        else:
            unsupported_streak += 1
            edge_bad_counts[edge] += 1
            node_bad_counts[chosen] += 1

            if edge_bad_counts[edge] >= 2:
                blocked_edges.add(edge)

        update_branch_memory_for_edge(
            branch_memory=branch_memory,
            edge=edge_str,
            decision=decision,
            candidate=chosen_candidate,
            step=step,
        )

        # Optional real-robot stop condition.
        if stop_on_unsupported and unsupported_streak >= max_unsupported_streak:
            step_log = {
                "step": step,
                "from": current,
                "to": chosen,
                "distance_m": round(d, 2),
                "mode": mode,
                "state": STATE_EXPLORING,
                "action": ACTION_MOVE,
                "decision": decision,
                "executed": True,
                "unsupported_streak": unsupported_streak,
                "candidate_audits": compact_candidate_audits(candidates),
            }

            decisions.append(step_log)

            return {
                "success": False,
                "failure_reason": "insufficient_visual_grounding",
                "final_node": current,
                "steps": step,
                "total_distance_m": round(total_distance, 2),
                "trajectory": trajectory,
                "decisions": decisions,
                "visited_counts": dict(visited_counts),
                "invalid_decisions": invalid_decisions,
                "reprompts": 0,
                "fallback_used": fallback_used,
                "branch_memory": branch_memory,
            }

        edge_counts[edge] += 1
        recent_branch_edges.append(edge_str)
        
        d = path_distance(nodes, [current, chosen])
        total_distance += d

        step_log = {
            "step": step,
            "from": current,
            "to": chosen,
            "distance_m": round(d, 2),
            "mode": mode,
            "decision": decision,
            "executed": True,
            "unsupported_streak": unsupported_streak,
            "candidate_audits": compact_candidate_audits(candidates),
        }

        trajectory.append(chosen)
        decisions.append(step_log)

        print(
            f"[step {step}] {current} -> {chosen} | "
            f"{decision.get('decision_basis')} | "
            f"visual={decision.get('move_is_visually_justified')} | "
            f"conf={decision.get('confidence')} | "
            f"{decision.get('reason')}"
        )
        
        if chosen in goal_nodes:
            reward_recent_branches(
                branch_memory=branch_memory,
                recent_branch_edges=recent_branch_edges,
                reason="Recent branch sequence reached the goal.",
                step=step,
                amount=3,
            )
            
        previous_node = current
        current = chosen

    return {
        "success": current in goal_nodes,
        "failure_reason": "max_steps",
        "final_node": current,
        "steps": max_steps,
        "total_distance_m": round(total_distance, 2),
        "trajectory": trajectory,
        "decisions": decisions,
        "visited_counts": dict(visited_counts),
        "invalid_decisions": invalid_decisions,
        "reprompts": 0,
        "fallback_used": fallback_used,
        "branch_memory": branch_memory,
    }
  
  
def run_episode(
    nodes,
    model,
    task,
    start_node,
    goal_nodes,
    max_steps,
    memory_mode,
    temperature=0.0,
    controller_mode="dfs",
    visual_stop_on_unsupported=False,
    visual_max_unsupported_streak=3,
    visual_max_edge_repeats=2,
    visual_max_node_visits=4,
    task_prior=None,
    model_stop_on_target=False,
    stop_confidence_threshold=STOP_CONFIDENCE_THRESHOLD,
    max_invalid_stop_attempts=MAX_INVALID_STOP_ATTEMPTS,
):
    if controller_mode == "visual":
        return run_episode_visual(
            nodes=nodes,
            model=model,
            task=task,
            start_node=start_node,
            goal_nodes=goal_nodes,
            max_steps=max_steps,
            memory_mode=memory_mode,
            temperature=temperature,
            stop_on_unsupported=visual_stop_on_unsupported,
            max_unsupported_streak=visual_max_unsupported_streak,
            max_edge_repeats=visual_max_edge_repeats,
            max_node_visits=visual_max_node_visits,
            task_prior=task_prior,
            model_stop_on_target=model_stop_on_target,
            stop_confidence_threshold=stop_confidence_threshold,
            max_invalid_stop_attempts=max_invalid_stop_attempts,
        )

    return run_episode_dfs(
        nodes=nodes,
        model=model,
        task=task,
        start_node=start_node,
        goal_nodes=goal_nodes,
        max_steps=max_steps,
        memory_mode=memory_mode,
        temperature=temperature,
    )
      
    
def compute_visual_grounding_metrics(decisions):
    model_steps = [
        step for step in decisions
        if step.get("mode") in MODEL_DECISION_MODES
    ]

    controller_steps = [
        step for step in decisions
        if step.get("mode") not in MODEL_DECISION_MODES
    ]

    basis_counts = Counter()
    justified_moves = 0
    unsupported_moves = 0
    commonsense_moves = 0
    exploration_moves = 0
    insufficient_visual_moves = 0

    for step in model_steps:
        decision = step.get("decision", {})
        basis = decision.get("decision_basis", "unknown")
        basis_counts[basis] += 1

        if decision.get("move_is_visually_justified") is True:
            justified_moves += 1
        else:
            unsupported_moves += 1

        if basis == "commonsense_prior":
            commonsense_moves += 1

        if basis == "exploration":
            exploration_moves += 1

        if basis == "insufficient_visual_evidence":
            insufficient_visual_moves += 1

    total_model_moves = len(model_steps)

    return {
        "model_decision_steps": total_model_moves,
        "controller_steps": len(controller_steps),
        "visually_justified_moves": justified_moves,
        "unsupported_visual_moves": unsupported_moves,
        "commonsense_moves": commonsense_moves,
        "exploration_moves": exploration_moves,
        "insufficient_visual_evidence_moves": insufficient_visual_moves,
        "visual_grounding_ratio": (
            justified_moves / total_model_moves
            if total_model_moves > 0 else None
        ),
        "decision_basis_counts": dict(basis_counts),
    }


def model_stopped_early_on_non_goal(result, goal_nodes):
    stop_node = result.get("stop_node")
    return bool(
        result.get("model_stopped", False)
        and stop_node is not None
        and stop_node not in set(goal_nodes)
    )


def has_rejected_loop(decisions):
    return any(
        step.get("mode") == "visual_policy_rejected_repeated_edge"
        for step in decisions
    )


def classify_failure_reason(result, visual_metrics, goal_nodes):
    if result.get("success"):
        return None

    failure_reason = result.get("failure_reason")

    if model_stopped_early_on_non_goal(result, goal_nodes):
        return "wrong_stop"

    if failure_reason in {
        "model_stopped_wrong_node",
        "too_many_invalid_stop_attempts",
        "cancelled_due_to_repeated_rejected_moves",
    }:
        return "wrong_stop"

    if failure_reason == "max_steps":
        return "max_steps"

    if failure_reason == "loop" or has_rejected_loop(result.get("decisions", [])):
        return "loop"

    if failure_reason in {
        "insufficient_visual_grounding",
        "no_allowed_visual_moves_after_memory_filter",
    }:
        return "no_visually_justified_move"

    if (
        visual_metrics.get("model_decision_steps", 0) > 0
        and visual_metrics.get("visually_justified_moves", 0) == 0
    ):
        return "no_visually_justified_move"

    if result.get("invalid_decisions", 0) > 0:
        return "invalid_decision"

    return "semantic_failure"


def classify_expected_outcome(result, failure_category):
    if result.get("success"):
        return "success"

    if failure_category == "max_steps":
        return "timeout"

    if failure_category == "wrong_stop":
        return "stop_failure"

    return "semantic_failure"
    
      
def plot_route(nodes, start_node, goal_nodes, route, output_path, manual_path=None):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))

    # Draw edges
    for node_id, node in nodes.items():
        x1 = node["pose"]["x"]
        y1 = node["pose"]["y"]

        for nb in node.get("neighbors", []):
            if nb not in nodes:
                continue

            x2 = nodes[nb]["pose"]["x"]
            y2 = nodes[nb]["pose"]["y"]
            plt.plot([x1, x2], [y1, y2], linewidth=0.5, alpha=0.35)

    # Draw all nodes
    xs = [node["pose"]["x"] for node in nodes.values()]
    ys = [node["pose"]["y"] for node in nodes.values()]
    plt.scatter(xs, ys, s=20)

    for node_id, node in nodes.items():
        plt.text(node["pose"]["x"], node["pose"]["y"], node_id, fontsize=7)

    # Draw model route
    full_route = [start_node] + route
    for a, b in zip(full_route[:-1], full_route[1:]):
        x1 = nodes[a]["pose"]["x"]
        y1 = nodes[a]["pose"]["y"]
        x2 = nodes[b]["pose"]["x"]
        y2 = nodes[b]["pose"]["y"]
        plt.plot([x1, x2], [y1, y2], linewidth=3)

    route_x = [nodes[n]["pose"]["x"] for n in full_route]
    route_y = [nodes[n]["pose"]["y"] for n in full_route]
    plt.scatter(route_x, route_y, s=60)

    # Draw manual path if provided
    if manual_path:
        for a, b in zip(manual_path[:-1], manual_path[1:]):
            if a not in nodes or b not in nodes:
                continue
            x1 = nodes[a]["pose"]["x"]
            y1 = nodes[a]["pose"]["y"]
            x2 = nodes[b]["pose"]["x"]
            y2 = nodes[b]["pose"]["y"]
            plt.plot([x1, x2], [y1, y2], linestyle="--", linewidth=2)

    # Mark start and goals
    plt.scatter(
        [nodes[start_node]["pose"]["x"]],
        [nodes[start_node]["pose"]["y"]],
        s=140,
        marker="s",
        label="start"
    )

    for g in goal_nodes:
        if g in nodes:
            plt.scatter(
                [nodes[g]["pose"]["x"]],
                [nodes[g]["pose"]["y"]],
                s=140,
                marker="*",
                label="goal"
            )

    plt.axis("equal")
    plt.title("Semantic navigation route")
    plt.xlabel("map x [m]")
    plt.ylabel("map y [m]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--graph-json", required=True)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--task", required=True)
    parser.add_argument("--start-node", required=True)
    parser.add_argument("--goal-nodes", required=True, help="Comma-separated list, e.g. n12,n13")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--memory-mode", choices=["partial", "oracle"], default="partial")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--manual-path", default=None, help="Optional comma-separated reference path")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional stable experiment id. Defaults to the timestamped run directory name."
    )
    parser.add_argument(
        "--goal-definition",
        choices=["target_visible_node", "final_object_reaching_node"],
        default="target_visible_node",
        help="How to interpret the provided goal nodes for reporting."
    )
    parser.add_argument(
        "--controller-mode",
        choices=["dfs", "visual"],
        default="dfs",
        help="dfs = graph-search baseline, visual = real-robot-like visual policy"
    )
    parser.add_argument(
        "--visual-stop-on-unsupported",
        action="store_true",
        help="In visual mode, stop if the model repeatedly cannot justify motion visually."
    )

    parser.add_argument(
        "--visual-max-unsupported-streak",
        type=int,
        default=3,
        help="Maximum consecutive visually unsupported moves before stopping."
    )

    parser.add_argument(
        "--visual-max-edge-repeats",
        type=int,
        default=2,
        help="Maximum times the same directed edge can be executed in visual mode."
    )
    parser.add_argument(
        "--visual-max-node-visits",
        type=int,
        default=4,
        help="Soft maximum visits per node in visual mode. Use 0 to disable."
    )
    
    parser.add_argument(
        "--model-stop-on-target",
        action="store_true",
        help="Let the model decide when the target has been reached. Goal nodes are used only for validation."
    )

    parser.add_argument(
        "--stop-confidence-threshold",
        type=float,
        default=STOP_CONFIDENCE_THRESHOLD,
        help="Minimum confidence required for accepting model STOP_TARGET_REACHED."
    )

    parser.add_argument(
        "--max-invalid-stop-attempts",
        type=int,
        default=MAX_INVALID_STOP_ATTEMPTS,
        help="Cancel after this many invalid model stop attempts."
    )

    args = parser.parse_args()
    
    graph_data, nodes = load_graph(args.graph_json)

    start_node = args.start_node
    goal_nodes = [x.strip() for x in args.goal_nodes.split(",") if x.strip()]

    if start_node not in nodes:
        raise ValueError(f"Start node {start_node} not found.")

    for g in goal_nodes:
        if g not in nodes:
            raise ValueError(f"Goal node {g} not found.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.model.replace(':', '_')}_{start_node}"
    experiment_id = args.experiment_id or run_name
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Running semantic navigation experiment")
    print("Model:", args.model)
    print("Task:", args.task)
    print("Start:", start_node)
    print("Goals:", goal_nodes)
    print("Memory mode:", args.memory_mode)
    print("Output:", run_dir)
    print("Controller mode:", args.controller_mode)

    task_prior = enrich_task_prior(
        args.task,
        analyze_task_prior(args.model, args.task)
    )
    print("Task prior:")
    print(json.dumps(task_prior, indent=2, ensure_ascii=False))

    result = run_episode(
        nodes=nodes,
        model=args.model,
        task=args.task,
        start_node=start_node,
        goal_nodes=goal_nodes,
        max_steps=args.max_steps,
        memory_mode=args.memory_mode,
        temperature=args.temperature,
        controller_mode=args.controller_mode,
        visual_stop_on_unsupported=args.visual_stop_on_unsupported,
        visual_max_unsupported_streak=args.visual_max_unsupported_streak,
        visual_max_edge_repeats=args.visual_max_edge_repeats,
        visual_max_node_visits=args.visual_max_node_visits,
        task_prior=task_prior,
        model_stop_on_target=args.model_stop_on_target,
        stop_confidence_threshold=args.stop_confidence_threshold,
        max_invalid_stop_attempts=args.max_invalid_stop_attempts,
    )

    shortest = shortest_path(nodes, start_node, goal_nodes)

    if shortest:
        shortest_steps = len(shortest) - 1
        shortest_distance = path_distance(nodes, shortest)
    else:
        shortest_steps = None
        shortest_distance = None

    manual_path = None
    manual_steps = None
    manual_distance = None

    if args.manual_path:
        manual_path = [x.strip() for x in args.manual_path.split(",") if x.strip()]
        manual_steps = len(manual_path) - 1
        manual_distance = path_distance(nodes, manual_path)

    oracle_path = manual_path if manual_path is not None else shortest
    oracle_path_source = "manual" if manual_path is not None else "shortest_path"
    oracle_steps = manual_steps if manual_path is not None else shortest_steps
    oracle_distance = manual_distance if manual_path is not None else shortest_distance

    actual_route = [start_node] + result["trajectory"]

    if result["success"]:
        spl = shortest_steps / max(result["steps"], shortest_steps) if shortest_steps is not None else None
    else:
        spl = 0.0

    repeated_node_count = sum(
        count - 1 for count in result["visited_counts"].values()
        if count > 1
    )

    unique_nodes_visited = len(result["visited_counts"])
    visual_metrics = compute_visual_grounding_metrics(result["decisions"])
    failure_category = classify_failure_reason(result, visual_metrics, goal_nodes)
    expected_outcome = classify_expected_outcome(result, failure_category)
    target_visible_at_start = target_visibility_at_node(nodes[start_node], task_prior)
    stopped_early_on_non_goal = model_stopped_early_on_non_goal(result, goal_nodes)
    graph_path = Path(args.graph_json)
    graph_meta = graph_data.get("meta", {})
    
    metrics = {
        "experiment_id": experiment_id,
        "expected_outcome": expected_outcome,
        "model": args.model,
        "task": args.task,
        "dataset_file": str(graph_path),
        "dataset_file_resolved": str(graph_path.resolve()),
        "dataset_version": graph_dataset_version(graph_data),
        "dataset_hash_sha256": file_sha256(graph_path),
        "dataset_source": graph_meta.get("source_dataset"),
        "graph_node_count": len(nodes),
        "graph_edge_count": graph_edge_count(nodes),
        "graph_directed_edge_count": graph_directed_edge_count(nodes),
        "start_node": start_node,
        "goal_nodes": goal_nodes,
        "goal_definition": args.goal_definition,
        "memory_mode": args.memory_mode,
        "success": result["success"],
        "failure_reason": result.get("failure_reason"),
        "reason_for_failure": failure_category,
        "final_node": result["final_node"],
        "steps": result["steps"],
        "total_distance_m": result["total_distance_m"],
        "shortest_path": shortest,
        "shortest_steps": shortest_steps,
        "shortest_distance_m": round(shortest_distance, 2) if shortest_distance is not None else None,
        "shortest_path_length_to_nearest_goal_steps": shortest_steps,
        "shortest_path_length_to_nearest_goal_m": round(shortest_distance, 2) if shortest_distance is not None else None,
        "manual_path": manual_path,
        "manual_steps": manual_steps,
        "manual_distance_m": round(manual_distance, 2) if manual_distance is not None else None,
        "manual_oracle_path": oracle_path,
        "manual_oracle_path_source": oracle_path_source,
        "manual_oracle_path_length_steps": oracle_steps,
        "manual_oracle_path_length_m": round(oracle_distance, 2) if oracle_distance is not None else None,
        "spl": spl,
        "invalid_decisions": result["invalid_decisions"],
        "reprompts": result["reprompts"],
        "fallback_used": result["fallback_used"],
        "actual_route": actual_route,
        "actual_path_length_steps": len(actual_route) - 1,
        "actual_path_length_m": result["total_distance_m"],
        "visited_counts": result["visited_counts"],
        "repeated_node_count": repeated_node_count,
        "unique_nodes_visited": unique_nodes_visited,
        "visual_grounding_ratio": visual_metrics["visual_grounding_ratio"],
        "visually_justified_moves": visual_metrics["visually_justified_moves"],
        "unsupported_visual_moves": visual_metrics["unsupported_visual_moves"],
        "commonsense_moves": visual_metrics["commonsense_moves"],
        "exploration_moves": visual_metrics["exploration_moves"],
        "insufficient_visual_evidence_moves": visual_metrics["insufficient_visual_evidence_moves"],
        "model_decision_steps": visual_metrics["model_decision_steps"],
        "controller_steps": visual_metrics["controller_steps"],
        "decision_basis_counts": visual_metrics["decision_basis_counts"],
        "controller_mode": args.controller_mode,
        "visual_stop_on_unsupported": args.visual_stop_on_unsupported,
        "visual_max_unsupported_streak": args.visual_max_unsupported_streak,
        "visual_max_edge_repeats": args.visual_max_edge_repeats,
        "task_prior_output": task_prior,
        "task_prior": task_prior,
        "target_visible_at_start_node": target_visible_at_start["visible"],
        "target_visible_at_start_evidence_quote": target_visible_at_start["evidence_quote"],
        "target_visible_at_start_matched_terms": target_visible_at_start["matched_terms"],
        "visual_max_node_visits": args.visual_max_node_visits,
        "model_stop_on_target": args.model_stop_on_target,
        "final_state": result.get("final_state"),
        "model_stopped": result.get("model_stopped", False),
        "model_stopped_early_on_non_goal_node": stopped_early_on_non_goal,
        "stop_correct": result.get("stop_correct", False),
        "stop_node": result.get("stop_node"),
        "stop_threshold": args.stop_confidence_threshold,
        "stop_confidence": result.get("stop_confidence"),
        "stop_reason": result.get("stop_reason"),
        "stop_evidence_quote": result.get("stop_evidence_quote"),
        "prompt_versions": {
            "moondream": graph_moondream_prompt_version(graph_data),
            "qwen_task_prior": QWEN_TASK_PRIOR_PROMPT_VERSION,
            "qwen_planner": QWEN_PLANNER_PROMPT_VERSION,
            "qwen_target_reached_checker": QWEN_TARGET_REACHED_PROMPT_VERSION,
        },
        "model_temperature": args.temperature,
        "decoding_settings": {
            "qwen_task_prior": {
                "temperature": 0.0,
                "num_predict": QWEN_TASK_PRIOR_NUM_PREDICT,
            },
            "qwen_planner": {
                "temperature": args.temperature,
                "num_predict": QWEN_PLANNER_NUM_PREDICT,
            },
            "qwen_target_reached_checker": {
                "temperature": args.temperature,
                "num_predict": QWEN_TARGET_REACHED_NUM_PREDICT,
            },
            "moondream": {
                "temperature": None,
                "num_predict": None,
                "source": "precomputed graph descriptions",
            },
        },
        "visited_goal_nodes": result.get("visited_goal_nodes", []),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "branch_memory": result.get("branch_memory", {}),
            "decisions": result["decisions"],
        }, f, indent=2, ensure_ascii=False)

    plot_route(
        nodes=nodes,
        start_node=start_node,
        goal_nodes=goal_nodes,
        route=result["trajectory"],
        output_path=run_dir / "route.png",
        manual_path=manual_path,
    )

    print("\nFinished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nSaved:")
    print(" ", run_dir / "result.json")
    print(" ", run_dir / "route.png")


if __name__ == "__main__":
    main()
