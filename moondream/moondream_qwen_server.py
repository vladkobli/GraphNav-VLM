#!/usr/bin/env python3
"""
Moondream + Qwen HTTP server for online semantic navigation.

Run OUTSIDE Docker in the Moondream venv:

    cd /home/vladkobli/rocon-demos/moondream
    source venv_moondream/bin/activate
    python moondream_qwen_server.py

The ROS2 / Docker pipeline can call:

    GET  http://127.0.0.1:8890/health

    POST http://127.0.0.1:8890/describe_image
    POST http://127.0.0.1:8890/describe_360
    POST http://127.0.0.1:8890/task_prior
    POST http://127.0.0.1:8890/check_target
    POST http://127.0.0.1:8890/score_candidates
    POST http://127.0.0.1:8890/decide

Typical Docker/shared-folder setup:

    Host path:
        /home/vladkobli/rocon-demos/online_nav_shared

    Docker path:
        /shared

    Run this server with:
        PATH_MAPS="/shared=/home/vladkobli/rocon-demos/online_nav_shared" \
        python moondream_qwen_server.py
"""

import json
import os
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from textwrap import dedent

from PIL import Image
from transformers import AutoModelForCausalLM
from ollama import chat


# -----------------------------------------------------------------------------
# Server config
# -----------------------------------------------------------------------------

HOST = os.environ.get("MOONDREAM_SERVER_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("MOONDREAM_SERVER_PORT", "8890"))

MOONDREAM_MODEL_ID = os.environ.get(
    "MOONDREAM_MODEL_ID",
    "moondream/moondream-2b-2025-04-14-4bit",
)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Example:
#   PATH_MAPS="/shared=/home/vladkobli/rocon-demos/online_nav_shared"
PATH_MAPS = os.environ.get("PATH_MAPS", "")

STOP_CONFIDENCE_THRESHOLD = float(os.environ.get("STOP_CONFIDENCE_THRESHOLD", "0.75"))

ACTION_MOVE = "MOVE"
ACTION_STOP = "STOP_TARGET_REACHED"
ACTION_CANCEL = "CANCELLED"

model = None
model_lock = threading.Lock()
ollama_lock = threading.Lock()


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def parse_path_maps():
    maps = []

    if not PATH_MAPS.strip():
        return maps

    for pair in PATH_MAPS.split(","):
        pair = pair.strip()
        if not pair:
            continue

        if "=" not in pair:
            raise ValueError(f"Invalid PATH_MAPS entry: {pair!r}")

        left, right = pair.split("=", 1)
        maps.append((left.rstrip("/"), right.rstrip("/")))

    return maps


PATH_MAP_LIST = parse_path_maps()


def remap_path(path_str: str) -> Path:
    """
    Converts Docker-visible paths into host-visible paths.

    Example:
        /shared/node_000/front.png
    becomes:
        /home/vladkobli/rocon-demos/online_nav_shared/node_000/front.png
    """
    path_str = str(path_str)

    for docker_prefix, host_prefix in PATH_MAP_LIST:
        if path_str == docker_prefix or path_str.startswith(docker_prefix + "/"):
            suffix = path_str[len(docker_prefix):].lstrip("/")
            return Path(host_prefix) / suffix

    return Path(path_str)


def extract_json_object(text):
    text = str(text).strip()

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


def short_description(desc, max_chars=400):
    desc = str(desc).replace("\n", " | ").strip()
    if len(desc) > max_chars:
        return desc[:max_chars] + "..."
    return desc


def clean_one_line(text: str) -> str:
    return str(text).strip().split("\n")[0].strip()


def clean_prefixed_line(text: str, prefixes: list[str]) -> str:
    text = clean_one_line(text)

    for prefix in prefixes:
        text = re.sub(
            rf"^\s*{prefix}\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

    return text.strip()


def dedupe_comma_list(text: str, max_items: int = 8) -> str:
    text = clean_one_line(text)
    parts = [p.strip(" .") for p in re.split(r",|;", text) if p.strip(" .")]

    cleaned = []
    seen = set()

    for item in parts:
        key = item.lower()

        if key in {"objects", "things", "equipment", "machinery"}:
            continue

        if key not in seen:
            seen.add(key)
            cleaned.append(item)

        if len(cleaned) >= max_items:
            break

    return ", ".join(cleaned) if cleaned else "none"


def observation_text(observations):
    return "\n".join(str(obs.get("description", "")) for obs in observations)


def quote_is_in_observations(quote, observations):
    quote_l = normalize_for_match(quote)
    obs_l = normalize_for_match(observation_text(observations))

    return bool(quote_l and quote_l != "none" and quote_l in obs_l)


# -----------------------------------------------------------------------------
# Moondream prompts
# -----------------------------------------------------------------------------

def build_scene_prompt():
    return """
Describe the visible place in one short sentence.

Rules:
- Mention only what is directly visible.
- Do not guess hidden objects.
- Do not repeat phrases.
- Use at most 20 words.
""".strip()


def build_objects_prompt():
    return """
List up to 8 main concrete objects directly visible in the image.

Rules:
- Return only a comma-separated list.
- Do not repeat any item.
- If many identical objects are visible, summarize them once.
- Prefer specific objects over generic words like equipment, machinery, things, or objects.
- Do not explain.
""".strip()


def build_landmarks_prompt():
    return """
List distinctive visual landmarks directly visible in the image.

A landmark is a visible feature that would help recognize this place again.
Only include things you can clearly see.
Do not guess hidden objects.
Do not copy examples from the instruction.

Return only a comma-separated list.
If there are no distinctive landmarks, write: none.
""".strip()


def build_navigation_prompt():
    return """
Describe visible navigation cues in the image.

Mention only directly visible:
- floor or ground area
- path direction
- doorways or entrances
- corridors or hallways
- ramps or stairs
- walls or blocked paths
- large obstacles

Do not guess where paths lead.

Use one short sentence.
If no useful path cue is visible, write: no clear navigation cue visible.
""".strip()


def build_traversable_openings_prompt():
    return """
Describe visible traversable openings or routes in the image.

Only mention an opening or route if it is clearly visible.

Include only directly visible:
- open doorway
- entrance
- corridor
- hallway
- ramp
- stairs
- walkway
- open passage
- clear route through or around objects

Only mention a traversable opening if the path is physically open and visible.
A closed door is not a traversable opening.
A wall, facade, window, courtyard, or building does not imply an entrance.
If only a closed door is visible, write: closed door visible, not confirmed traversable.
If none, write: none.
""".strip()


def build_scene_type_prompt():
    return """
Classify the visible scene type using only what is directly visible.

Choose one:
indoor, outdoor, transition, unknown

Definitions:
- indoor: inside a building or enclosed room/hallway
- outdoor: outside, courtyard, street, garden, exterior path
- transition: doorway, entrance, exit, lobby, covered entrance, threshold, or area connecting indoor and outdoor
- unknown: unclear

Return only one word.
""".strip()


def describe_image(image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")

    with model_lock:
        model.generation_config.max_new_tokens = 35
        scene = model.query(image, build_scene_prompt())["answer"]
        scene = clean_prefixed_line(scene, ["scene"])

        model.generation_config.max_new_tokens = 35
        visible_objects = model.query(image, build_objects_prompt())["answer"]
        visible_objects = clean_prefixed_line(
            visible_objects,
            ["visible_objects", "visible objects", "objects"],
        )
        visible_objects = dedupe_comma_list(visible_objects, max_items=8)

        model.generation_config.max_new_tokens = 40
        landmarks = model.query(image, build_landmarks_prompt())["answer"]
        landmarks = clean_prefixed_line(
            landmarks,
            ["distinctive_landmarks", "distinctive landmarks", "landmarks"],
        )
        landmarks = dedupe_comma_list(landmarks, max_items=8)

        model.generation_config.max_new_tokens = 35
        navigation_cues = model.query(image, build_navigation_prompt())["answer"]
        navigation_cues = clean_prefixed_line(
            navigation_cues,
            ["navigation_cues", "navigation cues", "navigation"],
        )

        model.generation_config.max_new_tokens = 50
        traversable_openings = model.query(image, build_traversable_openings_prompt())["answer"]
        traversable_openings = clean_prefixed_line(
            traversable_openings,
            [
                "traversable_openings",
                "traversable openings",
                "visible traversable openings",
                "openings",
                "routes",
            ],
        )

        model.generation_config.max_new_tokens = 20
        scene_type = model.query(image, build_scene_type_prompt())["answer"]
        scene_type = clean_prefixed_line(
            scene_type,
            ["scene_type", "scene type", "type"],
        ).lower()

    if scene_type not in {"indoor", "outdoor", "transition", "unknown"}:
        scene_type = "unknown"

    description = (
        f"Scene: {scene}\n"
        f"Visible objects: {visible_objects}\n"
        f"Distinctive landmarks: {landmarks}\n"
        f"Navigation cues: {navigation_cues}\n"
        f"Traversable openings: {traversable_openings}\n"
        f"Scene type: {scene_type}"
    )

    return {
        "scene": scene,
        "visible_objects": visible_objects,
        "distinctive_landmarks": landmarks,
        "navigation_cues": navigation_cues,
        "traversable_openings": traversable_openings,
        "scene_type": scene_type,
        "description": description,
        "model": MOONDREAM_MODEL_ID,
    }


# -----------------------------------------------------------------------------
# Qwen / Ollama schemas and calls
# -----------------------------------------------------------------------------

def ollama_json_chat(messages, schema, temperature=0.0, num_predict=300):
    with ollama_lock:
        response = chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": num_predict,
            },
            format=schema,
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
                "enum": ["indoor", "outdoor", "either", "unknown"],
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
                    "unknown",
                ],
            },
            "target_synonyms": {
                "type": "array",
                "items": {"type": "string"},
            },
            "likely_contexts": {
                "type": "array",
                "items": {"type": "string"},
            },
            "unlikely_contexts": {
                "type": "array",
                "items": {"type": "string"},
            },
            "useful_visual_cues": {
                "type": "array",
                "items": {"type": "string"},
            },
            "search_strategy": {
                "type": "string",
            },
        },
        "required": [
            "target_environment",
            "target_place_type",
            "target_synonyms",
            "likely_contexts",
            "unlikely_contexts",
            "useful_visual_cues",
            "search_strategy",
        ],
    }


def analyze_task_prior(task):
    prompt = f"""
Analyze this mobile robot navigation task.

Task:
{task}

Infer general commonsense search priors.

Rules:
- Do not assume the target is visible.
- Do not use map-specific knowledge.
- These are only commonsense priors, not visual evidence.
- If the task asks for an object, keep object names in target_synonyms.
- Put rooms or areas where the object is usually found in likely_contexts.
- Keep lists short and general.

Examples:
- bathrooms: corridors, public indoor areas, signs, doors, sinks, tiled rooms.
- classrooms: corridors, lecture rooms, desks, chairs, boards.
- vending machines: hallways, lobbies, common areas.
- coffee machines: kitchens, break rooms, cafeterias, common areas.
- blackboards: classrooms, lecture halls, seminar rooms.

Return only JSON.
""".strip()

    return ollama_json_chat(
        messages=[
            {
                "role": "system",
                "content": "You infer general task priors for robot navigation. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        schema=make_task_prior_schema(),
        temperature=0.0,
        num_predict=260,
    )


def make_target_reached_schema():
    return {
        "type": "object",
        "properties": {
            "target_reached": {
                "type": "boolean",
            },
            "confidence": {
                "type": "number",
            },
            "evidence_quote": {
                "type": "string",
            },
            "matched_target": {
                "type": "string",
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "target_reached",
            "confidence",
            "evidence_quote",
            "matched_target",
            "reason",
        ],
    }


def check_target_reached(task, observations, task_prior=None, threshold=STOP_CONFIDENCE_THRESHOLD):
    compact_observations = []

    for obs in observations:
        compact_observations.append({
            "camera": obs.get("camera_id") or obs.get("camera"),
            "relative_yaw_deg": obs.get("relative_yaw_deg"),
            "description": short_description(obs.get("description", ""), 600),
        })

    payload = {
        "task": task,
        "task_prior": task_prior or {},
        "current_observations": compact_observations,
    }

    prompt = dedent(f"""
    You decide whether a mobile robot has already reached the target.

    Rules:
    - Use ONLY current_observations.
    - Do NOT use candidate actions.
    - Do NOT use visited memory.
    - Do NOT use future possibilities.
    - The task itself is not evidence.
    - Stop only if the current scene clearly satisfies the task.
    - For object targets, the object or a close synonym must be visible or explicitly named.
    - For place targets, the current scene must explicitly match the requested place type.
    - For "classroom", accept classroom, lecture hall, seminar room, teaching room, room with desks/chairs, chalkboard, blackboard, whiteboard, or podium if clearly described.
    - For "vending machine", accept only vending machine, snack machine, drink machine, or a clearly described machine matching that object.
    - Do not stop because the target might be nearby.
    - Do not stop because a doorway/corridor could lead to the target.
    - evidence_quote must be copied exactly from current_observations.
    - If there is no exact evidence, return target_reached=false and evidence_quote="none".

    Input:
    {json.dumps(payload, indent=2, ensure_ascii=False)}

    Return only JSON.
    """).strip()

    decision = ollama_json_chat(
        messages=[
            {
                "role": "system",
                "content": "You are a target-reached checker for a mobile robot. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        schema=make_target_reached_schema(),
        temperature=0.0,
        num_predict=220,
    )

    decision["confidence"] = clamp01(decision.get("confidence", 0.0))

    quote_ok = quote_is_in_observations(decision.get("evidence_quote", ""), observations)
    accepted = (
        decision.get("target_reached") is True
        and decision["confidence"] >= threshold
        and quote_ok
    )

    decision["accepted"] = accepted
    decision["validation"] = {
        "threshold": threshold,
        "quote_found_in_current_observations": quote_ok,
    }

    return decision


def make_score_candidates_schema(candidate_ids):
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [ACTION_MOVE, ACTION_CANCEL],
            },
            "selected_candidate_id": {
                "type": "string",
                "enum": list(candidate_ids) + ["none"],
            },
            "confidence": {
                "type": "number",
            },
            "decision_basis": {
                "type": "string",
                "enum": [
                    "direct_visual_evidence",
                    "contextual_visual_evidence",
                    "navigation_affordance",
                    "commonsense_prior",
                    "exploration",
                    "insufficient_visual_evidence",
                    "uncertain",
                ],
            },
            "visual_evidence_quote": {
                "type": "string",
            },
            "affordance_quote": {
                "type": "string",
            },
            "reason": {
                "type": "string",
            },
            "candidate_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "score", "reason"],
                },
            },
        },
        "required": [
            "action",
            "selected_candidate_id",
            "confidence",
            "decision_basis",
            "visual_evidence_quote",
            "affordance_quote",
            "reason",
            "candidate_scores",
        ],
    }


def score_candidates(task, observations, candidates, task_prior=None, memory=None):
    candidate_ids = [str(c["candidate_id"]) for c in candidates]

    compact_observations = [
        {
            "camera": obs.get("camera_id") or obs.get("camera"),
            "relative_yaw_deg": obs.get("relative_yaw_deg"),
            "description": short_description(obs.get("description", ""), 350),
        }
        for obs in observations
    ]

    compact_candidates = []

    for c in candidates:
        compact_candidates.append({
            "candidate_id": c.get("candidate_id"),
            "action_type": c.get("action_type", "local_goal"),
            "direction": c.get("direction"),
            "heading_deg": c.get("heading_deg"),
            "distance_m": c.get("distance_m"),
            "is_collision_free": c.get("is_collision_free", True),
            "free_space_distance_m": c.get("free_space_distance_m"),
            "view_descriptions": [
                short_description(v.get("description", ""), 350)
                for v in c.get("views_toward_candidate", [])
            ],
            "geometry_note": c.get("geometry_note", ""),
            "memory_status": c.get("memory_status", "untried"),
        })

    payload = {
        "task": task,
        "task_prior": task_prior or {},
        "memory": memory or {},
        "current_observations": compact_observations,
        "candidate_actions": compact_candidates,
        "allowed_candidate_ids": candidate_ids,
    }

    prompt = dedent(f"""
    Choose the next action candidate for a mobile robot.

    Rules:
    - Choose exactly one candidate_id from allowed_candidate_ids, or CANCELLED if no useful/safe candidate exists.
    - Use geometry safety first: do not choose candidates marked is_collision_free=false.
    - Use the current image descriptions and candidate direction descriptions.
    - Do not invent objects, signs, doors, rooms, or paths.
    - Prefer candidates whose visible direction supports the user's task.
    - Prefer open paths, corridors, hallways, doorways, ramps, stairs, or clear free space.
    - Closed doors are blocked, not promising.
    - For hidden targets, use task_prior only as commonsense guidance, not visual evidence.
    - If the target itself is visible, prefer moving toward it unless the target checker should have stopped.
    - visual_evidence_quote and affordance_quote must be copied from the provided descriptions, or "none".
    - confidence must be between 0 and 1.
    - Return only JSON.

    Input:
    {json.dumps(payload, indent=2, ensure_ascii=False)}
    """).strip()

    decision = ollama_json_chat(
        messages=[
            {
                "role": "system",
                "content": "You are a robot local-navigation candidate scorer. Return only valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        schema=make_score_candidates_schema(candidate_ids),
        temperature=0.0,
        num_predict=550,
    )

    decision["confidence"] = clamp01(decision.get("confidence", 0.0))

    if decision.get("selected_candidate_id") not in candidate_ids:
        decision["action"] = ACTION_CANCEL
        decision["selected_candidate_id"] = "none"
        decision["confidence"] = min(decision["confidence"], 0.2)
        decision["reason"] = "No valid candidate was selected."

    return decision


def decide(task, observations, candidates, task_prior=None, memory=None, stop_threshold=STOP_CONFIDENCE_THRESHOLD):
    if task_prior is None:
        task_prior = analyze_task_prior(task)

    stop_decision = check_target_reached(
        task=task,
        observations=observations,
        task_prior=task_prior,
        threshold=stop_threshold,
    )

    if stop_decision.get("accepted") is True:
        return {
            "action": ACTION_STOP,
            "target_reached": True,
            "stop_decision": stop_decision,
            "task_prior": task_prior,
            "selected_candidate_id": "none",
            "candidate_decision": None,
        }

    if not candidates:
        return {
            "action": ACTION_CANCEL,
            "target_reached": False,
            "reason": "No candidate actions were provided.",
            "stop_decision": stop_decision,
            "task_prior": task_prior,
            "selected_candidate_id": "none",
            "candidate_decision": None,
        }

    candidate_decision = score_candidates(
        task=task,
        observations=observations,
        candidates=candidates,
        task_prior=task_prior,
        memory=memory or {},
    )

    return {
        "action": candidate_decision.get("action", ACTION_MOVE),
        "target_reached": False,
        "stop_decision": stop_decision,
        "task_prior": task_prior,
        "selected_candidate_id": candidate_decision.get("selected_candidate_id", "none"),
        "candidate_decision": candidate_decision,
    }


# -----------------------------------------------------------------------------
# HTTP handler
# -----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body) if body.strip() else {}

    def log_message(self, fmt, *args):
        sys.stderr.write("HTTP: " + fmt % args + "\n")

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return

        self._send_json(200, {
            "ok": True,
            "service": "moondream_qwen_server",
            "moondream_model": MOONDREAM_MODEL_ID,
            "ollama_model": OLLAMA_MODEL,
            "path_maps": PATH_MAP_LIST,
            "endpoints": [
                "/describe_image",
                "/describe_360",
                "/task_prior",
                "/check_target",
                "/score_candidates",
                "/decide",
            ],
        })

    def do_POST(self):
        try:
            payload = self._read_json()

            if self.path == "/describe_image":
                image_path = remap_path(payload["image_path"])

                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                desc = describe_image(image_path)

                self._send_json(200, {
                    "ok": True,
                    "image_path": str(image_path),
                    **desc,
                })
                return

            if self.path == "/describe_360":
                images = payload.get("images", [])
                observations = []

                for item in images:
                    image_path = remap_path(item["image_path"])

                    if not image_path.exists():
                        raise FileNotFoundError(f"Image not found: {image_path}")

                    desc = describe_image(image_path)

                    observations.append({
                        "image_path": str(image_path),
                        "camera_id": item.get("camera_id"),
                        "stop_name": item.get("stop_name"),
                        "relative_yaw_deg": item.get("relative_yaw_deg"),
                        "heading_abs_deg": item.get("heading_abs_deg"),
                        "description": desc["description"],
                        "semantic_description": {
                            "scene": desc["scene"],
                            "visible_objects": desc["visible_objects"],
                            "distinctive_landmarks": desc["distinctive_landmarks"],
                            "navigation_cues": desc["navigation_cues"],
                            "traversable_openings": desc["traversable_openings"],
                            "scene_type": desc["scene_type"],
                            "model": desc["model"],
                        },
                    })

                self._send_json(200, {
                    "ok": True,
                    "observations": observations,
                })
                return

            if self.path == "/task_prior":
                task = payload["task"]
                prior = analyze_task_prior(task)
                self._send_json(200, {
                    "ok": True,
                    "task": task,
                    "task_prior": prior,
                })
                return

            if self.path == "/check_target":
                decision = check_target_reached(
                    task=payload["task"],
                    observations=payload.get("observations", []),
                    task_prior=payload.get("task_prior", {}),
                    threshold=float(payload.get("threshold", STOP_CONFIDENCE_THRESHOLD)),
                )
                self._send_json(200, {
                    "ok": True,
                    "decision": decision,
                })
                return

            if self.path == "/score_candidates":
                decision = score_candidates(
                    task=payload["task"],
                    observations=payload.get("observations", []),
                    candidates=payload.get("candidates", []),
                    task_prior=payload.get("task_prior", {}),
                    memory=payload.get("memory", {}),
                )
                self._send_json(200, {
                    "ok": True,
                    "decision": decision,
                })
                return

            if self.path == "/decide":
                result = decide(
                    task=payload["task"],
                    observations=payload.get("observations", []),
                    candidates=payload.get("candidates", []),
                    task_prior=payload.get("task_prior"),
                    memory=payload.get("memory", {}),
                    stop_threshold=float(payload.get("stop_threshold", STOP_CONFIDENCE_THRESHOLD)),
                )
                self._send_json(200, {
                    "ok": True,
                    "result": result,
                })
                return

            self._send_json(404, {"ok": False, "error": "unknown endpoint"})

        except Exception as exc:
            self._send_json(500, {
                "ok": False,
                "error": str(exc),
                "endpoint": self.path,
            })


# -----------------------------------------------------------------------------
# Startup / shutdown
# -----------------------------------------------------------------------------

def shutdown(*_):
    print("\nShutting down Moondream/Qwen server...")
    sys.exit(0)


def main():
    global model

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Loading Moondream model...")
    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID,
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.35
    model.generation_config.no_repeat_ngram_size = 2

    print("Moondream loaded.")
    print("Server config:")
    print(json.dumps({
        "host": HOST,
        "port": HTTP_PORT,
        "moondream_model": MOONDREAM_MODEL_ID,
        "ollama_model": OLLAMA_MODEL,
        "path_maps": PATH_MAP_LIST,
    }, indent=2))

    print(f"Listening on http://{HOST}:{HTTP_PORT}")
    httpd = ThreadingHTTPServer((HOST, HTTP_PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()