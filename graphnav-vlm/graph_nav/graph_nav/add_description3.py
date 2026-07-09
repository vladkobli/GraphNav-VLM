from pathlib import Path
import argparse
import json
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


MODEL_ID = "moondream-2b-2025-04-14-4bit"
PROMPT_MODE = "visual_summary_prompt_v6"


def build_single_prompt(task: str | None = None, target_prompt: str | None = None) -> str:
    return """
Look at this robot camera image and describe only what is visible.

Do not guess hidden rooms, objects, signs, paths, or destinations.
Do not infer what the robot is searching for.

Return exactly this format:

Summary: one natural sentence about the visible scene
Scene type:
Visible objects:
Distinctive landmarks:
Open traversable routes:
Blocked or unclear routes:
Indoor/outdoor transition evidence:

Rules:
- Scene type must be one word: indoor, outdoor, transition, or unknown.
- Visible objects should list concrete objects only.
- Write none for any field that is not visible.
- A closed door is blocked, not open.
- A wall, window, courtyard, or facade is not a route unless an open entrance is visible.
""".strip()


def query_moondream_answer(model, image, prompt: str) -> str:
    result = model.query(image, prompt)

    if isinstance(result, dict):
        if "answer" not in result:
            raise KeyError(f"Moondream query response has no 'answer' key: {result!r}")
        return str(result["answer"])

    if isinstance(result, str):
        return result

    raise TypeError(f"Unexpected Moondream query response type: {type(result).__name__}")


def clean_one_line(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_labeled_lines(text: str) -> dict:
    fields = {}
    current_key = None
    aliases = {
        "summary": "summary",
        "scene": "scene",
        "scene type": "scene_type",
        "scene_type": "scene_type",
        "objects": "visible_objects",
        "visible objects": "visible_objects",
        "landmarks": "distinctive_landmarks",
        "distinctive landmarks": "distinctive_landmarks",
        "open traversable routes": "open_traversable_routes",
        "open_traversable_routes": "open_traversable_routes",
        "open routes": "open_traversable_routes",
        "open route": "open_traversable_routes",
        "blocked or unclear routes": "blocked_or_unclear_routes",
        "blocked_or_unclear_routes": "blocked_or_unclear_routes",
        "blocked routes": "blocked_or_unclear_routes",
        "blocked route": "blocked_or_unclear_routes",
        "indoor/outdoor transition evidence": "indoor_outdoor_transition_evidence",
        "indoor outdoor transition evidence": "indoor_outdoor_transition_evidence",
        "indoor_outdoor_transition_evidence": "indoor_outdoor_transition_evidence",
        "transition": "indoor_outdoor_transition_evidence",
        "direct target evidence": "direct_target_evidence",
        "direct_target_evidence": "direct_target_evidence",
        "target visible": "direct_target_evidence",
        "useful context for the task": "useful_context_for_task",
        "useful_context_for_task": "useful_context_for_task",
        "useful context": "useful_context_for_task",
        "context": "useful_context_for_task",
        "navigation recommendation for this view": "navigation_recommendation",
        "navigation_recommendation": "navigation_recommendation",
        "recommendation": "navigation_recommendation",
        "task relevance score": "task_relevance_score",
        "task_relevance_score": "task_relevance_score",
        "score": "task_relevance_score",
        "navigation": "navigation_cues",
        "navigation cues": "navigation_cues",
        "traversable openings": "traversable_openings",
        "task relevance": "task_relevance",
        "task_relevance": "task_relevance",
        "best direction cue": "best_direction_cue",
        "best_direction_cue": "best_direction_cue",
    }

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = re.match(r"^\s*[-*]?\s*([A-Za-z _/]+)\s*:\s*(.*)$", line)
        if match:
            label = clean_one_line(match.group(1)).lower()
            key = aliases.get(label)
            if key:
                value = clean_one_line(match.group(2))
                fields[key] = value
                current_key = key if value else None
                continue

        if current_key:
            fields[current_key] = clean_one_line(f"{fields[current_key]} {line}")

    if not fields:
        fields["scene"] = clean_one_line(text)

    return fields


def normalize_scene_type(value: str) -> str:
    value_l = clean_one_line(value).lower()
    for scene_type in ("indoor", "outdoor", "transition", "unknown"):
        if scene_type in value_l:
            return scene_type
    return "unknown"


def normalize_relevance(value: str) -> str:
    value_l = clean_one_line(value).lower()
    score_match = re.search(r"\bscore\s*([0-5])\s*/\s*5\b", value_l)
    if score_match:
        score = int(score_match.group(1))
        if score >= 5:
            return "direct"
        if score >= 3:
            return "contextual"
        if score >= 1:
            return "weak"
        return "none"
    if "direct target evidence: yes" in value_l:
        return "direct"
    for relevance in ("direct", "contextual", "weak", "none"):
        if value_l.startswith(relevance) or f"{relevance} -" in value_l:
            return relevance
    return "none"


def normalize_direction_value(value: str) -> str:
    value_l = clean_one_line(value).lower()
    for level in ("high", "medium", "low"):
        if value_l.startswith(level) or f"{level} -" in value_l:
            return level
    return "low"


def normalize_score(value: str) -> int:
    match = re.search(r"\b([0-5])\b", clean_one_line(value))
    return int(match.group(1)) if match else 0


def normalize_direct_target(value: str) -> str:
    value_l = clean_one_line(value).lower()
    if value_l.startswith("yes"):
        return "yes"
    if value_l.startswith("no"):
        return "no"
    return "no"


def target_terms(target_prompt: str | None) -> list[str]:
    target_l = clean_one_line(target_prompt).lower()
    terms = []
    if target_l:
        terms.append(target_l)
    for token in re.findall(r"[a-z0-9_]+", target_l):
        if len(token) >= 3 and token not in terms:
            terms.append(token)
    return terms


def text_mentions_target(text: str, target_prompt: str | None) -> bool:
    text_l = clean_one_line(text).lower()
    return any(term and term in text_l for term in target_terms(target_prompt))


def validate_direct_target_evidence(
    evidence: str,
    target_prompt: str | None,
    *visible_fields: str,
) -> tuple[str, bool]:
    evidence = clean_generated_field(evidence, "no")
    visible_has_target = any(text_mentions_target(field, target_prompt) for field in visible_fields)

    if normalize_direct_target(evidence) != "yes":
        if visible_has_target:
            return f"yes - {clean_one_line(target_prompt)} visible in visual description", True
        return evidence, False

    evidence_has_target = text_mentions_target(evidence, target_prompt)
    if evidence_has_target or visible_has_target:
        return evidence, True

    return "no - target not explicitly identified", False


def derive_navigation_recommendation(open_routes: str, blocked_routes: str) -> str:
    open_l = clean_one_line(open_routes).lower()
    blocked_l = clean_one_line(blocked_routes).lower()
    if open_l and open_l != "none":
        return f"promising - visible open route: {open_routes}"
    if blocked_l and blocked_l != "none":
        return f"blocked - {blocked_routes}"
    return "weak - no visible open route"


def derive_task_relevance_score(direct_target_visible: bool, open_routes: str) -> int:
    if direct_target_visible:
        return 5
    open_l = clean_one_line(open_routes).lower()
    if open_l and open_l != "none":
        return 2
    return 0


def normalize_recommendation_level(value: str) -> str:
    value_l = clean_one_line(value).lower()
    for level in ("promising", "weak", "blocked", "irrelevant"):
        if level in value_l:
            return level
    return "weak"


def looks_like_prompt_template(value: str) -> bool:
    value_l = clean_one_line(value).lower()
    template_fragments = (
        "one short sentence",
        "comma-separated",
        "maximum 10 items",
        "max 8",
        "max 6",
        "direct/contextual/weak/none",
        "short reason using only visible evidence",
        "visible open doorway/corridor/hallway",
        "visible floor/path/door/corridor",
        "or none; say closed door",
        "no strong direction cue",
    )
    return any(fragment in value_l for fragment in template_fragments)


def clean_generated_field(value: str, default: str) -> str:
    value = clean_one_line(value)
    if not value or looks_like_prompt_template(value):
        return default
    return value


def is_none_like(value: str) -> bool:
    value_l = clean_one_line(value).lower().strip(" .")
    if not value_l:
        return True
    if value_l in {"none", "unknown", "n/a", "not visible"}:
        return True
    none_fragments = (
        "no visible",
        "no open",
        "no clear",
        "none visible",
        "not visible",
        "no blocked",
        "no unclear",
    )
    return any(fragment in value_l for fragment in none_fragments)


def clean_route_field(value: str, default: str = "none") -> str:
    value = clean_generated_field(value, default)
    value_l = value.lower()
    instruction_fragments = (
        "visible entrance, exit, doorway",
        "courtyard, or facade",
        "closed doors, walls, obstacles",
        "unclear paths, or none",
        "physically open visible routes",
    )
    if is_none_like(value) or any(fragment in value_l for fragment in instruction_fragments):
        return default
    return value


def first_sentence(text: str) -> str:
    text = clean_one_line(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return parts[0].strip()


def infer_scene_type(text: str) -> str:
    text_l = clean_one_line(text).lower()
    if any(word in text_l for word in ("outdoor", "outside", "street", "sidewalk", "courtyard", "parking")):
        return "outdoor"
    if any(word in text_l for word in ("entrance", "exit", "doorway", "threshold", "lobby")):
        return "transition"
    if any(word in text_l for word in ("indoor", "inside", "room", "classroom", "office", "hallway", "lab", "laboratory")):
        return "indoor"
    return "unknown"


def infer_visible_objects(text: str) -> str:
    text_l = clean_one_line(text).lower()
    object_terms = [
        "whiteboard", "blackboard", "door", "cabinet", "desk", "table", "chair",
        "monitor", "computer", "keyboard", "mouse", "window", "radiator",
        "wall", "sign", "panel", "fire extinguisher", "person", "laptop",
        "tank", "machinery",
    ]
    found = []
    for term in object_terms:
        if re.search(rf"\b{re.escape(term)}s?\b", text_l) and term not in found:
            found.append(term)
    return ", ".join(found) if found else "none"


def clean_object_list(value: str) -> str:
    value = clean_generated_field(value, "none")
    if is_none_like(value):
        return "none"
    parts = []
    seen = set()
    for part in re.split(r",|;", value):
        item = clean_one_line(part).strip(" .")
        if not item:
            continue
        item_l = item.lower()
        if item_l in {"objects", "visible objects", "concrete objects", "scene", "none"}:
            continue
        if looks_like_prompt_template(item_l):
            continue
        if item_l not in seen:
            seen.add(item_l)
            parts.append(item)
        if len(parts) >= 10:
            break
    return ", ".join(parts) if parts else "none"


def infer_open_routes(text: str) -> str:
    text_l = clean_one_line(text).lower()
    if any(fragment in text_l for fragment in ("open doorway", "open door", "corridor", "hallway", "open passage", "open entrance")):
        return "visible open route"
    return "none"


def infer_blocked_routes(text: str) -> str:
    text_l = clean_one_line(text).lower()
    if any(fragment in text_l for fragment in ("closed door", "closed doors", "wall", "obstacle", "blocked")):
        return "closed or blocked route visible"
    return "none"


def describe_image(model, image_path: Path, task: str | None = None, target_prompt: str | None = None) -> dict:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    model.generation_config.max_new_tokens = 220
    answer = query_moondream_answer(
        model,
        image,
        build_single_prompt(task=task, target_prompt=target_prompt),
    )
    fields = parse_labeled_lines(answer)
    objects_answer = ""

    summary = clean_generated_field(fields.get("summary", ""), "")
    fallback_scene = summary or first_sentence(answer)
    scene = clean_generated_field(fields.get("scene", ""), fallback_scene or "unknown visible scene")
    scene_type = normalize_scene_type(fields.get("scene_type", "unknown"))
    if scene_type == "unknown":
        scene_type = infer_scene_type(scene)
    visible_objects = clean_object_list(fields.get("visible_objects", ""))
    if visible_objects == "none":
        visible_objects = infer_visible_objects("\n".join([scene, summary, answer]))
    distinctive_landmarks = clean_generated_field(fields.get("distinctive_landmarks", ""), "none")
    open_traversable_routes = clean_route_field(
        fields.get("open_traversable_routes", fields.get("traversable_openings", "")),
        "none",
    )
    if open_traversable_routes == "none":
        open_traversable_routes = infer_open_routes("\n".join([scene, summary]))
    blocked_or_unclear_routes = clean_route_field(fields.get("blocked_or_unclear_routes", ""), "none")
    if blocked_or_unclear_routes == "none":
        blocked_or_unclear_routes = infer_blocked_routes("\n".join([scene, summary]))
    indoor_outdoor_transition_evidence = clean_route_field(
        fields.get("indoor_outdoor_transition_evidence", ""),
        "none",
    )
    direct_target_evidence, direct_target_visible = validate_direct_target_evidence(
        "no",
        target_prompt,
        scene,
        visible_objects,
        distinctive_landmarks,
    )
    useful_context_for_task = "none"
    if direct_target_visible:
        useful_context_for_task = (
            f"{clean_one_line(target_prompt)} appears in visible objects, landmarks, or scene"
        )
    navigation_recommendation = derive_navigation_recommendation(
        open_traversable_routes,
        blocked_or_unclear_routes,
    )
    task_relevance_score = derive_task_relevance_score(
        direct_target_visible,
        open_traversable_routes,
    )

    navigation_cues = fields.get("navigation_cues")
    if not navigation_cues:
        navigation_cues = (
            f"open routes: {open_traversable_routes}; "
            f"blocked or unclear routes: {blocked_or_unclear_routes}; "
            f"recommendation: {navigation_recommendation}"
    )
    traversable_openings = open_traversable_routes
    task_relevance = (
        f"score {task_relevance_score}/5; "
        f"direct target evidence: {direct_target_evidence}; "
        f"useful context: {useful_context_for_task}"
    )
    best_direction_cue = navigation_recommendation

    description = (
        f"Scene type: {scene_type}\n"
        f"Scene: {scene}\n"
        f"Visible objects: {visible_objects}\n"
        f"Distinctive landmarks: {distinctive_landmarks}\n"
        f"Open traversable routes: {open_traversable_routes}\n"
        f"Blocked or unclear routes: {blocked_or_unclear_routes}\n"
        f"Indoor/outdoor transition evidence: {indoor_outdoor_transition_evidence}\n"
        f"Direct target evidence: {direct_target_evidence}\n"
        f"Useful context for the task: {useful_context_for_task}\n"
        f"Navigation recommendation for this view: {navigation_recommendation}\n"
        f"Task relevance score: {task_relevance_score}"
    )

    return {
        "scene": scene,
        "scene_type": scene_type,
        "visible_objects": visible_objects,
        "distinctive_landmarks": distinctive_landmarks,
        "open_traversable_routes": open_traversable_routes,
        "blocked_or_unclear_routes": blocked_or_unclear_routes,
        "indoor_outdoor_transition_evidence": indoor_outdoor_transition_evidence,
        "direct_target_evidence": direct_target_evidence,
        "direct_target_visible": direct_target_visible,
        "useful_context_for_task": useful_context_for_task,
        "navigation_recommendation": navigation_recommendation,
        "navigation_recommendation_level": normalize_recommendation_level(navigation_recommendation),
        "task_relevance_score": task_relevance_score,
        "navigation_cues": navigation_cues,
        "traversable_openings": traversable_openings,
        "task_relevance": task_relevance,
        "task_relevance_level": normalize_relevance(task_relevance),
        "best_direction_cue": best_direction_cue,
        "best_direction_level": normalize_direction_value(best_direction_cue),
        "raw_answer": answer,
        "raw_objects_answer": objects_answer,
        "description": description,
    }


def load_moondream_model():
    from transformers import AutoModelForCausalLM

    print("Loading Moondream model...")
    model = AutoModelForCausalLM.from_pretrained(
        f"moondream/{MODEL_ID}",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.2
    model.generation_config.no_repeat_ngram_size = 2
    return model


def semantic_description_payload(desc: dict) -> dict:
    return {
        "scene": desc["scene"],
        "scene_type": desc["scene_type"],
        "visible_objects": desc["visible_objects"],
        "distinctive_landmarks": desc["distinctive_landmarks"],
        "open_traversable_routes": desc["open_traversable_routes"],
        "blocked_or_unclear_routes": desc["blocked_or_unclear_routes"],
        "indoor_outdoor_transition_evidence": desc["indoor_outdoor_transition_evidence"],
        "direct_target_evidence": desc["direct_target_evidence"],
        "direct_target_visible": desc["direct_target_visible"],
        "useful_context_for_task": desc["useful_context_for_task"],
        "navigation_recommendation": desc["navigation_recommendation"],
        "navigation_recommendation_level": desc["navigation_recommendation_level"],
        "task_relevance_score": desc["task_relevance_score"],
        "navigation_cues": desc["navigation_cues"],
        "traversable_openings": desc["traversable_openings"],
        "task_relevance": desc["task_relevance"],
        "task_relevance_level": desc["task_relevance_level"],
        "best_direction_cue": desc["best_direction_cue"],
        "best_direction_level": desc["best_direction_level"],
        "raw_answer": desc["raw_answer"],
        "raw_objects_answer": desc["raw_objects_answer"],
        "model": MODEL_ID,
    }


def color_image_from_snapshot(snapshot: dict) -> str | None:
    image_stamps = snapshot.get("image_stamps", {})
    color_stamp = image_stamps.get("realsense_color", {})
    color_artifacts = color_stamp.get("artifacts", {})
    if color_artifacts.get("png"):
        return color_artifacts["png"]

    for filename in snapshot.get("filenames", []):
        lower_name = str(filename).lower()
        if lower_name.startswith("c") and lower_name.endswith((".png", ".jpg", ".jpeg")):
            return filename

    return None


def resolve_mapped_path(path: Path, path_maps: list[tuple[Path, Path]] | None = None) -> Path:
    if path.exists():
        return path

    path_str = str(path)
    for source_prefix, target_prefix in path_maps or []:
        source_str = str(source_prefix)
        if path_str == source_str or path_str.startswith(source_str + "/"):
            suffix = path_str[len(source_str):].lstrip("/")
            mapped = target_prefix / suffix
            if mapped.exists():
                return mapped

    return path


def parse_path_maps(values: list[str] | None) -> list[tuple[Path, Path]]:
    path_maps = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --path-map {value!r}. Expected FROM=TO.")
        source, target = value.split("=", 1)
        path_maps.append((Path(source), Path(target)))
    return path_maps


class PathDateTime:
    @staticmethod
    def now() -> str:
        from datetime import datetime

        return datetime.now().isoformat()


def describe_node_dir(
    model,
    node_dir: Path,
    overwrite: bool = False,
    path_maps=None,
    write_metadata: bool = True,
    task: str | None = None,
    target_prompt: str | None = None,
) -> dict:
    requested_node_dir = node_dir
    node_dir = resolve_mapped_path(node_dir, path_maps)
    metadata_path = node_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {requested_node_dir} "
            f"(resolved to {node_dir})"
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    processed = 0
    skipped = 0
    missing = 0
    errors = []
    described_snapshots = []

    for snapshot in metadata.get("snapshots", []):
        if snapshot.get("description") and not overwrite:
            skipped += 1
            continue

        filename = color_image_from_snapshot(snapshot)
        if not filename:
            missing += 1
            errors.append({
                "snapshot": snapshot.get("label"),
                "error": "No color image filename found",
            })
            continue

        image_path = node_dir / filename
        if not image_path.exists():
            missing += 1
            errors.append({
                "snapshot": snapshot.get("label"),
                "image_path": str(image_path),
                "error": "Image file does not exist",
            })
            continue

        desc = describe_image(
            model,
            image_path,
            task=task or metadata.get("task") or metadata.get("user_prompt"),
            target_prompt=target_prompt or metadata.get("target_prompt"),
        )
        semantic_description = semantic_description_payload(desc)
        described_at = PathDateTime.now()
        snapshot["description"] = desc["description"]
        snapshot["semantic_description"] = semantic_description
        snapshot["description_model"] = MODEL_ID
        snapshot["description_prompt_mode"] = PROMPT_MODE
        snapshot["described_at"] = described_at
        described_snapshots.append({
            "label": snapshot.get("label"),
            "stop_index": snapshot.get("stop_index"),
            "stop_name": snapshot.get("stop_name"),
            "description": desc["description"],
            "semantic_description": semantic_description,
            "description_model": MODEL_ID,
            "description_prompt_mode": PROMPT_MODE,
            "described_at": described_at,
        })
        processed += 1

    descriptions_status = {
        "status": "completed" if not errors else "completed_with_errors",
        "updated_at": PathDateTime.now(),
        "model": MODEL_ID,
        "prompt_mode": PROMPT_MODE,
        "processed": processed,
        "skipped": skipped,
        "missing": missing,
        "errors": errors,
    }
    metadata["descriptions"] = descriptions_status

    if write_metadata:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "ok": True,
        "requested_node_dir": str(requested_node_dir),
        "node_dir": str(node_dir),
        "metadata_path": str(metadata_path),
        "wrote_metadata": write_metadata,
        "processed": processed,
        "skipped": skipped,
        "missing": missing,
        "errors": errors,
        "descriptions": descriptions_status,
        "described_snapshots": described_snapshots,
    }


class MoondreamDescriptionServer:
    def __init__(self, path_maps=None, task=None, target_prompt=None):
        self.path_maps = path_maps or []
        self.task = task
        self.target_prompt = target_prompt
        self.model = load_moondream_model()

    def describe_image_path(self, image_path: Path, task=None, target_prompt=None) -> dict:
        image_path = resolve_mapped_path(image_path, self.path_maps)
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        desc = describe_image(
            self.model,
            image_path,
            task=task or self.task,
            target_prompt=target_prompt or self.target_prompt,
        )
        return {
            "ok": True,
            "image_path": str(image_path),
            "description": desc["description"],
            "semantic_description": semantic_description_payload(desc),
        }

    def describe_node(
        self,
        node_dir: Path,
        overwrite: bool = False,
        write_metadata: bool = False,
        task=None,
        target_prompt=None,
    ) -> dict:
        return describe_node_dir(
            self.model,
            node_dir=node_dir,
            overwrite=overwrite,
            path_maps=self.path_maps,
            write_metadata=write_metadata,
            task=task or self.task,
            target_prompt=target_prompt or self.target_prompt,
        )


class MoondreamRequestHandler(BaseHTTPRequestHandler):
    server_state: MoondreamDescriptionServer = None

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body) if body else {}

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "ok": True,
                "model": MODEL_ID,
                "prompt_mode": PROMPT_MODE,
                "endpoints": ["/health", "/prompt", "/describe_image", "/describe_node"],
            })
            return
        if self.path == "/prompt":
            self._send_json(200, {
                "ok": True,
                "model": MODEL_ID,
                "prompt_mode": PROMPT_MODE,
                "prompt": build_single_prompt(
                    task=self.server_state.task,
                    target_prompt=self.server_state.target_prompt,
                ),
            })
            return

        self._send_json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        try:
            payload = self._read_json()

            if self.path == "/describe_image":
                image_path = Path(payload["image_path"])
                response = self.server_state.describe_image_path(
                    image_path,
                    task=payload.get("task") or payload.get("user_prompt"),
                    target_prompt=payload.get("target_prompt"),
                )
                self._send_json(200, response)
                return

            if self.path == "/describe_node":
                node_dir = Path(payload["node_dir"])
                overwrite = bool(payload.get("overwrite", False))
                write_metadata = bool(payload.get("write_metadata", False))
                response = self.server_state.describe_node(
                    node_dir,
                    overwrite=overwrite,
                    write_metadata=write_metadata,
                    task=payload.get("task") or payload.get("user_prompt"),
                    target_prompt=payload.get("target_prompt"),
                )
                self._send_json(200, response)
                return

            self._send_json(404, {"ok": False, "error": "unknown endpoint"})

        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            })


def serve(host: str, port: int, path_maps=None, task=None, target_prompt=None):
    MoondreamRequestHandler.server_state = MoondreamDescriptionServer(
        path_maps=path_maps,
        task=task,
        target_prompt=target_prompt,
    )
    httpd = ThreadingHTTPServer((host, port), MoondreamRequestHandler)
    print(f"Moondream single-prompt server listening on http://{host}:{port}")
    if path_maps:
        print("Path maps:")
        for source, target in path_maps:
            print(f"  {source} -> {target}")
    if task or target_prompt:
        print(f"Default task: {task or target_prompt}")
    print("Endpoints: GET /health, POST /describe_image, POST /describe_node")
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run as a Moondream HTTP description server")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument("--task", default=None, help="Optional default task used when requests do not include one")
    parser.add_argument("--target-prompt", default=None, help="Optional default target phrase")
    parser.add_argument(
        "--path-map",
        action="append",
        default=None,
        help="Translate paths received from Docker before opening files. Can be repeated. Format: FROM=TO",
    )
    args = parser.parse_args()
    if args.path_map is None:
        args.path_map = [
            "/rgbd_camera_intel_dev/src=/home/vladkobli/rocon-demos/jazzy-full/src",
        ]

    if not args.server:
        parser.error("add_descriptions2.py currently supports server mode only; pass --server")

    serve(
        args.host,
        args.port,
        path_maps=parse_path_maps(args.path_map),
        task=args.task,
        target_prompt=args.target_prompt,
    )


if __name__ == "__main__":
    main()
