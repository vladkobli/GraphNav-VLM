from pathlib import Path
import argparse
import json
import re
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


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


def clean_one_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()


def clean_prefixed_line(text: str, prefixes: list[str]) -> str:
    text = clean_one_line(text)

    for prefix in prefixes:
        text = re.sub(
            rf"^\s*{prefix}\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE
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


def query_moondream_answer(model, image, prompt: str) -> str:
    result = model.query(image, prompt)

    if isinstance(result, dict):
        if "answer" not in result:
            raise KeyError(f"Moondream query response has no 'answer' key: {result!r}")
        return str(result["answer"])

    if isinstance(result, str):
        return result

    raise TypeError(f"Unexpected Moondream query response type: {type(result).__name__}")


def is_color_image_entry(image_entry: dict) -> bool:
    """
    Keeps color images and skips depth images.
    Works both for your current color-only JSON and possible mixed JSONs.
    """
    image_path = str(image_entry.get("image_path", "")).lower()
    frame = str(image_entry.get("frame", "")).lower()

    if "depth" in image_path or "depth" in frame:
        return False

    capture = image_entry.get("capture", {})
    stamp = capture.get("image_stamp", {})
    encoding = str(stamp.get("encoding", "")).lower()

    if encoding:
        return encoding in {"rgb8", "bgr8", "rgba8", "bgra8"}

    return image_path.endswith((".png", ".jpg", ".jpeg"))


def resolve_image_path(image_entry: dict, graph_json_path: Path, dataset_root: Path | None, source_dataset: str | None) -> Path | None:
    rel_path = image_entry.get("image_path") or image_entry.get("frame")

    if not rel_path:
        return None

    rel_path = Path(rel_path)

    candidates = []

    if dataset_root is not None:
        candidates.append(dataset_root / rel_path)

    if source_dataset:
        candidates.append(Path(source_dataset) / rel_path)

    candidates.append(graph_json_path.parent / rel_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def describe_image(model, image_path: Path) -> dict:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    model.generation_config.max_new_tokens = 35
    scene = query_moondream_answer(model, image, build_scene_prompt())
    scene = clean_prefixed_line(scene, ["scene"])

    model.generation_config.max_new_tokens = 35
    visible_objects = query_moondream_answer(model, image, build_objects_prompt())
    visible_objects = clean_prefixed_line(
        visible_objects,
        ["visible_objects", "visible objects", "objects"]
    )    
    visible_objects = dedupe_comma_list(visible_objects, max_items=8)

    model.generation_config.max_new_tokens = 40
    landmarks = query_moondream_answer(model, image, build_landmarks_prompt())
    landmarks = clean_prefixed_line(
        landmarks,
        ["distinctive_landmarks", "distinctive landmarks", "landmarks"]
    )
    landmarks = dedupe_comma_list(landmarks, max_items=8)

    model.generation_config.max_new_tokens = 35
    navigation_cues = query_moondream_answer(model, image, build_navigation_prompt())
    navigation_cues = clean_prefixed_line(
        navigation_cues,
        ["navigation_cues", "navigation cues", "navigation"]
    )
    
    model.generation_config.max_new_tokens = 50
    traversable_openings  = query_moondream_answer(model, image, build_traversable_openings_prompt())
    traversable_openings = clean_prefixed_line(
        traversable_openings,
        [
            "traversable_openings",
            "traversable openings",
            "visible traversable openings",
            "openings",
            "routes"
        ]
    )
    
    model.generation_config.max_new_tokens = 20
    scene_type = query_moondream_answer(model, image, build_scene_type_prompt())
    scene_type = clean_prefixed_line(
        scene_type,
        ["scene_type", "scene type", "type"]
    )

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
        "description": description
    }


def load_moondream_model():
    from transformers import AutoModelForCausalLM

    print("Loading Moondream model...")
    model = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream-2b-2025-04-14-4bit",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.35
    model.generation_config.no_repeat_ngram_size = 2
    return model


def semantic_description_payload(desc: dict) -> dict:
    return {
        "scene": desc["scene"],
        "visible_objects": desc["visible_objects"],
        "distinctive_landmarks": desc["distinctive_landmarks"],
        "navigation_cues": desc["navigation_cues"],
        "traversable_openings": desc["traversable_openings"],
        "scene_type": desc["scene_type"],
        "model": "moondream-2b-2025-04-14-4bit",
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


def describe_node_dir(
    model,
    node_dir: Path,
    overwrite: bool = False,
    path_maps=None,
    write_metadata: bool = True,
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

        desc = describe_image(model, image_path)
        semantic_description = semantic_description_payload(desc)
        described_at = PathDateTime.now()
        snapshot["description"] = desc["description"]
        snapshot["semantic_description"] = semantic_description
        snapshot["description_model"] = "moondream-2b-2025-04-14-4bit"
        snapshot["described_at"] = described_at
        described_snapshots.append({
            "label": snapshot.get("label"),
            "stop_index": snapshot.get("stop_index"),
            "stop_name": snapshot.get("stop_name"),
            "description": desc["description"],
            "semantic_description": semantic_description,
            "description_model": "moondream-2b-2025-04-14-4bit",
            "described_at": described_at,
        })
        processed += 1

    descriptions_status = {
        "status": "completed" if not errors else "completed_with_errors",
        "updated_at": PathDateTime.now(),
        "model": "moondream-2b-2025-04-14-4bit",
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


class PathDateTime:
    @staticmethod
    def now() -> str:
        from datetime import datetime

        return datetime.now().isoformat()


class MoondreamDescriptionServer:
    def __init__(self, path_maps=None):
        self.path_maps = path_maps or []
        self.model = load_moondream_model()

    def describe_image_path(self, image_path: Path) -> dict:
        image_path = resolve_mapped_path(image_path, self.path_maps)
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        desc = describe_image(self.model, image_path)
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
    ) -> dict:
        return describe_node_dir(
            self.model,
            node_dir=node_dir,
            overwrite=overwrite,
            path_maps=self.path_maps,
            write_metadata=write_metadata,
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
                "model": "moondream-2b-2025-04-14-4bit",
                "endpoints": ["/health", "/describe_image", "/describe_node"],
            })
            return

        self._send_json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self):
        try:
            payload = self._read_json()

            if self.path == "/describe_image":
                image_path = Path(payload["image_path"])
                response = self.server_state.describe_image_path(image_path)
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


def serve(host: str, port: int, path_maps=None):
    MoondreamRequestHandler.server_state = MoondreamDescriptionServer(path_maps=path_maps)
    httpd = ThreadingHTTPServer((host, port), MoondreamRequestHandler)
    print(f"Moondream description server listening on http://{host}:{port}")
    if path_maps:
        print("Path maps:")
        for source, target in path_maps:
            print(f"  {source} -> {target}")
    print("Endpoints: GET /health, POST /describe_image, POST /describe_node")
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true", help="Run as a Moondream HTTP description server")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8766, help="Server port")
    parser.add_argument(
        "--path-map",
        action="append",
        default=None,
        help="Translate paths received from Docker before opening files. Can be repeated. Format: FROM=TO",
    )
    parser.add_argument("--graph-json", default=None, help="Input graph JSON, e.g. nodes_graph_8_color.json")
    parser.add_argument("--dataset-root", default=None, help="Root folder containing node image folders, e.g. /rgbd_camera_intel_dev/src/nodes")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: <input>_described.json")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate descriptions even if they already exist")
    parser.add_argument("--max-nodes", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()
    if args.path_map is None:
        args.path_map = [
            "/rgbd_camera_intel_dev/src=/home/vladkobli/rocon-demos/jazzy-full/src",
        ]

    if args.server:
        serve(args.host, args.port, path_maps=parse_path_maps(args.path_map))
        return

    if not args.graph_json:
        parser.error("--graph-json is required unless --server is used")

    graph_json_path = Path(args.graph_json)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    source_dataset = graph_data.get("meta", {}).get("source_dataset")

    model = load_moondream_model()

    nodes = graph_data.get("nodes", [])

    if args.max_nodes is not None:
        nodes = nodes[:args.max_nodes]

    total_processed = 0
    total_skipped = 0
    total_missing = 0

    for node in nodes:
        node_id = node.get("id", "unknown_node")
        images = node.get("images", [])

        print(f"\nNode {node_id}: {len(images)} image entries")

        for image_entry in sorted(images, key=lambda x: x.get("stop_index", 999)):
            if not is_color_image_entry(image_entry):
                total_skipped += 1
                continue

            existing_description = image_entry.get("description", "").strip()
            if existing_description and not args.overwrite:
                print(f"  Skipping {image_entry.get('image_id')} because description already exists")
                total_skipped += 1
                continue

            image_file = resolve_image_path(
                image_entry=image_entry,
                graph_json_path=graph_json_path,
                dataset_root=dataset_root,
                source_dataset=source_dataset
            )

            if image_file is None:
                print(f"  Missing image: {image_entry.get('image_path')}")
                total_missing += 1
                continue

            image_id = image_entry.get("image_id", image_file.name)
            camera_id = image_entry.get("camera_id", "unknown")
            rel_yaw = image_entry.get("relative_yaw_deg")
            abs_yaw = image_entry.get("heading_abs_deg")

            print(f"  Processing {image_id} | camera={camera_id} | rel_yaw={rel_yaw} | abs_yaw={abs_yaw}")

            desc = describe_image(model, image_file)

            image_entry["description"] = desc["description"]
            image_entry["semantic_description"] = semantic_description_payload(desc)

            total_processed += 1

    output_path = Path(args.output) if args.output else graph_json_path.with_name(
        graph_json_path.stem + "_described.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Processed color images: {total_processed}")
    print(f"Skipped entries: {total_skipped}")
    print(f"Missing images: {total_missing}")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
