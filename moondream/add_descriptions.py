from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM
import argparse
import json
import re


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
    image = Image.open(image_path).convert("RGB")

    model.generation_config.max_new_tokens = 35
    scene = model.query(image, build_scene_prompt())["answer"]
    scene = clean_prefixed_line(scene, ["scene"])

    model.generation_config.max_new_tokens = 35
    visible_objects = model.query(image, build_objects_prompt())["answer"]
    visible_objects = clean_prefixed_line(
        visible_objects,
        ["visible_objects", "visible objects", "objects"]
    )    
    visible_objects = dedupe_comma_list(visible_objects, max_items=8)

    model.generation_config.max_new_tokens = 40
    landmarks = model.query(image, build_landmarks_prompt())["answer"]
    landmarks = clean_prefixed_line(
        landmarks,
        ["distinctive_landmarks", "distinctive landmarks", "landmarks"]
    )
    landmarks = dedupe_comma_list(landmarks, max_items=8)

    model.generation_config.max_new_tokens = 35
    navigation_cues = model.query(image, build_navigation_prompt())["answer"]
    navigation_cues = clean_prefixed_line(
        navigation_cues,
        ["navigation_cues", "navigation cues", "navigation"]
    )
    
    model.generation_config.max_new_tokens = 50
    traversable_openings  = model.query(image, build_traversable_openings_prompt())["answer"]
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
    scene_type = model.query(image, build_scene_type_prompt())["answer"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-json", required=True, help="Input graph JSON, e.g. nodes_graph_8_color.json")
    parser.add_argument("--dataset-root", default=None, help="Root folder containing node image folders, e.g. /rgbd_camera_intel_dev/src/nodes")
    parser.add_argument("--output", default=None, help="Output JSON path. Default: <input>_described.json")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate descriptions even if they already exist")
    parser.add_argument("--max-nodes", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    graph_json_path = Path(args.graph_json)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    with open(graph_json_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    source_dataset = graph_data.get("meta", {}).get("source_dataset")

    print("Loading Moondream model...")
    model = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream-2b-2025-04-14-4bit",
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = 1.35
    model.generation_config.no_repeat_ngram_size = 2

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
            image_entry["semantic_description"] = {
                "scene": desc["scene"],
                "visible_objects": desc["visible_objects"],
                "distinctive_landmarks": desc["distinctive_landmarks"],
                "navigation_cues": desc["navigation_cues"],
                "traversable_openings": desc["traversable_openings"],
                "scene_type": desc["scene_type"],
                "model": "moondream-2b-2025-04-14-4bit"
            }

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