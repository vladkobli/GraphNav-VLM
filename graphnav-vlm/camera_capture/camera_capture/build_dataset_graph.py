#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Graph topology
# -----------------------------------------------------------------------------

def n(i: int) -> str:
    return f"n{i:02d}"

# lost oudoor dataset
# NEIGHBORS_INT = {
#     1: [2],
#     2: [1, 3],
#     3: [2, 4],
#     4: [3, 5],
#     5: [4, 6, 11, 15],
#     6: [5, 7],
#     7: [6, 8],
#     8: [7, 9],
#     9: [8],
#     11: [5, 12, 14, 15],
#     12: [11, 13],
#     13: [12],
#     14: [11, 15],
#     15: [5, 11, 14, 16],
#     16: [15, 17, 19],
#     17: [16, 18],
#     18: [17],
#     19: [16, 20],
#     20: [19, 21, 24],
#     21: [20, 22, 23],
#     22: [21],
#     23: [21, 24, 41],
#     24: [20, 23, 25, 41],
#     25: [24, 40, 41],
#     26: [27, 32, 40],
#     27: [26, 28],
#     28: [27, 29],
#     29: [28, 30],
#     30: [29, 31, 32],
#     31: [30],
#     32: [26, 30, 33],
#     33: [32, 34],
#     34: [33, 35],
#     35: [34, 36],
#     36: [35, 37, 38],
#     37: [36, 38],
#     38: [36, 37, 39],
#     39: [38],
#     40: [25, 26],
#     41: [23, 24, 25, 42],
#     42: [41, 43],
#     43: [42, 44, 45],
#     44: [43, 45, 46],
#     45: [43, 44, 46, 48],
#     46: [44, 45, 47, 48],
#     47: [46, 48],
#     48: [45, 46, 47, 49, 50],
#     49: [48, 50],
#     50: [48, 49],
# }

# new dataset
NEIGHBORS_INT = {
    1: [2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 6, 7, 11, 12],
    6: [5, 7, 11],
    7: [5, 6, 8, 11],
    8: [7, 9],
    9: [8, 10],
    10: [9],
    11: [5, 6, 7, 12, 21],
    12: [5, 11, 13, 16, 17, 21],
    13: [12, 14],
    14: [13, 15],
    15: [14],
    16: [12, 17, 20, 21],
    17: [12, 16, 18, 20],
    18: [17, 19],
    19: [18],
    20: [16, 17],
    21: [11, 12, 16, 22],
    22: [21, 23, 25],
    23: [22, 24],
    24: [23],
    25: [22, 26],
    26: [25, 27],
    27: [26, 28, 30],
    28: [27, 29, 32, 35],
    29: [28, 30, 32],
    30: [27, 29, 31],
    31: [30],
    32: [28, 29, 33, 35],
    33: [32, 34],
    34: [33, 57],
    35: [28, 32, 36],
    36: [35, 37],
    37: [36, 38, 56],
    38: [37, 39],
    39: [38, 40, 52],
    40: [39, 41],
    41: [40, 42],
    42: [41, 43, 48],
    43: [42, 44],
    44: [43, 45, 46],
    45: [44],
    46: [44, 47],
    47: [46],
    48: [42, 49, 51],
    49: [48, 50],
    50: [49, 51, 53],
    51: [48, 50, 52],
    52: [39, 51, 53],
    53: [50, 52, 54],
    54: [53, 55],
    55: [54, 56],
    56: [37, 55],
    57: [34, 58],
    58: [57, 59],
    59: [58, 60, 61],
    60: [59, 61],
    61: [59, 60, 62, 64],
    62: [61, 63, 64],
    63: [62],
    64: [61, 62, 65],
    65: [64, 66, 67],
    66: [65, 67],
    67: [65, 66, 68],
    68: [67],
}

NEIGHBORS = {n(k): [n(v) for v in vals] for k, vals in NEIGHBORS_INT.items()}


# -----------------------------------------------------------------------------
# Camera / pan-stop definitions
# -----------------------------------------------------------------------------
# Relative yaw convention:
#   robot front = 0 deg
#   left = +90 deg
#   right = -90 deg
#   heading_abs_deg = node_yaw_deg + relative_yaw_deg, normalized to [0, 360)
#
# Your 8 captured color frames are:
#   c1_back.png, c2_back_left.png, ..., c8_back_right.png
# Depth frames d*.png are ignored by this graph builder.

CAMERA_STOPS = [
    {"stop_index": 1, "stop_name": "back",        "relative_yaw_deg": 180.0},
    {"stop_index": 2, "stop_name": "back_left",   "relative_yaw_deg": 135.0},
    {"stop_index": 3, "stop_name": "left",        "relative_yaw_deg": 90.0},
    {"stop_index": 4, "stop_name": "front_left",  "relative_yaw_deg": 45.0},
    {"stop_index": 5, "stop_name": "front",       "relative_yaw_deg": 0.0},
    {"stop_index": 6, "stop_name": "front_right", "relative_yaw_deg": -45.0},
    {"stop_index": 7, "stop_name": "right",       "relative_yaw_deg": -90.0},
    {"stop_index": 8, "stop_name": "back_right",  "relative_yaw_deg": -135.0},
]

STOP_BY_INDEX = {entry["stop_index"]: entry for entry in CAMERA_STOPS}
STOP_BY_NAME = {entry["stop_name"]: entry for entry in CAMERA_STOPS}

CARDINAL_STOP_ORDER = ["front", "left", "back", "right"]

# Static transform from panther/base_link to the RealSense camera origin,
# before applying each virtual pan direction around z.
CAMERA_TRANSLATION_IN_BASE = {
    "x": -0.125,
    "y": 0.02,
    "z": 1.0,
}
CAMERA_STATIC_RPY_IN_BASE_DEG = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
}


# -----------------------------------------------------------------------------
# Math / parsing helpers
# -----------------------------------------------------------------------------

def normalize_deg(angle: float) -> float:
    return angle % 360.0


def normalize_rad(angle: float) -> float:
    """Normalize to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rad_to_deg(rad: float) -> float:
    return math.degrees(rad)


def deg_to_rad(deg: float) -> float:
    return math.radians(deg)


def safe_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def node_sort_key(path: Path) -> int:
    match = re.fullmatch(r"n(\d+)", path.name.lower())
    if not match:
        return 10**9
    return int(match.group(1))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_node_pose(metadata: Dict[str, Any], node_name: str) -> Dict[str, Any]:
    """
    Prefer the top-level planned_pose. If missing, fall back to the first
    snapshot.actual_robot_pose that contains x/y/yaw.
    """
    pose = metadata.get("planned_pose")
    source = "planned_pose"

    if not (isinstance(pose, dict) and all(k in pose for k in ("x", "y", "yaw"))):
        pose = None
        source = "snapshot_actual_robot_pose"
        for snapshot in metadata.get("snapshots", []):
            candidate = snapshot.get("actual_robot_pose")
            if isinstance(candidate, dict) and all(k in candidate for k in ("x", "y", "yaw")):
                pose = candidate
                break

    if pose is None:
        raise ValueError(f"{node_name} has no planned_pose or snapshot actual_robot_pose")

    yaw_rad = float(pose["yaw"])
    yaw_deg = normalize_deg(rad_to_deg(yaw_rad))

    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose.get("z", 0.0)),
        "yaw_rad": yaw_rad,
        "yaw_deg": yaw_deg,
        "orientation": pose.get("orientation"),
        "source": source,
    }


def rotate_translation(dx: float, dy: float, yaw_rad: float) -> Tuple[float, float]:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return c * dx - s * dy, s * dx + c * dy


def camera_pose_in_map(node_pose: Dict[str, Any], relative_yaw_deg: float) -> Dict[str, Any]:
    dx = CAMERA_TRANSLATION_IN_BASE["x"]
    dy = CAMERA_TRANSLATION_IN_BASE["y"]
    dz = CAMERA_TRANSLATION_IN_BASE["z"]

    rx, ry = rotate_translation(dx, dy, node_pose["yaw_rad"])

    camera_yaw_rad = normalize_rad(node_pose["yaw_rad"] + deg_to_rad(relative_yaw_deg))
    camera_yaw_deg = normalize_deg(rad_to_deg(camera_yaw_rad))

    return {
        "frame": "map",
        "x": node_pose["x"] + rx,
        "y": node_pose["y"] + ry,
        "z": node_pose["z"] + dz,
        "yaw_rad": camera_yaw_rad,
        "yaw_deg": camera_yaw_deg,
    }


def parse_stop_from_snapshot(snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stop_index = snapshot.get("stop_index")
    stop_name = snapshot.get("stop_name")

    if stop_index is not None:
        try:
            stop_index = int(stop_index)
            if stop_index in STOP_BY_INDEX:
                return STOP_BY_INDEX[stop_index]
        except Exception:
            pass

    if stop_name is not None:
        stop_name = safe_name(stop_name)
        if stop_name in STOP_BY_NAME:
            return STOP_BY_NAME[stop_name]

    # Fall back to the label, e.g. camera_stop_5_front.
    label = safe_name(snapshot.get("label", ""))
    match = re.search(r"camera_stop_(\d+)_([a-z0-9_]+)", label)
    if match:
        stop_index = int(match.group(1))
        if stop_index in STOP_BY_INDEX:
            return STOP_BY_INDEX[stop_index]

    # Fall back to color filename, e.g. c5_front.png.
    for filename in snapshot.get("filenames", []):
        base = Path(filename).name.lower()
        match = re.fullmatch(r"c(\d+)_([a-z0-9_]+)\.png", base)
        if match:
            stop_index = int(match.group(1))
            if stop_index in STOP_BY_INDEX:
                return STOP_BY_INDEX[stop_index]

    return None


def find_color_filename(snapshot: Dict[str, Any], stop: Dict[str, Any]) -> Optional[str]:
    """Return the color image filename only; depth d*.png files are ignored."""
    stop_index = int(stop["stop_index"])
    stop_name = safe_name(stop["stop_name"])
    expected = f"c{stop_index}_{stop_name}.png"

    filenames = [Path(f).name for f in snapshot.get("filenames", [])]

    # Prefer the exact expected name.
    for filename in filenames:
        if filename.lower() == expected.lower():
            return filename

    # Otherwise take any c<stop_index>_*.png.
    pattern = re.compile(rf"^c{stop_index}_[a-z0-9_]+\.png$", re.IGNORECASE)
    for filename in filenames:
        if pattern.fullmatch(filename):
            return filename

    # If metadata lists image stamps but not filenames, still generate expected path.
    image_stamps = snapshot.get("image_stamps", {})
    if "realsense_color" in image_stamps:
        return expected

    return None


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------

def build_image_entry(
    node_name: str,
    node_pose: Dict[str, Any],
    snapshot: Dict[str, Any],
    stop: Dict[str, Any],
    color_filename: str,
    fov_deg: float,
) -> Dict[str, Any]:
    stop_index = int(stop["stop_index"])
    stop_name = safe_name(stop["stop_name"])
    relative_yaw_deg = float(stop["relative_yaw_deg"])

    heading_rel_deg = normalize_deg(relative_yaw_deg)
    heading_abs_deg = normalize_deg(node_pose["yaw_deg"] + relative_yaw_deg)
    heading_abs_rad = normalize_rad(node_pose["yaw_rad"] + deg_to_rad(relative_yaw_deg))

    image_path = str(Path(node_name) / color_filename)
    camera_pose = camera_pose_in_map(node_pose, relative_yaw_deg)

    return {
        "image_id": f"{node_name}__{stop_name}",
        "camera_id": stop_name,
        "stop_index": stop_index,
        "stop_name": stop_name,
        "frame": color_filename,
        "image_path": image_path,
        "description": "",
        "relative_yaw_deg": relative_yaw_deg,
        "heading_rel_deg": heading_rel_deg,
        "heading_abs_deg": heading_abs_deg,
        "heading_abs_rad": heading_abs_rad,
        "fov_deg": fov_deg,
        "camera_pose": camera_pose,
        "camera_transform_from_base_link": {
            "x": CAMERA_TRANSLATION_IN_BASE["x"],
            "y": CAMERA_TRANSLATION_IN_BASE["y"],
            "z": CAMERA_TRANSLATION_IN_BASE["z"],
            "roll_deg": CAMERA_STATIC_RPY_IN_BASE_DEG["roll"],
            "pitch_deg": CAMERA_STATIC_RPY_IN_BASE_DEG["pitch"],
            "yaw_deg": relative_yaw_deg,
        },
        "capture": {
            "label": snapshot.get("label"),
            "captured_at": snapshot.get("captured_at"),
            "image_stamp": snapshot.get("image_stamps", {}).get("realsense_color"),
            "frame_token": snapshot.get("frame_tokens", {}).get("realsense_color"),
        },
    }


def build_image_entries(
    node_name: str,
    node_pose: Dict[str, Any],
    metadata: Dict[str, Any],
    fov_deg: float,
    mode: str,
) -> List[Dict[str, Any]]:
    # Collect by stop_name so the 4-camera version can be ordered front,left,back,right.
    by_stop_name: Dict[str, Dict[str, Any]] = {}

    for snapshot in metadata.get("snapshots", []):
        stop = parse_stop_from_snapshot(snapshot)
        if stop is None:
            print(f"[WARN] Could not identify stop for {node_name} snapshot {snapshot.get('label')}. Skipping.")
            continue

        color_filename = find_color_filename(snapshot, stop)
        if color_filename is None:
            print(f"[WARN] No color image filename found for {node_name} stop {stop['stop_index']} {stop['stop_name']}. Skipping.")
            continue

        entry = build_image_entry(
            node_name=node_name,
            node_pose=node_pose,
            snapshot=snapshot,
            stop=stop,
            color_filename=color_filename,
            fov_deg=fov_deg,
        )
        by_stop_name[entry["stop_name"]] = entry

    if mode == "all8":
        ordered_names = [safe_name(stop["stop_name"]) for stop in CAMERA_STOPS]
    elif mode == "cardinal4":
        ordered_names = CARDINAL_STOP_ORDER
    else:
        raise ValueError(f"Unknown mode: {mode}")

    images = [by_stop_name[name] for name in ordered_names if name in by_stop_name]
    return images


def build_node(
    node_dir: Path,
    metadata: Dict[str, Any],
    map_name: str,
    fov_deg: float,
    mode: str,
) -> Dict[str, Any]:
    node_name = metadata.get("node_name", node_dir.name).lower()
    node_pose = extract_node_pose(metadata, node_name)

    images = build_image_entries(
        node_name=node_name,
        node_pose=node_pose,
        metadata=metadata,
        fov_deg=fov_deg,
        mode=mode,
    )

    return {
        "id": node_name,
        "map": map_name,
        "pose": {
            "x": node_pose["x"],
            "y": node_pose["y"],
            "z": node_pose["z"],
            "yaw_rad": node_pose["yaw_rad"],
            "yaw_deg": node_pose["yaw_deg"],
            "orientation": node_pose.get("orientation"),
            "source": node_pose["source"],
        },
        "reachable": True,
        "node_type": "decision_point",
        "summary": "",
        "images": images,
        "neighbors": NEIGHBORS.get(node_name, []),
    }


def validate_topology(nodes: List[Dict[str, Any]]) -> None:
    existing = {node["id"] for node in nodes}

    for node in nodes:
        missing = [neighbor for neighbor in node["neighbors"] if neighbor not in existing]
        if missing:
            print(f"[WARN] {node['id']} has neighbors not found as folders: {missing}")

    # Check symmetry only among nodes that exist in the dataset.
    for node in nodes:
        node_id = node["id"]
        for neighbor in node["neighbors"]:
            if neighbor not in existing:
                continue
            neighbor_node = next(n for n in nodes if n["id"] == neighbor)
            if node_id not in neighbor_node["neighbors"]:
                print(f"[WARN] Asymmetric edge: {node_id} -> {neighbor}, but {neighbor} does not list {node_id}")


def build_graph(dataset_root: Path, map_name: str, fov_deg: float, mode: str) -> Dict[str, Any]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    node_dirs = [
        path for path in dataset_root.iterdir()
        if path.is_dir() and re.fullmatch(r"n\d+", path.name.lower())
    ]
    node_dirs.sort(key=node_sort_key)

    nodes: List[Dict[str, Any]] = []

    for node_dir in node_dirs:
        metadata_path = node_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"[WARN] Missing metadata.json in {node_dir}. Skipping.")
            continue

        try:
            metadata = load_json(metadata_path)
            node = build_node(
                node_dir=node_dir,
                metadata=metadata,
                map_name=map_name,
                fov_deg=fov_deg,
                mode=mode,
            )
            nodes.append(node)
        except Exception as exc:
            print(f"[WARN] Failed to process {metadata_path}: {exc}")

    validate_topology(nodes)

    if mode == "all8":
        name_suffix = "8_color_views"
        notes = "Contains all eight RealSense color views: back, back_left, left, front_left, front, front_right, right, back_right. Depth frames are omitted."
    else:
        name_suffix = "4_cardinal_color_views"
        notes = "Contains only the four cardinal RealSense color views: front, left, back, right. Depth frames are omitted."

    return {
        "meta": {
            "name": f"{dataset_root.name}_{name_suffix}_topological_graph",
            "frame": "map",
            "source_dataset": str(dataset_root),
            "image_mode": mode,
            "angle_convention": (
                "degrees; relative_yaw_deg is relative to the robot/node yaw; "
                "heading_abs_deg = normalize(node_yaw_deg + relative_yaw_deg). "
                "Positive yaw is left/CCW; front=0, left=+90, right=-90."
            ),
            "camera_mount_transform_from_base_link": {
                "translation": dict(CAMERA_TRANSLATION_IN_BASE),
                "static_rpy_deg": dict(CAMERA_STATIC_RPY_IN_BASE_DEG),
                "note": "For each virtual camera view, relative_yaw_deg is applied as an additional rotation around z.",
            },
            "camera_stops": CAMERA_STOPS,
            "notes": notes,
        },
        "maps": [map_name],
        "nodes": nodes,
    }


def write_graph(graph: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build graph JSON files from LeRobot RealSense 8-stop node metadata."
    )
    parser.add_argument(
        "--dataset-root",
        default="/rgbd_camera_intel_dev/src/nodes",
        help="Folder containing n01, n02, ... folders with metadata.json files.",
    )
    parser.add_argument(
        "--output-all8",
        default="/rgbd_camera_intel_dev/src/nodes_graph_8_color.json",
        help="Output path for graph containing all 8 color images per node.",
    )
    parser.add_argument(
        "--output-cardinal4",
        default="/rgbd_camera_intel_dev/src/nodes_graph_4_cardinal.json",
        help="Output path for graph containing only front/left/back/right color images per node.",
    )
    parser.add_argument(
        "--map-name",
        default="warehouse",
        help="Logical map name stored in each node. Frame remains 'map'.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=90.0,
        help="Approximate horizontal FOV associated with each image.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_all8 = Path(args.output_all8)
    output_cardinal4 = Path(args.output_cardinal4)

    graph_all8 = build_graph(
        dataset_root=dataset_root,
        map_name=args.map_name,
        fov_deg=args.fov_deg,
        mode="all8",
    )
    write_graph(graph_all8, output_all8)

    graph_cardinal4 = build_graph(
        dataset_root=dataset_root,
        map_name=args.map_name,
        fov_deg=args.fov_deg,
        mode="cardinal4",
    )
    write_graph(graph_cardinal4, output_cardinal4)

    count_all8 = sum(len(node["images"]) for node in graph_all8["nodes"])
    count_cardinal4 = sum(len(node["images"]) for node in graph_cardinal4["nodes"])

    print(f"[OK] Wrote 8-view graph to: {output_all8}")
    print(f"[OK] Nodes: {len(graph_all8['nodes'])}, color images: {count_all8}")
    print(f"[OK] Wrote 4-view graph to: {output_cardinal4}")
    print(f"[OK] Nodes: {len(graph_cardinal4['nodes'])}, color images: {count_cardinal4}")


if __name__ == "__main__":
    main()