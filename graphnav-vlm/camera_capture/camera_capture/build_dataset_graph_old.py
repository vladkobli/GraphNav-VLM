#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# Camera mounting yaw relative to robot base_link / node yaw
# Your convention:
# camera2 -> front / 0 deg
# camera0 -> +90 deg
# camera1 -> +180 deg
# camera3 -> +270 deg
CAMERA_REL_YAW_DEG = {
    "camera2": 0.0,
    "camera0": 90.0,
    "camera1": 180.0,
    "camera3": 270.0,
}

# Frame-to-camera mapping
# camera2 -> frames 1,2,3
# camera0 -> frames 4,5,6
# camera1 -> frames 7,8,9
# camera3 -> frames 10,11,12
FRAME_TO_CAMERA = {
    1: "camera2",
    2: "camera2",
    3: "camera2",

    4: "camera0",
    5: "camera0",
    6: "camera0",

    7: "camera1",
    8: "camera1",
    9: "camera1",

    10: "camera3",
    11: "camera3",
    12: "camera3",
}


NEIGHBORS = {
    "n01": ["n02"],
    "n02": ["n01", "n03"],
    "n03": ["n02", "n04", "n05"],
    "n04": ["n03", "n05", "n06"],
    "n05": ["n03", "n04", "n07"],
    "n06": ["n04"],
    "n07": ["n05", "n08"],
    "n08": ["n07", "n09", "n15"],
    "n09": ["n08", "n10", "n11"],
    "n10": ["n09"],
    "n11": ["n09", "n12"],
    "n12": ["n11", "n13", "n16"],
    "n13": ["n12", "n14"],
    "n14": ["n13", "n18"],
    "n15": ["n08", "n16"],
    "n16": ["n12", "n15", "n18"],
    "n18": ["n14", "n16", "n19"],
    "n19": ["n18", "n20", "n21"],
    "n20": ["n19", "n21", "n22"],
    "n21": ["n19", "n20", "n22"],
    "n22": ["n20", "n21", "n23", "n24"],
    "n23": ["n22", "n24"],
    "n24": ["n22", "n23", "n25"],
    "n25": ["n22", "n24", "n26", "n28"],
    "n26": ["n25", "n27", "n28"],
    "n27": ["n26"],
    "n28": ["n25", "n26", "n29"],
    "n29": ["n28", "n30"],
    "n30": ["n29", "n31", "n34", "n35"],
    "n31": ["n30", "n32", "n33", "n34", "n35"],
    "n32": ["n31", "n33", "n34", "n35"],
    "n33": ["n31", "n32", "n34"],
    "n34": ["n31", "n32", "n33", "n35"],
    "n35": ["n30", "n31", "n32", "n34", "n36"],
    "n36": ["n35", "n37", "n38"],
    "n37": ["n36", "n38"],
    "n38": ["n36", "n37", "n39"],
    "n39": ["n38"],
}


def normalize_deg(angle: float) -> float:
    """Normalize angle to [0, 360)."""
    return angle % 360.0


def rad_to_deg(rad: float) -> float:
    return math.degrees(rad)


def extract_frame_number(filename: str) -> Optional[int]:
    """
    Extract frame number from names like:
      frame1.png
      frame10.png
      something_frame12.png
    """
    match = re.search(r"frame(\d+)", filename)
    if not match:
        return None
    return int(match.group(1))


def node_sort_key(path: Path) -> int:
    """
    Sort n01, n02, ..., n10 numerically.
    Unknown names go last.
    """
    match = re.search(r"n(\d+)", path.name.lower())
    if not match:
        return 10**9
    return int(match.group(1))


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_image_entries(
    node_name: str,
    node_yaw_deg: float,
    node_dir: Path,
    metadata: Dict[str, Any],
    fov_deg: float,
) -> List[Dict[str, Any]]:
    images = []

    snapshots = metadata.get("snapshots", [])

    for snapshot in snapshots:
        snapshot_label = snapshot.get("label", "")
        snapshot_yaw_offset_deg = float(snapshot.get("yaw_offset_degrees", 0.0))
        filenames = snapshot.get("filenames", [])

        for filename in filenames:
            frame_number = extract_frame_number(filename)

            if frame_number is None:
                print(f"[WARN] Could not extract frame number from {node_name}/{filename}. Skipping.")
                continue

            camera = FRAME_TO_CAMERA.get(frame_number)

            if camera is None:
                print(f"[WARN] No camera mapping for {node_name}/{filename}. Skipping.")
                continue

            camera_mount_yaw_deg = CAMERA_REL_YAW_DEG[camera]

            # Final camera yaw relative to the node yaw:
            # robot snapshot rotation + camera mounting angle
            heading_rel_deg = normalize_deg(
                snapshot_yaw_offset_deg + camera_mount_yaw_deg
            )

            # Absolute map heading:
            # node yaw in map + final relative camera yaw
            heading_abs_deg = normalize_deg(
                node_yaw_deg + heading_rel_deg
            )

            image_path = str(Path(node_name) / filename)

            image_entry = {
                "image_id": f"{node_name}__{Path(filename).stem}",
                "camera": camera,
                "frame": filename,
                "frame_number": frame_number,
                "snapshot": snapshot_label,
                "snapshot_yaw_offset_deg": snapshot_yaw_offset_deg,
                "camera_mount_rel_deg": camera_mount_yaw_deg,
                "heading_rel_deg": heading_rel_deg,
                "fov_deg": fov_deg,
                "heading_abs_deg": heading_abs_deg,
                "image_path": image_path,
            }

            images.append(image_entry)

    images.sort(key=lambda img: img["frame_number"])

    return images


def build_node(
    node_dir: Path,
    metadata: Dict[str, Any],
    dataset_root: Path,
    fov_deg: float,
) -> Dict[str, Any]:
    node_name = metadata.get("node_name", node_dir.name).lower()

    planned_pose = metadata.get("planned_pose")
    if planned_pose is None:
        raise ValueError(f"{node_name} has no planned_pose")

    x = float(planned_pose["x"])
    y = float(planned_pose["y"])
    z = float(planned_pose.get("z", 0.0))

    node_yaw_rad = float(planned_pose["yaw"])
    node_yaw_deg = normalize_deg(rad_to_deg(node_yaw_rad))

    images = build_image_entries(
        node_name=node_name,
        node_yaw_deg=node_yaw_deg,
        node_dir=node_dir,
        metadata=metadata,
        fov_deg=fov_deg,
    )

    node = {
        "id": node_name,
        "map": "warehouse",
        "pose": {
            "x": x,
            "y": y,
            "z": z,
            "yaw_rad": node_yaw_rad,
            "yaw_deg": node_yaw_deg,
        },
        "reachable": True,
        "node_type": "decision_point",
        "images": images,
        "neighbors": NEIGHBORS.get(node_name, []),
    }

    return node


def build_graph(dataset_root: Path, output_path: Path, fov_deg: float) -> Dict[str, Any]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    node_dirs = [
        p for p in dataset_root.iterdir()
        if p.is_dir() and re.fullmatch(r"n\d+", p.name.lower())
    ]

    node_dirs.sort(key=node_sort_key)

    nodes = []

    for node_dir in node_dirs:
        metadata_path = node_dir / "metadata.json"

        if not metadata_path.exists():
            print(f"[WARN] Missing metadata.json in {node_dir}. Skipping.")
            continue

        try:
            metadata = load_metadata(metadata_path)
            node = build_node(
                node_dir=node_dir,
                metadata=metadata,
                dataset_root=dataset_root,
                fov_deg=fov_deg,
            )
            nodes.append(node)

        except Exception as e:
            print(f"[WARN] Failed to process {metadata_path}: {e}")

    existing_node_ids = {node["id"] for node in nodes}

    for node in nodes:
        missing_neighbors = [
            neighbor for neighbor in node["neighbors"]
            if neighbor not in existing_node_ids
        ]

        if missing_neighbors:
            print(
                f"[WARN] {node['id']} has neighbors not found in dataset folders: "
                f"{missing_neighbors}"
            )

    graph = {
        "meta": {
            "name": "good_dataset_2_topological_graph",
            "frame": "map",
            "angle_convention": (
                "degrees; heading_rel_deg is relative to node yaw; "
                "heading_abs_deg = node yaw + snapshot yaw offset + camera mounting yaw"
            ),
            "source_dataset": str(dataset_root),
            "notes": (
                "Generated from metadata.json files. Captions are intentionally omitted. "
                "Camera mapping: camera2=0deg, camera0=90deg, camera1=180deg, camera3=270deg. "
                "Snapshot yaw offsets are added on top of camera mounting yaw."
            ),
        },
        "maps": ["warehouse"],
        "nodes": nodes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="Build final dataset graph JSON from good_dataset_2 metadata.json files."
    )

    parser.add_argument(
        "--dataset-root",
        default="/rgbd_camera_intel_dev/src/good_dataset_2",
        help="Path to dataset folder containing n00, n01, ... subfolders.",
    )

    parser.add_argument(
        "--output",
        default="/rgbd_camera_intel_dev/src/good_dataset_2_graph.json",
        help="Output graph JSON path.",
    )

    parser.add_argument(
        "--fov-deg",
        type=float,
        default=90.0,
        help="Camera FOV in degrees. Default is 90 because the physical cameras are spaced by 90 deg.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)

    graph = build_graph(
        dataset_root=dataset_root,
        output_path=output_path,
        fov_deg=args.fov_deg,
    )

    print(f"[OK] Wrote graph JSON to: {output_path}")
    print(f"[OK] Nodes: {len(graph['nodes'])}")

    image_count = sum(len(node["images"]) for node in graph["nodes"])
    print(f"[OK] Images: {image_count}")


if __name__ == "__main__":
    main()