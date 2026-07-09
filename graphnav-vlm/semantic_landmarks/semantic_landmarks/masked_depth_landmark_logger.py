#!/usr/bin/env python3

import ast
import os
import re
import time
from collections import defaultdict, deque

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image


def parse_prompts(param_value):
    if isinstance(param_value, list):
        return [str(x).strip() for x in param_value if str(x).strip()]

    if isinstance(param_value, str):
        s = param_value.strip()

        try:
            val = ast.literal_eval(s)
            if isinstance(val, str):
                return [val]
            if isinstance(val, (list, tuple)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass

        # Supports: "person,desk,chair"
        return [p.strip() for p in s.strip("[]").split(",") if p.strip()]

    return ["person"]


def sanitize_topic_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "object"


class MaskedDepthLandmarkLogger(Node):
    def __init__(self):
        super().__init__("masked_depth_landmark_logger")

        self.declare_parameter("prompts", "person")
        self.declare_parameter("masked_depth_prefix", "/clipseg/masked_depth")
        self.declare_parameter("camera_info_topic", "/camera/depth/camera_info_compressed")
        self.declare_parameter("target_point_prefix", "/target_point")

        self.declare_parameter("output_file", "/tmp/landmarks.yaml")
        self.declare_parameter("write_period_sec", 2.0)

        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 10.0)
        self.declare_parameter("min_valid_pixels", 50)

        # Number of recent 3D estimates to keep per object
        self.declare_parameter("samples_window", 15)

        param_prompts: Parameter = self.get_parameter("prompts")
        self.prompts = parse_prompts(param_prompts.value)

        self.masked_depth_prefix = self.get_parameter("masked_depth_prefix").value.rstrip("/")
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_point_prefix = self.get_parameter("target_point_prefix").value.rstrip("/")
        self.output_file = self.get_parameter("output_file").value
        self.write_period_sec = float(self.get_parameter("write_period_sec").value)

        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_valid_pixels = int(self.get_parameter("min_valid_pixels").value)
        self.samples_window = int(self.get_parameter("samples_window").value)

        self.bridge = CvBridge()

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame_id = None

        self.samples = defaultdict(lambda: deque(maxlen=self.samples_window))
        self.last_seen = {}

        self.point_pubs = {}
        self.depth_subs = []

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.info_callback,
            10,
        )

        for prompt in self.prompts:
            safe_name = sanitize_topic_name(prompt)

            masked_topic = f"{self.masked_depth_prefix}/{safe_name}"
            point_topic = f"{self.target_point_prefix}/{safe_name}"

            self.point_pubs[prompt] = self.create_publisher(PointStamped, point_topic, 10)

            sub = self.create_subscription(
                Image,
                masked_topic,
                lambda msg, p=prompt: self.depth_callback(msg, p),
                10,
            )
            self.depth_subs.append(sub)

            self.get_logger().info(f"[{prompt}] subscribing: {masked_topic}")
            self.get_logger().info(f"[{prompt}] publishing:  {point_topic}")

        self.timer = self.create_timer(self.write_period_sec, self.write_landmarks)

        self.get_logger().info(f"Landmark YAML output: {self.output_file}")
        self.get_logger().info(f"Prompts: {self.prompts}")

    def info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.camera_frame_id = msg.header.frame_id

    def depth_to_meters(self, depth_img, encoding):
        if encoding == "16UC1":
            return depth_img.astype(np.float32) / 1000.0
        if encoding in ("32FC1", "32FC1_s"):
            return depth_img.astype(np.float32)

        self.get_logger().warn(f"Unsupported depth encoding '{encoding}', assuming meters")
        return depth_img.astype(np.float32)

    def depth_callback(self, msg: Image, prompt: str):
        if self.fx is None:
            self.get_logger().warn("No camera_info received yet")
            return

        depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_m = self.depth_to_meters(depth_raw, msg.encoding)

        valid = np.isfinite(depth_m)
        valid &= depth_m > self.min_depth_m
        valid &= depth_m < self.max_depth_m

        ys, xs = np.where(valid)

        if xs.size < self.min_valid_pixels:
            return

        depths = depth_m[ys, xs]

        # Robust center estimate
        u = float(np.median(xs))
        v = float(np.median(ys))
        z = float(np.median(depths))

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        xyz = np.array([x, y, z], dtype=np.float32)

        self.samples[prompt].append(xyz)
        self.last_seen[prompt] = time.time()

        median_xyz = self.get_median_xyz(prompt)
        if median_xyz is None:
            return

        pt = PointStamped()
        pt.header.stamp = msg.header.stamp
        pt.header.frame_id = self.camera_frame_id or msg.header.frame_id

        pt.point.x = float(median_xyz[0])
        pt.point.y = float(median_xyz[1])
        pt.point.z = float(median_xyz[2])

        self.point_pubs[prompt].publish(pt)

    def get_median_xyz(self, prompt: str):
        if prompt not in self.samples or len(self.samples[prompt]) == 0:
            return None

        arr = np.stack(list(self.samples[prompt]), axis=0)
        return np.median(arr, axis=0)

    def write_landmarks(self):
        output = {
            "stamp": time.time(),
            "frame_id": self.camera_frame_id,
            "landmarks": {},
        }

        for prompt in self.prompts:
            median_xyz = self.get_median_xyz(prompt)
            if median_xyz is None:
                continue

            output["landmarks"][prompt] = {
                "x": float(median_xyz[0]),
                "y": float(median_xyz[1]),
                "z": float(median_xyz[2]),
                "samples": len(self.samples[prompt]),
                "last_seen": float(self.last_seen.get(prompt, 0.0)),
            }

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        tmp_file = self.output_file + ".tmp"
        with open(tmp_file, "w") as f:
            yaml.safe_dump(output, f, sort_keys=False)

        os.replace(tmp_file, self.output_file)

        self.get_logger().info(
            f"Wrote {len(output['landmarks'])} landmarks to {self.output_file}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MaskedDepthLandmarkLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()