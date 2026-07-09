#!/usr/bin/env python3

import ast
import os
import re
import time
from collections import defaultdict, deque

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.parameter import Parameter


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

        return [p.strip() for p in s.strip("[]").split(",") if p.strip()]

    return ["person"]


def sanitize_topic_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "object"


class LandmarkLogger(Node):
    def __init__(self):
        super().__init__("masked_depth_landmark_logger")

        self.declare_parameter("prompts", "person")
        self.declare_parameter("target_point_prefix", "/target_point")
        self.declare_parameter("output_file", "/tmp/landmarks.yaml")

        self.declare_parameter("write_period_sec", 2.0)
        self.declare_parameter("samples_window", 15)
        self.declare_parameter("min_samples_for_yaml", 10)
        self.declare_parameter("max_landmark_age_sec", 60.0)

        param_prompts: Parameter = self.get_parameter("prompts")
        self.prompts = parse_prompts(param_prompts.value)

        self.target_point_prefix = self.get_parameter("target_point_prefix").value.rstrip("/")
        self.output_file = self.get_parameter("output_file").value

        self.write_period_sec = float(self.get_parameter("write_period_sec").value)
        self.samples_window = int(self.get_parameter("samples_window").value)
        self.min_samples_for_yaml = int(self.get_parameter("min_samples_for_yaml").value)
        self.max_landmark_age_sec = float(self.get_parameter("max_landmark_age_sec").value)

        self.samples = defaultdict(lambda: deque(maxlen=self.samples_window))
        self.last_seen = {}
        self.frame_ids = {}

        self.subs = []

        for prompt in self.prompts:
            safe_name = sanitize_topic_name(prompt)
            topic = f"{self.target_point_prefix}/{safe_name}"

            sub = self.create_subscription(
                PointStamped,
                topic,
                lambda msg, p=prompt: self.point_callback(msg, p),
                10,
            )
            self.subs.append(sub)

            self.get_logger().info(f"[{prompt}] subscribing: {topic}")

        self.timer = self.create_timer(self.write_period_sec, self.write_landmarks)

        self.get_logger().info(f"Landmark YAML output: {self.output_file}")
        self.get_logger().info(f"Prompts: {self.prompts}")
        self.get_logger().info(f"samples_window: {self.samples_window}")
        self.get_logger().info(f"min_samples_for_yaml: {self.min_samples_for_yaml}")
        self.get_logger().info(f"max_landmark_age_sec: {self.max_landmark_age_sec}")

    def point_callback(self, msg: PointStamped, prompt: str):
        xyz = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(xyz)):
            self.get_logger().warn(
                f"[{prompt}] ignoring non-finite point: {xyz}",
                throttle_duration_sec=1.0,
            )
            return

        self.samples[prompt].append(xyz)
        self.last_seen[prompt] = time.time()
        self.frame_ids[prompt] = msg.header.frame_id

    def get_median_xyz(self, prompt: str):
        if prompt not in self.samples:
            return None

        if len(self.samples[prompt]) < self.min_samples_for_yaml:
            return None

        arr = np.stack(list(self.samples[prompt]), axis=0)

        if not np.all(np.isfinite(arr)):
            arr = arr[np.all(np.isfinite(arr), axis=1)]

        if arr.shape[0] < self.min_samples_for_yaml:
            return None

        return np.median(arr, axis=0)

    def write_landmarks(self):
        now = time.time()

        output = {
            "stamp": now,
            "landmarks": {},
        }

        for prompt in self.prompts:
            last_seen = self.last_seen.get(prompt, 0.0)

            if last_seen <= 0.0:
                continue

            age = now - last_seen

            if age > self.max_landmark_age_sec:
                self.samples[prompt].clear()
                continue

            median_xyz = self.get_median_xyz(prompt)
            if median_xyz is None:
                continue

            output["landmarks"][prompt] = {
                "frame_id": self.frame_ids.get(prompt, ""),
                "x": float(median_xyz[0]),
                "y": float(median_xyz[1]),
                "z": float(median_xyz[2]),
                "samples": len(self.samples[prompt]),
                "last_seen": float(last_seen),
                "age_sec": float(age),
            }

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        tmp_file = self.output_file + ".tmp"

        with open(tmp_file, "w") as f:
            yaml.safe_dump(output, f, sort_keys=False)

        os.replace(tmp_file, self.output_file)

        self.get_logger().info(
            f"Wrote {len(output['landmarks'])} landmarks to {self.output_file}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = LandmarkLogger()

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