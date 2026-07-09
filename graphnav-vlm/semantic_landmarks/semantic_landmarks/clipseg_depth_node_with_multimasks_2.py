#!/usr/bin/env python3

import ast
import re
import time
import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image
from transformers.models.clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header


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


class ClipSegDepthNode(Node):
    def __init__(self):
        super().__init__("clipseg_depth_node")

        self.declare_parameter("rgb_topic", "/camera/driver/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/driver/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/driver/aligned_depth_to_color/camera_info")

        self.declare_parameter("prompts", "person")
        self.declare_parameter("mask_threshold", 0.5)

        # Performance / output controls
        self.declare_parameter("max_inference_hz", 15.0)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("publish_union_masked_depth", False)
        self.declare_parameter("publish_per_prompt_masked_depth", False)
        self.declare_parameter("publish_masked_depth_rviz", False)

        # 3D point extraction
        self.declare_parameter("target_point_prefix", "/target_point")
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 10.0)
        self.declare_parameter("min_valid_pixels", 50)

        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value

        param_prompts: Parameter = self.get_parameter("prompts")
        self.prompts = parse_prompts(param_prompts.value)

        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.max_inference_hz = float(self.get_parameter("max_inference_hz").value)

        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.publish_union_masked_depth = bool(
            self.get_parameter("publish_union_masked_depth").value
        )
        self.publish_per_prompt_masked_depth = bool(
            self.get_parameter("publish_per_prompt_masked_depth").value
        )
        self.publish_masked_depth_rviz = bool(
            self.get_parameter("publish_masked_depth_rviz").value
        )

        self.target_point_prefix = self.get_parameter("target_point_prefix").value.rstrip("/")
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_valid_pixels = int(self.get_parameter("min_valid_pixels").value)

        self.last_inference_time = 0.0

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame_id = None

        self.bridge = CvBridge()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(f"Prompts: {self.prompts}")

        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)
        self.model.eval()

        # Publishers
        self.target_pubs = {}
        self.masked_depth_pubs = {}

        for prompt in self.prompts:
            safe_name = sanitize_topic_name(prompt)

            point_topic = f"{self.target_point_prefix}/{safe_name}"
            self.target_pubs[prompt] = self.create_publisher(PointStamped, point_topic, 10)
            self.get_logger().info(f"Publishing target point for '{prompt}' on: {point_topic}")

            if self.publish_per_prompt_masked_depth:
                mask_topic = f"clipseg/masked_depth/{safe_name}"
                self.masked_depth_pubs[prompt] = self.create_publisher(Image, mask_topic, 1)
                self.get_logger().info(
                    f"Publishing masked depth for '{prompt}' on: /{mask_topic}"
                )

        self.debug_pub = None
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, "clipseg/debug_image", 1)

        self.masked_depth_pub = None
        if self.publish_union_masked_depth:
            self.masked_depth_pub = self.create_publisher(Image, "clipseg/masked_depth", 1)

        self.masked_depth_rviz_pub = None
        if self.publish_masked_depth_rviz:
            self.masked_depth_rviz_pub = self.create_publisher(
                Image,
                "clipseg/masked_depth_rviz",
                1,
            )

        # Camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.info_callback,
            qos_profile_sensor_data,
        )

        # RGB + depth sync
        self.rgb_sub = Subscriber(
            self,
            Image,
            rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=0.5,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(f"Subscribing RGB:        {rgb_topic}")
        self.get_logger().info(f"Subscribing depth:      {depth_topic}")
        self.get_logger().info(f"Subscribing cameraInfo: {camera_info_topic}")
        self.get_logger().info(f"Max inference rate:     {self.max_inference_hz:.2f} Hz")

    def info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.camera_frame_id = msg.header.frame_id

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None or self.fy is None:
            self.get_logger().warn("No camera_info received yet", throttle_duration_sec=2.0)
            return

        if not self._valid_intrinsics():
            self.get_logger().warn(
                f"Invalid camera intrinsics: fx={self.fx}, fy={self.fy}, "
                f"cx={self.cx}, cy={self.cy}",
                throttle_duration_sec=2.0,
            )
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.max_inference_hz > 0.0:
            min_period = 1.0 / self.max_inference_hz
            if now - self.last_inference_time < min_period:
                return
            self.last_inference_time = now

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            cv_depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )

            depth_encoding = depth_msg.encoding
            depth_m = self._depth_to_meters(cv_depth_raw, depth_encoding)

            H_d, W_d = depth_m.shape
            H_rgb, W_rgb = cv_rgb.shape[:2]

            rgb_pil = PILImage.fromarray(cv_rgb)
            images = [rgb_pil] * len(self.prompts)

            inputs = self.processor(
                text=self.prompts,
                images=images,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.sigmoid(outputs.logits).detach().cpu().numpy()
            if probs.ndim == 4:
                probs = probs[:, 0, :, :]

            overlay = None
            if self.publish_debug:
                overlay = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)

            global_mask = None
            if self.publish_union_masked_depth or self.publish_masked_depth_rviz:
                global_mask = np.zeros((H_d, W_d), dtype=bool)

            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
            ]

            for i, prompt in enumerate(self.prompts):
                mask_prob = probs[i]
                H_m, W_m = mask_prob.shape

                if (H_m, W_m) != (H_d, W_d):
                    mask_prob_depth = cv2.resize(
                        mask_prob,
                        (W_d, H_d),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    mask_prob_depth = mask_prob

                mask_bin_depth = mask_prob_depth > self.mask_threshold

                if not np.any(mask_bin_depth):
                    self.get_logger().info(
                        f"[{prompt}] no mask pixels above threshold",
                        throttle_duration_sec=2.0,
                    )
                    continue

                if global_mask is not None:
                    global_mask |= mask_bin_depth

                target_xyz = self.compute_target_xyz(depth_m, mask_bin_depth)

                if target_xyz is None:
                    continue

                self.publish_target_point(prompt, target_xyz, depth_msg)

                self.get_logger().info(
                    f"[{prompt}] xyz=({target_xyz[0]:.2f}, "
                    f"{target_xyz[1]:.2f}, {target_xyz[2]:.2f}) m",
                    throttle_duration_sec=1.0,
                )

                if self.publish_per_prompt_masked_depth:
                    self.publish_prompt_masked_depth(
                        prompt,
                        cv_depth_raw,
                        depth_encoding,
                        mask_bin_depth,
                        depth_msg,
                    )

                if self.publish_debug and overlay is not None:
                    if (H_m, W_m) != (H_rgb, W_rgb):
                        mask_prob_rgb = cv2.resize(
                            mask_prob,
                            (W_rgb, H_rgb),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        mask_prob_rgb = mask_prob

                    mask_bin_rgb = mask_prob_rgb > self.mask_threshold
                    overlay = self.draw_overlay(
                        overlay,
                        mask_bin_rgb,
                        prompt,
                        target_xyz[2],
                        colors[i % len(colors)],
                    )

            if self.publish_debug and overlay is not None and self.debug_pub is not None:
                debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
                debug_msg.header = rgb_msg.header
                self.debug_pub.publish(debug_msg)

            if global_mask is not None and np.any(global_mask):
                masked_depth = self.build_masked_depth(
                    cv_depth_raw,
                    depth_encoding,
                    global_mask,
                )

                if masked_depth is not None and self.masked_depth_pub is not None:
                    masked_depth_msg = self.bridge.cv2_to_imgmsg(
                        masked_depth,
                        encoding=depth_encoding,
                    )
                    masked_depth_msg.header = depth_msg.header
                    self.masked_depth_pub.publish(masked_depth_msg)

                if masked_depth is not None and self.masked_depth_rviz_pub is not None:
                    depth_viz = self._depth_viz(masked_depth, depth_encoding)
                    depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding="mono8")
                    depth_viz_msg.header = depth_msg.header
                    self.masked_depth_rviz_pub.publish(depth_viz_msg)

        except Exception as e:
            self.get_logger().error(
                f"Exception in synced_callback: {repr(e)}",
                throttle_duration_sec=1.0,
            )
            return

    def _valid_intrinsics(self):
        values = [self.fx, self.fy, self.cx, self.cy]
        if not all(np.isfinite(v) for v in values):
            return False
        if self.fx == 0.0 or self.fy == 0.0:
            return False
        return True

    def compute_target_xyz(self, depth_m: np.ndarray, mask_bin: np.ndarray):
        valid = mask_bin.copy()
        valid &= np.isfinite(depth_m)
        valid &= depth_m > self.min_depth_m
        valid &= depth_m < self.max_depth_m

        ys, xs = np.where(valid)

        if xs.size < self.min_valid_pixels:
            return None

        depths = depth_m[ys, xs]
        depths = depths[np.isfinite(depths)]
        depths = depths[(depths > self.min_depth_m) & (depths < self.max_depth_m)]

        if depths.size < self.min_valid_pixels:
            return None

        u = float(np.median(xs))
        v = float(np.median(ys))
        z = float(np.median(depths))

        if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(z):
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        xyz = np.array([x, y, z], dtype=np.float32)

        if not np.all(np.isfinite(xyz)):
            return None

        return xyz

    def publish_target_point(self, prompt: str, xyz: np.ndarray, depth_msg: Image):
        pt = PointStamped()
        pt.header.stamp = depth_msg.header.stamp
        pt.header.frame_id = self.camera_frame_id or depth_msg.header.frame_id

        pt.point.x = float(xyz[0])
        pt.point.y = float(xyz[1])
        pt.point.z = float(xyz[2])

        self.target_pubs[prompt].publish(pt)

    def publish_prompt_masked_depth(
        self,
        prompt: str,
        cv_depth_raw: np.ndarray,
        depth_encoding: str,
        mask_bin: np.ndarray,
        depth_msg: Image,
    ):
        if prompt not in self.masked_depth_pubs:
            return

        prompt_masked_depth = self.build_masked_depth(
            cv_depth_raw,
            depth_encoding,
            mask_bin,
        )

        if prompt_masked_depth is None:
            return

        prompt_masked_depth_msg = self.bridge.cv2_to_imgmsg(
            prompt_masked_depth,
            encoding=depth_encoding,
        )
        prompt_masked_depth_msg.header = depth_msg.header
        self.masked_depth_pubs[prompt].publish(prompt_masked_depth_msg)

    def build_masked_depth(
        self,
        cv_depth_raw: np.ndarray,
        depth_encoding: str,
        mask_bin: np.ndarray,
    ):
        if depth_encoding == "16UC1":
            masked_depth = np.zeros_like(cv_depth_raw, dtype=np.uint16)
            masked_depth[mask_bin] = cv_depth_raw[mask_bin]
            return masked_depth

        if depth_encoding in ("32FC1", "32FC1_s"):
            masked_depth = np.zeros_like(cv_depth_raw, dtype=np.float32)
            masked_depth[mask_bin] = cv_depth_raw[mask_bin]
            return masked_depth

        self.get_logger().warn(
            f"Unsupported depth encoding for masked depth: {depth_encoding}",
            throttle_duration_sec=2.0,
        )
        return None

    def draw_overlay(
        self,
        overlay: np.ndarray,
        mask_bin_rgb: np.ndarray,
        prompt: str,
        distance_m: float,
        color,
    ):
        if not np.any(mask_bin_rgb):
            return overlay

        colored = np.zeros_like(overlay, dtype=np.uint8)
        colored[mask_bin_rgb] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)

        ys, xs = np.where(mask_bin_rgb)
        if xs.size == 0 or ys.size == 0:
            return overlay

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{prompt}: {distance_m:.2f}m"
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        return overlay

    def _depth_to_meters(self, depth_img, encoding: str):
        if encoding == "16UC1":
            return depth_img.astype(np.float32) / 1000.0
        if encoding in ("32FC1", "32FC1_s"):
            return depth_img.astype(np.float32)

        self.get_logger().warn(
            f"Unknown depth encoding '{encoding}', assuming meters.",
            throttle_duration_sec=2.0,
        )
        return depth_img.astype(np.float32)

    def _depth_viz(self, depth_img, encoding: str):
        if encoding == "16UC1":
            depth_m = depth_img.astype(np.float32) / 1000.0
        else:
            depth_m = depth_img.astype(np.float32)

        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        out = np.zeros(depth_m.shape, dtype=np.uint8)

        if np.any(valid):
            vals = depth_m[valid]
            mn, mx = vals.min(), vals.max()
            if mx > mn:
                scaled = (depth_m - mn) / (mx - mn)
                out[valid] = (scaled[valid] * 255).astype(np.uint8)
            else:
                out[valid] = 255

        return out


def main(args=None):
    rclpy.init(args=args)
    node = ClipSegDepthNode()
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