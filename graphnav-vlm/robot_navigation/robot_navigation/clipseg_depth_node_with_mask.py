#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import cv2
import torch
import numpy as np
from PIL import Image as PILImage
from transformers.models.clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor

from message_filters import Subscriber, ApproximateTimeSynchronizer

import ast
from rclpy.parameter import Parameter


class ClipSegDepthNode(Node):
    def __init__(self):
        super().__init__("clipseg_depth_node")

        self.declare_parameter("rgb_topic", "/camera0/driver/color/image_raw")
        self.declare_parameter("depth_topic", "/camera0/driver/aligned_depth_to_color/image_raw")
        self.declare_parameter("prompts", ["chair", "table", "person"])
        self.declare_parameter("mask_threshold", 0.5)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value

        param_prompts: Parameter = self.get_parameter("prompts")

        if param_prompts.type_ == Parameter.Type.STRING_ARRAY and param_prompts.value:
            prompts = list(param_prompts.value)
        elif param_prompts.type_ == Parameter.Type.STRING and param_prompts.value:
            s = param_prompts.value.strip()
            try:
                val = ast.literal_eval(s)
                if isinstance(val, str):
                    prompts = [val]
                elif isinstance(val, (list, tuple)):
                    prompts = [str(x) for x in val]
                else:
                    prompts = [str(val)]
            except Exception:
                prompts = [p.strip() for p in s.split(",") if p.strip()]
        else:
            prompts = ["chair", "table", "person"]

        self.prompts = prompts
        self.mask_threshold = self.get_parameter("mask_threshold").get_parameter_value().double_value or 0.5

        self.bridge = CvBridge()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)
        self.model.eval()

        self.debug_pub = self.create_publisher(Image, "clipseg/debug_image", 1)
        self.masked_depth_pub = self.create_publisher(Image, "clipseg/masked_depth", 1)
        self.get_logger().info("masked_depth publisher created")
        self.masked_depth_rviz_pub = self.create_publisher(Image, "clipseg/masked_depth_rviz", 1)

        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=5,
            slop=0.1,
        )
        self.sync.registerCallback(self.synced_callback)

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
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

        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        if probs.ndim == 4:
            probs = probs[:, 0, :, :]

        depth_encoding = depth_msg.encoding
        cv_depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = self._depth_to_meters(cv_depth_raw, depth_encoding)

        H_d, W_d = depth_m.shape
        overlay = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)

        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        # Union of all prompt masks
        global_mask = np.zeros((H_d, W_d), dtype=bool)

        for i, prompt in enumerate(self.prompts):
            mask_prob = probs[i]
            H_m, W_m = mask_prob.shape

            if (H_m, W_m) != (H_d, W_d):
                mask_prob_resized = cv2.resize(
                    mask_prob,
                    (W_d, H_d),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask_prob_resized = mask_prob

            mask_bin = mask_prob_resized > self.mask_threshold

            if not np.any(mask_bin):
                continue

            global_mask |= mask_bin

            depths = depth_m[mask_bin]
            depths = depths[np.isfinite(depths)]
            depths = depths[depths > 0.05]

            if depths.size > 0:
                dist_median = float(np.median(depths))
                self.get_logger().info(f"[{prompt}] median distance: {dist_median:.2f} m", throttle_duration_sec=1.0)
                

            color = colors[i % len(colors)]
            colored = np.zeros_like(overlay, dtype=np.uint8)
            colored[mask_bin] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.4, 0)

            ys, xs = np.where(mask_bin)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            label = prompt
            cv2.putText(
                overlay,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        # Publish RGB debug overlay
        debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
        debug_msg.header = rgb_msg.header
        self.debug_pub.publish(debug_msg)

        # Build masked depth image in ORIGINAL encoding
        if depth_encoding == "16UC1":
            masked_depth = np.zeros_like(cv_depth_raw, dtype=np.uint16)
            masked_depth[global_mask] = cv_depth_raw[global_mask]
        elif depth_encoding in ("32FC1", "32FC1_s"):
            masked_depth = np.zeros_like(cv_depth_raw, dtype=np.float32)
            masked_depth[global_mask] = cv_depth_raw[global_mask]
        else:
            self.get_logger().warn(f"Unsupported depth encoding for republish: {depth_encoding}")
            return

        masked_depth_msg = self.bridge.cv2_to_imgmsg(masked_depth, encoding=depth_encoding)
        masked_depth_msg.header = depth_msg.header
        self.masked_depth_pub.publish(masked_depth_msg)

        # Optional visualization-only depth image
        depth_viz = self._depth_viz(masked_depth, depth_encoding)
        depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding="mono8")
        depth_viz_msg.header = depth_msg.header
        self.masked_depth_rviz_pub.publish(depth_viz_msg)

    def _depth_to_meters(self, depth_img, encoding: str):
        if encoding == "16UC1":
            return depth_img.astype(np.float32) / 1000.0
        elif encoding in ("32FC1", "32FC1_s"):
            return depth_img.astype(np.float32)
        else:
            self.get_logger().warn(f"Unknown depth encoding '{encoding}', assuming meters.")
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
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()