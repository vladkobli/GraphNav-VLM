#!/usr/bin/env python3
import math
import time
import ast

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers.models.clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor

from message_filters import Subscriber, ApproximateTimeSynchronizer


class ClipSegFollower(Node):
    def __init__(self):
        super().__init__("clipseg_depth_follower")

        # -------- Topics / prompts --------
        self.declare_parameter("rgb_topic", "/camera0/driver/color/image_raw")
        self.declare_parameter("depth_topic", "/camera0/driver/aligned_depth_to_color/image_raw")
        self.declare_parameter("cmd_vel_topic", "/panther/cmd_vel")
        self.declare_parameter("prompts", ["person"])
        self.declare_parameter("follow_prompt", "person")     # choose which prompt drives motion
        self.declare_parameter("mask_threshold", 0.6)

        # -------- Camera intrinsics (you can keep these) --------
        self.declare_parameter("camera_fx", 570.3422241210938)
        self.declare_parameter("camera_fy", 570.3422241210938)
        self.declare_parameter("camera_cx", 319.5)
        self.declare_parameter("camera_cy", 239.5)

        # -------- Controller --------
        self.declare_parameter("stop_distance", 0.7)          # meters
        self.declare_parameter("max_linear_speed", 0.25)      # m/s
        self.declare_parameter("max_angular_speed", 0.4)      # rad/s
        self.declare_parameter("k_lin", 0.35)                  # P gain distance
        self.declare_parameter("k_ang", 0.75)                  # P gain angle
        self.declare_parameter("angle_allow_forward_deg", 25.0)
        self.declare_parameter("angle_deadband_deg", 5.0)
        self.declare_parameter("distance_deadband", 0.05)     # meters

        # -------- Robustness --------
        self.declare_parameter("ema_alpha", 0.2)             # 0..1 (higher=less smoothing)
        self.declare_parameter("lost_timeout", 1.2)           # seconds w/out detection => stop
        self.declare_parameter("control_rate", 15.0)          # Hz
        self.declare_parameter("min_mask_pixels", 500)        # ignore tiny masks

        # Smoother motion
        self.declare_parameter("angle_forward_enter_deg", 8.0)
        self.declare_parameter("angle_forward_exit_deg", 14.0)
        self.declare_parameter("cmd_ema_alpha", 0.25)
        self.declare_parameter("max_linear_accel", 0.12)
        self.declare_parameter("max_angular_accel", 0.5)
        self.declare_parameter("stop_release_distance", 0.95)
        
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        
        self.angle_forward_enter = math.radians(float(self.get_parameter("angle_forward_enter_deg").value))
        self.angle_forward_exit = math.radians(float(self.get_parameter("angle_forward_exit_deg").value))
        self.cmd_alpha = float(self.get_parameter("cmd_ema_alpha").value)
        self.max_linear_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_angular_accel = float(self.get_parameter("max_angular_accel").value)
        self.stop_release_distance = float(self.get_parameter("stop_release_distance").value)

        self.last_lin_cmd = 0.0
        self.last_ang_cmd = 0.0
        self.goal_stopped = False

        self.forward_enabled = False # for hysteresis
        # Parse prompts robustly
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
            prompts = ["person"]

        self.prompts = prompts
        self.follow_prompt = str(self.get_parameter("follow_prompt").value)

        if self.follow_prompt not in self.prompts:
            self.get_logger().warn(
                f"follow_prompt='{self.follow_prompt}' not in prompts={self.prompts}. "
                f"Will follow prompts[0]='{self.prompts[0]}'."
            )
            self.follow_prompt = self.prompts[0]

        # Camera intrinsics
        self.fx = float(self.get_parameter("camera_fx").value)
        self.fy = float(self.get_parameter("camera_fy").value)
        self.cx = float(self.get_parameter("camera_cx").value)
        self.cy = float(self.get_parameter("camera_cy").value)

        # Controller params
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.max_linear = float(self.get_parameter("max_linear_speed").value)
        self.max_angular = float(self.get_parameter("max_angular_speed").value)
        self.k_lin = float(self.get_parameter("k_lin").value)
        self.k_ang = float(self.get_parameter("k_ang").value)

        self.angle_allow_forward = math.radians(float(self.get_parameter("angle_allow_forward_deg").value))
        self.angle_deadband = math.radians(float(self.get_parameter("angle_deadband_deg").value))
        self.distance_deadband = float(self.get_parameter("distance_deadband").value)

        # Robustness params
        self.alpha = float(self.get_parameter("ema_alpha").value)
        self.lost_timeout = float(self.get_parameter("lost_timeout").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.min_mask_pixels = int(self.get_parameter("min_mask_pixels").value)

        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"Depth topic: {depth_topic}")
        self.get_logger().info(f"Prompts: {self.prompts}")
        self.get_logger().info(f"Following: {self.follow_prompt}")
        self.get_logger().info(f"cmd_vel: {cmd_vel_topic}")

        self.bridge = CvBridge()

        # Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.model.eval()

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        self.debug_pub = self.create_publisher(Image, "clipseg/debug_image", 1)

        # Sync RGB + depth
        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=5, slop=0.12)
        self.sync.registerCallback(self.synced_callback)

        # State
        self.last_seen_target = None
        self.filt_dist = None
        self.filt_ang = None
        self.last_overlay = None

        # Timer control loop
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

    # ---------- Utils ----------
    def _depth_to_meters(self, depth_img, encoding: str):
        if encoding == "16UC1":
            return depth_img.astype(np.float32) / 1000.0
        elif encoding in ("32FC1", "32FC1_s"):
            return depth_img.astype(np.float32)
        else:
            self.get_logger().warn(f"Unknown depth encoding '{encoding}', assuming meters.")
            return depth_img.astype(np.float32)

    def _ema(self, prev, new):
        if prev is None:
            return new
        return (1.0 - self.alpha) * prev + self.alpha * new
    
    def _ema_cmd(self, prev, new, alpha):
        return (1.0 - alpha) * prev + alpha * new

    # ---------- Per-frame processing ----------
    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        rgb_pil = PILImage.fromarray(cv_rgb)

        # CLIPSeg inputs for each prompt
        images = [rgb_pil] * len(self.prompts)
        inputs = self.processor(text=self.prompts, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()
        if probs.ndim == 4:
            probs = probs[:, 0, :, :]

        # Depth
        cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_m = self._depth_to_meters(cv_depth, depth_msg.encoding)
        H_d, W_d = depth_m.shape

        overlay = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)

        # Find index of follow_prompt
        follow_i = self.prompts.index(self.follow_prompt)

        # Build mask for follow prompt
        mask_prob = probs[follow_i]
        if mask_prob.shape != (H_d, W_d):
            mask_prob = cv2.resize(mask_prob, (W_d, H_d), interpolation=cv2.INTER_NEAREST)

        mask_bin = mask_prob > self.mask_threshold

        # Ignore tiny blobs
        if int(mask_bin.sum()) < self.min_mask_pixels:
            self.last_overlay = overlay
            self.publish_debug(rgb_msg.header, overlay, note=f"{self.follow_prompt}: no/too-small mask")
            return

        depths = depth_m[mask_bin]
        depths = depths[np.isfinite(depths)]
        depths = depths[depths > 0.20]   # ignore 0 and too-close nonsense

        if depths.size < 50:
            self.last_overlay = overlay
            self.publish_debug(rgb_msg.header, overlay, note=f"{self.follow_prompt}: no valid depth")
            return

        # Median depth is your robust distance
        dist = float(np.median(depths))

        ys, xs = np.where(mask_bin)
        u = float(xs.mean())
        v = float(ys.mean())

        # Angle computed from pixel u (more stable than X/Z from noisy depth centroid)
        # For small angles: x_cam ≈ (u - cx)/fx, angle ≈ atan(x_cam)
        x_norm = (u - self.cx) / self.fx
        ang = float(math.atan(x_norm))

        # Filter
        self.filt_dist = self._ema(self.filt_dist, dist)
        self.filt_ang = self._ema(self.filt_ang, ang)
        self.last_seen_target = time.time()

        # Draw overlay
        color = (0, 255, 0)
        colored = np.zeros_like(overlay, dtype=np.uint8)
        colored[mask_bin] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = f"{self.follow_prompt} d={self.filt_dist:.2f}m a={math.degrees(self.filt_ang):.1f}deg"
        cv2.putText(overlay, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Mark centroid
        cv2.circle(overlay, (int(u), int(v)), 5, (0, 0, 255), -1)

        self.last_overlay = overlay
        self.publish_debug(rgb_msg.header, overlay)

        self.get_logger().info(
            f"[{self.follow_prompt}] dist(med)={dist:.2f} -> filt={self.filt_dist:.2f}, "
            f"ang={ang:.3f} -> filt={self.filt_ang:.3f}"
        )

    def publish_debug(self, header, overlay_bgr, note: str = ""):
        if note:
            cv2.putText(overlay_bgr, note, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        msg.header = header
        self.debug_pub.publish(msg)

    # ---------- Control loop ----------
    def control_loop(self):
        cmd = Twist()

        # Lost target -> stop
        if self.last_seen_target is None or (time.time() - self.last_seen_target) > self.lost_timeout:
            self.last_lin_cmd = 0.0
            self.last_ang_cmd = 0.0
            self.forward_enabled = False
            self.goal_stopped = False
            self.cmd_pub.publish(cmd)
            return

        dist = self.filt_dist
        ang = self.filt_ang

        if dist is None or ang is None:
            self.cmd_pub.publish(cmd)
            return

        # Stop hysteresis
        if self.goal_stopped:
            if dist > self.stop_release_distance:
                self.goal_stopped = False
        else:
            if dist <= self.stop_distance:
                self.goal_stopped = True

        if self.goal_stopped:
            self.last_lin_cmd = 0.0
            self.last_ang_cmd = 0.0
            self.forward_enabled = False
            self.cmd_pub.publish(Twist())
            return

        # Angular control
        if abs(ang) > self.angle_deadband:
            cmd.angular.z = max(min(-self.k_ang * ang, self.max_angular), -self.max_angular)
        else:
            cmd.angular.z = 0.0

        # Forward hysteresis state update
        if self.forward_enabled:
            if abs(ang) > self.angle_forward_exit:
                self.forward_enabled = False
        else:
            if abs(ang) < self.angle_forward_enter:
                self.forward_enabled = True

        # Linear control
        if self.forward_enabled:
            distance_error = max(0.0, dist - self.stop_distance)

            if distance_error > 1.0:
                lin = self.max_linear
            else:
                lin = self.max_linear * distance_error

            lin = min(lin, self.k_lin * distance_error)

            # soften forward speed if target is not perfectly centered
            angle_scale = max(0.0, 1.0 - abs(ang) / self.angle_forward_exit)
            lin *= angle_scale

            cmd.linear.x = max(min(lin, self.max_linear), 0.0)
        else:
            cmd.linear.x = 0.0

        # Command smoothing
        cmd.linear.x = self._ema_cmd(self.last_lin_cmd, cmd.linear.x, self.cmd_alpha)
        cmd.angular.z = self._ema_cmd(self.last_ang_cmd, cmd.angular.z, self.cmd_alpha)

        # Acceleration limiting
        max_lin_step = self.max_linear_accel / self.control_rate
        max_ang_step = self.max_angular_accel / self.control_rate

        cmd.linear.x = float(np.clip(
            cmd.linear.x,
            self.last_lin_cmd - max_lin_step,
            self.last_lin_cmd + max_lin_step
        ))

        cmd.angular.z = float(np.clip(
            cmd.angular.z,
            self.last_ang_cmd - max_ang_step,
            self.last_ang_cmd + max_ang_step
        ))

        self.last_lin_cmd = cmd.linear.x
        self.last_ang_cmd = cmd.angular.z

        self.cmd_pub.publish(cmd)

    def destroy_node(self):
        # stop robot on exit
        self.cmd_pub.publish(Twist())
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ClipSegFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Avoid "rcl_shutdown already called" if something else already shut it down
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()