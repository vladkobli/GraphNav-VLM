#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


class MaskedDepthTargetPointNode(Node):
    def __init__(self):
        super().__init__("masked_depth_target_point_node")

        self.declare_parameter("masked_depth_topic", "/clipseg/masked_depth")
        self.declare_parameter("camera_info_topic", "/camera0/driver/aligned_depth_to_color/camera_info")
        self.declare_parameter("target_point_topic", "/target_point")

        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 10.0)
        self.declare_parameter("min_valid_pixels", 50)
        self.declare_parameter("ema_alpha", 0.25)

        self.masked_depth_topic = self.get_parameter("masked_depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_point_topic = self.get_parameter("target_point_topic").value

        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_valid_pixels = int(self.get_parameter("min_valid_pixels").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)

        self.bridge = CvBridge()
        self.camera_info = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.filtered_xyz = None

        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.info_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.masked_depth_topic,
            self.depth_callback,
            10,
        )

        self.target_pub = self.create_publisher(
            PointStamped,
            self.target_point_topic,
            10,
        )

        self.get_logger().info(f"Subscribing masked depth: {self.masked_depth_topic}")
        self.get_logger().info(f"Subscribing camera info:  {self.camera_info_topic}")
        self.get_logger().info(f"Publishing target point:  {self.target_point_topic}")

    def info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def depth_to_meters(self, depth_img, encoding):
        if encoding == "16UC1":
            return depth_img.astype(np.float32) / 1000.0
        if encoding in ("32FC1", "32FC1_s"):
            return depth_img.astype(np.float32)

        self.get_logger().warn(f"Unsupported depth encoding '{encoding}', assuming meters")
        return depth_img.astype(np.float32)

    def depth_callback(self, msg: Image):
        if self.camera_info is None:
            self.get_logger().warn("No camera_info received yet")
            return

        depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_m = self.depth_to_meters(depth_raw, msg.encoding)

        valid = np.isfinite(depth_m)
        valid &= depth_m > self.min_depth_m
        valid &= depth_m < self.max_depth_m

        # Since /clipseg/masked_depth has zeros outside the target,
        # all valid non-zero pixels are treated as target pixels.
        ys, xs = np.where(valid)

        if xs.size < self.min_valid_pixels:
            self.get_logger().warn(f"Not enough valid masked depth pixels: {xs.size}", throttle_duration_sec=1.0)
            return

        depths = depth_m[ys, xs]

        # Robust target center: median image coordinate and median depth
        u = float(np.median(xs))
        v = float(np.median(ys))
        z = float(np.median(depths))

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        xyz = np.array([x, y, z], dtype=np.float32)

        if self.filtered_xyz is None:
            self.filtered_xyz = xyz
        else:
            self.filtered_xyz = (
                (1.0 - self.ema_alpha) * self.filtered_xyz
                + self.ema_alpha * xyz
            )

        pt = PointStamped()
        pt.header.stamp = msg.header.stamp

        # Prefer the camera_info frame. It should correspond to the aligned depth frame.
        pt.header.frame_id = self.camera_info.header.frame_id or msg.header.frame_id

        pt.point.x = float(self.filtered_xyz[0])
        pt.point.y = float(self.filtered_xyz[1])
        pt.point.z = float(self.filtered_xyz[2])

        self.target_pub.publish(pt)

        self.get_logger().info(
            f"Published /target_point frame={pt.header.frame_id} "
            f"xyz=({pt.point.x:.2f}, {pt.point.y:.2f}, {pt.point.z:.2f}) "
            f"valid_pixels={xs.size}",
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = MaskedDepthTargetPointNode()
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