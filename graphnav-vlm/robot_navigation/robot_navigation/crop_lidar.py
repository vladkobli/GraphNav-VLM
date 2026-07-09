#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import tf2_ros


def quaternion_to_rotation_matrix(x, y, z, w):
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


class RoiSubtractLidarNode(Node):
    def __init__(self):
        super().__init__("roi_subtract_lidar_node")

        self.declare_parameter("rgbd_topic", "/custom/depth_points")
        self.declare_parameter("lidar_topic", "/velodyne_points")
        self.declare_parameter("roi_topic", "/roi_velodyne_points")
        self.declare_parameter("remaining_topic", "/velodyne_points_without_roi")

        self.declare_parameter("target_frame", "panther/base_link")

        # Padding around the ROI.
        self.declare_parameter("padding_x", 0.2)
        self.declare_parameter("padding_y", 0.3)
        self.declare_parameter("padding_z", 0.60)

        # Height limits in target_frame/base_link.
        self.declare_parameter("z_min", 0.05)
        self.declare_parameter("z_max", 2.2)

        # Ignore stale ROI boxes.
        self.declare_parameter("roi_timeout_sec", 0.3)

        # Use latest TF instead of timestamped TF. More robust for live camera/lidar streams.
        self.declare_parameter("use_latest_tf", True)
        self.declare_parameter("tf_timeout_sec", 0.3)

        self.rgbd_topic = self.get_parameter("rgbd_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.roi_topic = self.get_parameter("roi_topic").value
        self.remaining_topic = self.get_parameter("remaining_topic").value

        self.target_frame = self.get_parameter("target_frame").value

        self.padding_x = float(self.get_parameter("padding_x").value)
        self.padding_y = float(self.get_parameter("padding_y").value)
        self.padding_z = float(self.get_parameter("padding_z").value)

        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)

        self.roi_timeout_sec = float(self.get_parameter("roi_timeout_sec").value)
        self.use_latest_tf = bool(self.get_parameter("use_latest_tf").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        self.latest_box = None
        self.latest_box_time = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.rgbd_sub = self.create_subscription(
            PointCloud2,
            self.rgbd_topic,
            self.rgbd_callback,
            10,
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            10,
        )

        self.roi_pub = self.create_publisher(PointCloud2, self.roi_topic, 10)
        self.remaining_pub = self.create_publisher(PointCloud2, self.remaining_topic, 10)

        self.get_logger().info("ROI subtract lidar node started.")
        self.get_logger().info(f"RGBD input:      {self.rgbd_topic}")
        self.get_logger().info(f"LiDAR input:     {self.lidar_topic}")
        self.get_logger().info(f"ROI output:      {self.roi_topic}")
        self.get_logger().info(f"Remaining output:{self.remaining_topic}")
        self.get_logger().info(f"Target frame:    {self.target_frame}")

    def get_lookup_time(self, msg):
        if self.use_latest_tf:
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(msg.header.stamp)

    def rgbd_callback(self, msg: PointCloud2):
        rgbd_pts = self.pointcloud2_to_xyz(msg)

        if rgbd_pts.shape[0] == 0:
            self.get_logger().warn("RGBD ROI cloud is empty.")
            self.latest_box = None
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                self.get_lookup_time(msg),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as e:
            self.get_logger().warn(
                f"RGBD TF lookup failed: {msg.header.frame_id} -> {self.target_frame}: {e}"
            )
            self.latest_box = None
            return

        rgbd_pts_target = self.transform_points(rgbd_pts, tf_msg)

        if rgbd_pts_target.shape[0] == 0:
            self.latest_box = None
            return

        raw_min = np.min(rgbd_pts_target, axis=0)
        raw_max = np.max(rgbd_pts_target, axis=0)

        box_min = np.array([
            raw_min[0] - self.padding_x,
            raw_min[1] - self.padding_y,
            max(self.z_min, raw_min[2] - self.padding_z),
        ], dtype=np.float64)

        box_max = np.array([
            raw_max[0] + self.padding_x,
            raw_max[1] + self.padding_y,
            min(self.z_max, raw_max[2] + self.padding_z),
        ], dtype=np.float64)

        # Optional safety: enforce global height limits.
        box_min[2] = max(box_min[2], self.z_min)
        box_max[2] = min(box_max[2], self.z_max)

        self.latest_box = (box_min, box_max)
        self.latest_box_time = time.time()

        self.get_logger().info(
            f"Updated ROI box in {self.target_frame}: "
            f"raw_min={raw_min.round(3)}, raw_max={raw_max.round(3)}, "
            f"box_min={box_min.round(3)}, box_max={box_max.round(3)}, "
            f"rgbd_points={rgbd_pts_target.shape[0]}",
            throttle_duration_sec=1.0,
        )

    def lidar_callback(self, msg: PointCloud2):
        if self.latest_box is None:
            # Publish original cloud if no ROI exists yet.
            self.remaining_pub.publish(msg)
            return

        if (time.time() - self.latest_box_time) > self.roi_timeout_sec:
            self.get_logger().warn(
                "ROI box is stale; publishing original LiDAR cloud.",
                throttle_duration_sec=2.0,
            )            
            self.remaining_pub.publish(msg)
            return

        lidar_pts = self.pointcloud2_to_xyz(msg)

        if lidar_pts.shape[0] == 0:
            self.get_logger().warn("LiDAR cloud is empty.")
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                self.get_lookup_time(msg),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as e:
            self.get_logger().warn(
                f"LiDAR TF lookup failed: {msg.header.frame_id} -> {self.target_frame}: {e}"
            )
            return

        lidar_pts_target = self.transform_points(lidar_pts, tf_msg)

        box_min, box_max = self.latest_box

        inside = (
            (lidar_pts_target[:, 0] >= box_min[0]) & (lidar_pts_target[:, 0] <= box_max[0]) &
            (lidar_pts_target[:, 1] >= box_min[1]) & (lidar_pts_target[:, 1] <= box_max[1]) &
            (lidar_pts_target[:, 2] >= box_min[2]) & (lidar_pts_target[:, 2] <= box_max[2])
        )

        roi_pts = lidar_pts[inside]
        remaining_pts = lidar_pts[~inside]

        total = lidar_pts.shape[0]
        removed = roi_pts.shape[0]
        pct = 100.0 * removed / max(total, 1)

        self.get_logger().info(
            f"LiDAR split: total={total}, removed={removed} ({pct:.2f}%), "
            f"remaining={remaining_pts.shape[0]}",
            throttle_duration_sec=1.0
        )

        roi_msg = point_cloud2.create_cloud_xyz32(msg.header, roi_pts.astype(np.float32).tolist())
        remaining_msg = point_cloud2.create_cloud_xyz32(msg.header, remaining_pts.astype(np.float32).tolist())

        self.roi_pub.publish(roi_msg)
        self.remaining_pub.publish(remaining_msg)

    def pointcloud2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        pts = np.array(
            [
                [p[0], p[1], p[2]]
                for p in point_cloud2.read_points(
                    msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ],
            dtype=np.float32,
        )

        if pts.ndim != 2 or pts.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float32)

        return pts

    def transform_points(self, points_xyz: np.ndarray, tf_msg) -> np.ndarray:
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation

        R = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        T = np.array([t.x, t.y, t.z], dtype=np.float64)

        return (R @ points_xyz.T).T + T


def main(args=None):
    rclpy.init(args=args)
    node = RoiSubtractLidarNode()

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