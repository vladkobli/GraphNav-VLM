#!/usr/bin/env python3

import math

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
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
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


class PointCloudHeightFilter(Node):
    """Filter 3D lidar by height and project it into a 2D scan for SLAM."""

    def __init__(self):
        super().__init__("pointcloud_height_filter")

        self.declare_parameter("input_topic", "/velodyne_points")
        self.declare_parameter("filtered_cloud_topic", "/velodyne_points_without_roi")
        self.declare_parameter("scan_topic", "/scan_frontier_3d")
        self.declare_parameter("target_frame", "panther/base_link")
        self.declare_parameter("use_latest_tf", True)
        self.declare_parameter("tf_timeout_sec", 0.2)

        self.declare_parameter("z_min", 0.3)
        self.declare_parameter("z_max", 2.0)
        self.declare_parameter("range_min", 0.25)
        self.declare_parameter("range_max", 20.0)
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)
        self.declare_parameter("angle_increment", 0.005)
        self.declare_parameter("scan_time", 0.1)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.filtered_cloud_topic = str(self.get_parameter("filtered_cloud_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.use_latest_tf = bool(self.get_parameter("use_latest_tf").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        self.z_min = float(self.get_parameter("z_min").value)
        self.z_max = float(self.get_parameter("z_max").value)
        self.range_min = float(self.get_parameter("range_min").value)
        self.range_max = float(self.get_parameter("range_max").value)
        self.angle_min = float(self.get_parameter("angle_min").value)
        self.angle_max = float(self.get_parameter("angle_max").value)
        self.angle_increment = float(self.get_parameter("angle_increment").value)
        self.scan_time = float(self.get_parameter("scan_time").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cloud_sub = self.create_subscription(PointCloud2, self.input_topic, self.cloud_cb, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, self.filtered_cloud_topic, 10)
        self.scan_pub = self.create_publisher(LaserScan, self.scan_topic, 10)

        self.get_logger().info(
            f"Height-filtering {self.input_topic} into {self.filtered_cloud_topic} and "
            f"{self.scan_topic}; z=[{self.z_min:.2f}, {self.z_max:.2f}] in {self.target_frame}"
        )

    def get_lookup_time(self, msg):
        if self.use_latest_tf:
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(msg.header.stamp)

    def cloud_cb(self, msg: PointCloud2):
        pts = self.pointcloud2_to_xyz(msg)
        if pts.shape[0] == 0:
            return

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                self.get_lookup_time(msg),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"LiDAR TF lookup failed: {msg.header.frame_id} -> {self.target_frame}: {exc}",
                throttle_duration_sec=2.0,
            )
            return

        pts_target = self.transform_points(pts, tf_msg)
        ranges = np.hypot(pts_target[:, 0], pts_target[:, 1])
        mask = (
            (pts_target[:, 2] >= self.z_min) &
            (pts_target[:, 2] <= self.z_max) &
            (ranges >= self.range_min) &
            (ranges <= self.range_max)
        )

        filtered_target = pts_target[mask]
        self.publish_filtered_cloud(msg, filtered_target)
        self.publish_scan(msg, filtered_target)

    def publish_filtered_cloud(self, source_msg: PointCloud2, points_target: np.ndarray):
        header = source_msg.header
        header.frame_id = self.target_frame
        cloud = point_cloud2.create_cloud_xyz32(
            header,
            points_target.astype(np.float32).tolist(),
        )
        self.cloud_pub.publish(cloud)

    def publish_scan(self, source_msg: PointCloud2, points_target: np.ndarray):
        scan = LaserScan()
        scan.header = source_msg.header
        scan.header.frame_id = self.target_frame
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = self.scan_time
        scan.range_min = self.range_min
        scan.range_max = self.range_max

        bin_count = int(math.ceil((self.angle_max - self.angle_min) / self.angle_increment))
        scan.ranges = [math.inf] * bin_count

        if points_target.shape[0] > 0:
            angles = np.arctan2(points_target[:, 1], points_target[:, 0])
            ranges = np.hypot(points_target[:, 0], points_target[:, 1])
            bins = ((angles - self.angle_min) / self.angle_increment).astype(np.int32)
            valid = (bins >= 0) & (bins < bin_count)

            for bin_index, point_range in zip(bins[valid], ranges[valid]):
                if point_range < scan.ranges[bin_index]:
                    scan.ranges[bin_index] = float(point_range)

        self.scan_pub.publish(scan)

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

        rotation = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)

        return (rotation @ points_xyz.T).T + translation


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudHeightFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
