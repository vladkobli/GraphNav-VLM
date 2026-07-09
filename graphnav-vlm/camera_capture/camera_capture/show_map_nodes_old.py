#!/usr/bin/env python3

import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class DatasetWaypointMarkerPublisher(Node):
    def __init__(self):
        super().__init__("dataset_waypoint_marker_publisher")

        self.declare_parameter(
            "dataset_root",
            "/rgbd_camera_intel_dev/good_dataset_2",
        )
        self.declare_parameter("marker_topic", "/waypoint_logging/markers")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("use_actual_pose", False)

        self.dataset_root = Path(
            self.get_parameter("dataset_root").get_parameter_value().string_value
        )
        self.marker_topic = (
            self.get_parameter("marker_topic").get_parameter_value().string_value
        )
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.use_actual_pose = (
            self.get_parameter("use_actual_pose").get_parameter_value().bool_value
        )

        publish_rate_hz = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.publisher = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            qos,
        )

        self.waypoints = self.load_waypoints()

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} dataset nodes from: {self.dataset_root}"
        )
        self.get_logger().info(f"Publishing markers on: {self.marker_topic}")
        self.get_logger().info(f"RViz fixed frame should be: {self.frame_id}")

        period = 1.0 / max(publish_rate_hz, 0.1)
        self.timer = self.create_timer(period, self.publish_markers)

    def load_waypoints(self):
        waypoints = []

        if not self.dataset_root.exists():
            self.get_logger().error(f"Dataset root does not exist: {self.dataset_root}")
            return waypoints

        metadata_files = sorted(
            self.dataset_root.glob("n*/metadata.json"),
            key=lambda p: p.parent.name,
        )

        for metadata_path in metadata_files:
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)

                node_name = metadata.get("node_name", metadata_path.parent.name)
                node_id = metadata.get("node_id", len(waypoints) + 1)

                if self.use_actual_pose:
                    snapshots = metadata.get("snapshots", [])
                    if not snapshots:
                        self.get_logger().warn(
                            f"No snapshots found in {metadata_path}; skipping."
                        )
                        continue

                    pose = snapshots[0].get("actual_robot_pose", None)
                    if pose is None:
                        self.get_logger().warn(
                            f"No actual_robot_pose in first snapshot of {metadata_path}; skipping."
                        )
                        continue
                else:
                    pose = metadata.get("planned_pose", None)
                    if pose is None:
                        self.get_logger().warn(
                            f"No planned_pose found in {metadata_path}; skipping."
                        )
                        continue

                x = float(pose["x"])
                y = float(pose["y"])
                z = float(pose.get("z", 0.0))
                yaw = float(pose["yaw"])

                waypoints.append(
                    {
                        "node_id": node_id,
                        "node_name": node_name,
                        "x": x,
                        "y": y,
                        "z": z,
                        "yaw": yaw,
                    }
                )

            except Exception as e:
                self.get_logger().warn(f"Failed to read {metadata_path}: {e}")

        return waypoints

    @staticmethod
    def yaw_to_quaternion(yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    def make_arrow_marker(self, waypoint, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "waypoint_arrows"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        marker.pose.position.x = waypoint["x"]
        marker.pose.position.y = waypoint["y"]
        marker.pose.position.z = waypoint["z"] + 0.15

        qz, qw = self.yaw_to_quaternion(waypoint["yaw"])
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

        marker.scale.x = 0.8   # arrow length
        marker.scale.y = 0.12  # arrow width
        marker.scale.z = 0.12  # arrow height

        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0

        return marker

    def make_label_marker(self, waypoint, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "waypoint_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = waypoint["x"]
        marker.pose.position.y = waypoint["y"]
        marker.pose.position.z = waypoint["z"] + 0.8

        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.45

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        yaw_deg = math.degrees(waypoint["yaw"])
        marker.text = (
            f'{waypoint["node_name"]}\n'
            f'x={waypoint["x"]:.2f}, y={waypoint["y"]:.2f}\n'
            f'yaw={yaw_deg:.1f}°'
        )

        return marker

    def make_point_marker(self, waypoint, marker_id):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "waypoint_points"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = waypoint["x"]
        marker.pose.position.y = waypoint["y"]
        marker.pose.position.z = waypoint["z"] + 0.08
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.35
        marker.scale.y = 0.35
        marker.scale.z = 0.35

        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0

        return marker

    def publish_markers(self):
        marker_array = MarkerArray()

        # Clear previous markers first
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0

        for waypoint in self.waypoints:
            marker_array.markers.append(self.make_point_marker(waypoint, marker_id))
            marker_id += 1

            marker_array.markers.append(self.make_arrow_marker(waypoint, marker_id))
            marker_id += 1

            marker_array.markers.append(self.make_label_marker(waypoint, marker_id))
            marker_id += 1

        self.publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DatasetWaypointMarkerPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()