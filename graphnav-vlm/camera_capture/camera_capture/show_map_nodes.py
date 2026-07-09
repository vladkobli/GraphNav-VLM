#!/usr/bin/env python3

import json
import math
import re
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class NodeMetadataMapMarkers(Node):
    def __init__(self):
        super().__init__("node_metadata_map_markers")

        self.declare_parameter("nodes_dir", "/rgbd_camera_intel_dev/src/nodes")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("marker_topic", "/metadata_node_markers")
        self.declare_parameter("marker_scale", 0.45)
        self.declare_parameter("arrow_length", 1.2)
        self.declare_parameter("publish_period_sec", 1.0)

        self.nodes_dir = Path(self.get_parameter("nodes_dir").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        self.arrow_length = float(self.get_parameter("arrow_length").value)

        self.map_frame = "map"
        self.have_map = False

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            marker_qos,
        )

        period = float(self.get_parameter("publish_period_sec").value)
        self.timer = self.create_timer(period, self.publish_markers)

        self.get_logger().info(f"Listening to map topic: {self.map_topic}")
        self.get_logger().info(f"Reading node metadata from: {self.nodes_dir}")
        self.get_logger().info(f"Publishing markers to: {self.marker_topic}")

    def map_callback(self, msg: OccupancyGrid):
        self.map_frame = msg.header.frame_id or "map"
        self.have_map = True

    def find_metadata_files(self):
        if not self.nodes_dir.exists():
            self.get_logger().warn(f"nodes_dir does not exist: {self.nodes_dir}")
            return []

        def node_sort_key(path: Path):
            match = re.match(r"n(\d+)$", path.parent.name)
            return int(match.group(1)) if match else 10**9

        files = sorted(
            self.nodes_dir.glob("n*/metadata.json"),
            key=node_sort_key,
        )
        return files

    def extract_pose(self, metadata):
        """
        Returns x, y, yaw, source_name.
        Prefer top-level planned_pose, then fall back to first snapshot actual_robot_pose.
        """
        planned = metadata.get("planned_pose")
        if planned is not None:
            if all(k in planned for k in ("x", "y", "yaw")):
                return (
                    float(planned["x"]),
                    float(planned["y"]),
                    float(planned["yaw"]),
                    "planned_pose",
                )

        for snapshot in metadata.get("snapshots", []):
            actual = snapshot.get("actual_robot_pose")
            if actual is not None and all(k in actual for k in ("x", "y", "yaw")):
                return (
                    float(actual["x"]),
                    float(actual["y"]),
                    float(actual["yaw"]),
                    "snapshot_actual_robot_pose",
                )

        return None

    def make_sphere_marker(self, marker_id, node_name, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "metadata_node_positions"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0

        marker.scale.x = self.marker_scale
        marker.scale.y = self.marker_scale
        marker.scale.z = self.marker_scale

        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 0.2
        marker.color.a = 1.0

        return marker

    def make_arrow_marker(self, marker_id, x, y, yaw):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "metadata_node_arrows"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        start = Point()
        start.x = x
        start.y = y
        start.z = 0.25

        end = Point()
        end.x = x + self.arrow_length * math.cos(yaw)
        end.y = y + self.arrow_length * math.sin(yaw)
        end.z = 0.25

        marker.points = [start, end]

        marker.scale.x = 0.08   # shaft diameter
        marker.scale.y = 0.18   # head diameter
        marker.scale.z = 0.25   # head length

        marker.color.r = 0.9
        marker.color.g = 0.2
        marker.color.b = 0.1
        marker.color.a = 1.0

        return marker

    def make_text_marker(self, marker_id, node_name, x, y):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "metadata_node_labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.8
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.45

        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.text = node_name
        return marker

    def publish_delete_all(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def publish_markers(self):
        metadata_files = self.find_metadata_files()

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        marker_id = 0
        valid_count = 0

        for metadata_file in metadata_files:
            try:
                with metadata_file.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)

                pose = self.extract_pose(metadata)
                if pose is None:
                    self.get_logger().warn(
                        f"No valid pose found in {metadata_file}"
                    )
                    continue

                x, y, yaw, pose_source = pose
                node_name = metadata.get("node_name", metadata_file.parent.name)

                sphere = self.make_sphere_marker(marker_id, node_name, x, y)
                marker_array.markers.append(sphere)
                marker_id += 1

                arrow = self.make_arrow_marker(marker_id, x, y, yaw)
                marker_array.markers.append(arrow)
                marker_id += 1

                text = self.make_text_marker(marker_id, node_name, x, y)
                marker_array.markers.append(text)
                marker_id += 1

                valid_count += 1

            except Exception as exc:
                self.get_logger().warn(f"Could not read {metadata_file}: {exc}")

        self.marker_pub.publish(marker_array)

        if valid_count > 0:
            self.get_logger().info(
                f"Published {valid_count} node poses as markers in frame '{self.map_frame}'"
            )


def main(args=None):
    rclpy.init(args=args)
    node = NodeMetadataMapMarkers()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()