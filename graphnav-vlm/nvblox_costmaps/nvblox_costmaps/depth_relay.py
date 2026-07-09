#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import CameraInfo, Image


class DepthRelayNode(Node):
    def __init__(self):
        super().__init__("nvblox_depth_relay")

        self.declare_parameter(
            "input_depth_topic",
            "/camera0/driver/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "input_camera_info_topic",
            "/camera0/driver/aligned_depth_to_color/camera_info",
        )
        self.declare_parameter(
            "output_depth_topic",
            "/nvblox/relay/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "output_camera_info_topic",
            "/nvblox/relay/aligned_depth_to_color/camera_info",
        )

        input_depth_topic = self.get_parameter("input_depth_topic").value
        input_camera_info_topic = self.get_parameter("input_camera_info_topic").value
        output_depth_topic = self.get_parameter("output_depth_topic").value
        output_camera_info_topic = self.get_parameter("output_camera_info_topic").value

        qos = QoSProfile(depth=10)
        qos.history = QoSHistoryPolicy.KEEP_LAST
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.durability = DurabilityPolicy.VOLATILE

        self.depth_pub = self.create_publisher(Image, output_depth_topic, qos)
        self.info_pub = self.create_publisher(CameraInfo, output_camera_info_topic, qos)

        self.create_subscription(Image, input_depth_topic, self.depth_callback, qos)
        self.create_subscription(CameraInfo, input_camera_info_topic, self.info_callback, qos)

        self.get_logger().info(
            f"Relaying depth image {input_depth_topic} -> {output_depth_topic} "
            f"and camera info {input_camera_info_topic} -> {output_camera_info_topic}"
        )

    def depth_callback(self, msg: Image) -> None:
        self.depth_pub.publish(msg)

    def info_callback(self, msg: CameraInfo) -> None:
        self.info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DepthRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
