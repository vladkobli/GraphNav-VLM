#!/usr/bin/env python3

import copy

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


class ResizeCameraInfoNode(Node):
    def __init__(self):
        super().__init__("resize_camera_info_node")

        self.declare_parameter("input_topic", "/camera0/driver/color/camera_info")
        self.declare_parameter("output_topic", "/camera/color/camera_info_compressed")
        self.declare_parameter("new_width", 352)
        # self.declare_parameter("new_height", 264)  # for 4:3 aspect ratio
        self.declare_parameter("new_height", 198)  # for 16:9 aspect ratio  

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.new_width = int(self.get_parameter("new_width").value)
        self.new_height = int(self.get_parameter("new_height").value)

        self.pub = self.create_publisher(CameraInfo, self.output_topic, 10)

        self.sub = self.create_subscription(
            CameraInfo,
            self.input_topic,
            self.callback,
            10
        )

        self.get_logger().info(
            f"Resizing CameraInfo {self.input_topic} -> {self.output_topic} "
            f"to {self.new_width}x{self.new_height}"
        )

    def callback(self, msg: CameraInfo):
        out = copy.deepcopy(msg)

        old_width = msg.width
        old_height = msg.height

        if old_width == 0 or old_height == 0:
            self.get_logger().warn("Received CameraInfo with zero width/height")
            return

        sx = self.new_width / float(old_width)
        sy = self.new_height / float(old_height)

        out.width = self.new_width
        out.height = self.new_height

        # K:
        # [fx  0 cx
        #  0 fy cy
        #  0  0  1]
        out.k[0] *= sx  # fx
        out.k[2] *= sx  # cx
        out.k[4] *= sy  # fy
        out.k[5] *= sy  # cy

        # P:
        # [fx' 0 cx' Tx
        #  0 fy' cy' Ty
        #  0  0  1   0]
        out.p[0] *= sx  # fx
        out.p[2] *= sx  # cx
        out.p[3] *= sx  # Tx, usually 0 for monocular color/depth
        out.p[5] *= sy  # fy
        out.p[6] *= sy  # cy
        out.p[7] *= sy  # Ty, usually 0

        # D and R stay unchanged.
        # Distortion coefficients do not change under simple image scaling.

        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ResizeCameraInfoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()