#!/usr/bin/env python3

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ResizeImageNode(Node):
    def __init__(self):
        super().__init__("resize_image_node")

        self.declare_parameter("input_topic", "/camera0/color/image_raw")
        self.declare_parameter("output_topic", "/camera/color/image_compressed")
        self.declare_parameter("width", 352)
        # self.declare_parameter("height", 264)  # for 4:3 aspect ratio
        self.declare_parameter("height", 198)  # for 16:9 aspect ratio

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)

        self.bridge = CvBridge()

        self.pub = self.create_publisher(Image, self.output_topic, 10)
        self.sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(
            f"Resizing {self.input_topic} -> {self.output_topic} "
            f"to {self.width}x{self.height}"
        )

    def image_callback(self, msg: Image):
        try:
            is_depth = msg.encoding in ["16UC1", "32FC1", "mono16"]

            if is_depth:
                # Keep depth format unchanged
                cv_img = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough"
                )

                # Use nearest-neighbor for depth so values are not averaged
                resized = cv2.resize(
                    cv_img,
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST
                )

                out_msg = self.bridge.cv2_to_imgmsg(
                    resized,
                    encoding=msg.encoding
                )

            else:
                # RGB/color image
                cv_img = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8"
                )

                resized = cv2.resize(
                    cv_img,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA
                )

                out_msg = self.bridge.cv2_to_imgmsg(
                    resized,
                    encoding="bgr8"
                )

            out_msg.header = msg.header
            self.pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(
                f"Resize failed for topic {self.input_topic}, "
                f"encoding={msg.encoding}: {e}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ResizeImageNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()