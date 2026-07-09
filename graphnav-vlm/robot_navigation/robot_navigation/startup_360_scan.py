#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node


class Startup360Scan(Node):
    """Do one slow in-place scan to help SLAM initialize the local map."""

    def __init__(self):
        super().__init__("startup_360_scan")

        self.declare_parameter("enabled", True)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_nav")
        self.declare_parameter("direct_cmd_vel_topic", "/panther/cmd_vel")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("angular_speed", 0.35)
        self.declare_parameter("rotations", 1.0)
        self.declare_parameter("start_delay_sec", 1.0)
        self.declare_parameter("settle_sec", 2.0)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.direct_cmd_vel_topic = str(self.get_parameter("direct_cmd_vel_topic").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.angular_speed = abs(float(self.get_parameter("angular_speed").value))
        self.rotations = float(self.get_parameter("rotations").value)
        self.start_delay_sec = float(self.get_parameter("start_delay_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self.direct_cmd_pub = None
        if self.direct_cmd_vel_topic and self.direct_cmd_vel_topic != self.cmd_vel_topic:
            self.direct_cmd_pub = self.create_publisher(TwistStamped, self.direct_cmd_vel_topic, 10)
        self.start_time = self.get_clock().now()
        self.scan_duration_sec = 0.0
        if self.angular_speed > 0.0 and self.rotations > 0.0:
            self.scan_duration_sec = (2.0 * math.pi * self.rotations) / self.angular_speed

        timer_period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_cb)

        if self.enabled and self.scan_duration_sec > 0.0:
            self.get_logger().info(
                f"Startup scan armed: {self.rotations:.1f} rotation(s) at "
                f"{self.angular_speed:.2f} rad/s on {self.cmd_vel_topic}"
            )
            if self.direct_cmd_pub is not None:
                self.get_logger().info(f"Also publishing startup scan directly on {self.direct_cmd_vel_topic}")
        else:
            self.get_logger().info("Startup scan disabled.")

    def elapsed_sec(self) -> float:
        return (self.get_clock().now() - self.start_time).nanoseconds / 1e9

    def publish_cmd(self, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.robot_frame
        msg.twist.angular.z = angular_z
        self.cmd_pub.publish(msg)
        if self.direct_cmd_pub is not None:
            self.direct_cmd_pub.publish(msg)

    def timer_cb(self):
        if not self.enabled or self.scan_duration_sec <= 0.0:
            self.timer.cancel()
            return

        elapsed = self.elapsed_sec()
        scan_start = self.start_delay_sec
        scan_end = scan_start + self.scan_duration_sec
        settle_end = scan_end + self.settle_sec

        if elapsed < scan_start:
            self.publish_cmd(0.0)
            return

        if elapsed < scan_end:
            self.publish_cmd(self.angular_speed)
            return

        if elapsed < settle_end:
            self.publish_cmd(0.0)
            return

        self.publish_cmd(0.0)
        self.get_logger().info("Startup scan complete.")
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = Startup360Scan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
