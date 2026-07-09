#!/usr/bin/env python3

import json
import math
from urllib import request

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster


def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return {
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy,
    }


def get_nested(payload, dotted_path):
    value = payload
    for key in str(dotted_path).split("."):
        if not key:
            return None
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(value, list):
            try:
                value = value[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return value


class LeRobotCameraTfBroadcaster(Node):
    def __init__(self):
        super().__init__("lerobot_camera_tf_broadcaster")

        self.declare_parameter("source", "http")
        self.declare_parameter("parent_frame", "panther/body_link")
        self.declare_parameter("child_frame", "camera_link")
        self.declare_parameter("xyz", [-0.125, 0.02, 0.818])
        self.declare_parameter("roll", 0.0)
        self.declare_parameter("pitch", 0.0)
        self.declare_parameter("yaw_offset", 0.0)

        self.declare_parameter("joint_state_topic", "/lerobot/joint_states")
        self.declare_parameter("joint_name", "m5")

        self.declare_parameter("lerobot_server_url", "http://172.17.0.1:8765")
        self.declare_parameter("http_state_path", "/health")
        self.declare_parameter("http_yaw_path", "")
        self.declare_parameter("http_position_path", "positions.m5")
        self.declare_parameter("http_timeout_sec", 0.5)

        self.declare_parameter("position_zero", 2233.0)
        self.declare_parameter("position_to_rad", -0.001525045)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.source = str(self.get_parameter("source").value).strip().lower()
        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        self.xyz = [float(v) for v in self.get_parameter("xyz").value]
        self.roll = float(self.get_parameter("roll").value)
        self.pitch = float(self.get_parameter("pitch").value)
        self.yaw_offset = float(self.get_parameter("yaw_offset").value)

        self.joint_name = str(self.get_parameter("joint_name").value)
        self.lerobot_server_url = str(
            self.get_parameter("lerobot_server_url").value
        ).rstrip("/")
        self.http_state_path = str(self.get_parameter("http_state_path").value)
        self.http_yaw_path = str(self.get_parameter("http_yaw_path").value)
        self.http_position_path = str(self.get_parameter("http_position_path").value)
        self.http_timeout_sec = float(self.get_parameter("http_timeout_sec").value)
        self.position_zero = float(self.get_parameter("position_zero").value)
        self.position_to_rad = float(self.get_parameter("position_to_rad").value)

        self.last_yaw = self.yaw_offset
        self.broadcaster = TransformBroadcaster(self)

        if self.source == "joint_state":
            topic = str(self.get_parameter("joint_state_topic").value)
            self.create_subscription(JointState, topic, self.joint_state_callback, 10)
            self.get_logger().info(
                f"Broadcasting {self.parent_frame} -> {self.child_frame} from "
                f"JointState {topic}:{self.joint_name}"
            )
        elif self.source == "http":
            self.get_logger().info(
                f"Broadcasting {self.parent_frame} -> {self.child_frame} from "
                f"{self.lerobot_server_url}{self.http_state_path}"
            )
        else:
            raise ValueError("source must be either 'http' or 'joint_state'")

        rate = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate, self.timer_callback)

    def joint_state_callback(self, msg):
        if self.joint_name not in msg.name:
            return
        index = msg.name.index(self.joint_name)
        if index >= len(msg.position):
            return
        self.last_yaw = self.yaw_from_position(float(msg.position[index]))

    def timer_callback(self):
        if self.source == "http":
            yaw = self.read_http_yaw()
            if yaw is not None:
                self.last_yaw = yaw

        self.broadcast(self.last_yaw)

    def read_http_yaw(self):
        url = f"{self.lerobot_server_url}{self.http_state_path}"
        try:
            with request.urlopen(url, timeout=self.http_timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.get_logger().warn(
                f"Could not read LeRobot state from {url}: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        if self.http_yaw_path:
            yaw = get_nested(payload, self.http_yaw_path)
            if yaw is not None:
                return float(yaw) + self.yaw_offset

        position = get_nested(payload, self.http_position_path)
        if position is None:
            self.get_logger().warn(
                f"LeRobot state has no '{self.http_position_path}' field",
                throttle_duration_sec=2.0,
            )
            return None

        return self.yaw_from_position(float(position))

    def yaw_from_position(self, position):
        return self.yaw_offset + (position - self.position_zero) * self.position_to_rad

    def broadcast(self, yaw):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = self.xyz[0]
        transform.transform.translation.y = self.xyz[1]
        transform.transform.translation.z = self.xyz[2]

        quat = quaternion_from_euler(self.roll, self.pitch, yaw)
        transform.transform.rotation.x = quat["x"]
        transform.transform.rotation.y = quat["y"]
        transform.transform.rotation.z = quat["z"]
        transform.transform.rotation.w = quat["w"]

        self.broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = LeRobotCameraTfBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
