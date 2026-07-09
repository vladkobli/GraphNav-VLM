#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


def yaw_to_quaternion(yaw: float):
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    return qx, qy, qz, qw


class TargetToNav2Follower(Node):
    def __init__(self):
        super().__init__("target_to_nav2_follower")

        self.declare_parameter("target_topic", "/target_point")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("stop_distance", 0.7)
        self.declare_parameter("update_threshold", 0.35)
        self.declare_parameter("min_target_distance", 0.35)
        self.declare_parameter("tf_timeout_sec", 0.5)

        self.target_topic = self.get_parameter("target_topic").value
        self.global_frame = self.get_parameter("global_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.update_threshold = float(self.get_parameter("update_threshold").value)
        self.min_target_distance = float(self.get_parameter("min_target_distance").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.target_sub = self.create_subscription(
            PointStamped, self.target_topic, self.target_cb, 10
        )

        self.current_goal_handle = None
        self.pending_send = False
        self.last_goal_x: Optional[float] = None
        self.last_goal_y: Optional[float] = None

        self.get_logger().info(f"Listening for targets on {self.target_topic}")

    def target_cb(self, msg: PointStamped):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn("navigate_to_pose action server not ready yet")
            return

        # Transform detected target to map
        try:
            target_tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
            target_map = do_transform_point(msg, target_tf)

            base_tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return

        robot_x = base_tf.transform.translation.x
        robot_y = base_tf.transform.translation.y

        obj_x = target_map.point.x
        obj_y = target_map.point.y

        dx = obj_x - robot_x
        dy = obj_y - robot_y
        dist = math.hypot(dx, dy)

        if dist < self.min_target_distance:
            self.get_logger().info("Target too close to robot, skipping new goal")
            return

        yaw = math.atan2(dy, dx)

        # Stop before the target
        approach = max(dist - self.stop_distance, 0.0)
        goal_x = robot_x + approach * math.cos(yaw)
        goal_y = robot_y + approach * math.sin(yaw)

        # Ignore tiny goal changes
        if self.last_goal_x is not None and self.last_goal_y is not None:
            shift = math.hypot(goal_x - self.last_goal_x, goal_y - self.last_goal_y)
            if shift < self.update_threshold:
                return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.last_goal_x = goal_x
        self.last_goal_y = goal_y

        self.get_logger().info(
            f"target(map)=({obj_x:.2f}, {obj_y:.2f}) "
            f"robot(map)=({robot_x:.2f}, {robot_y:.2f}) "
            f"goal(map)=({goal_x:.2f}, {goal_y:.2f}) yaw={yaw:.2f}"
        )

        self.send_goal(pose)

    def send_goal(self, pose: PoseStamped):
        if self.pending_send:
            return

        # Cancel previous goal if one exists
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling previous Nav2 goal")
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(lambda _: self._actually_send_goal(pose))
            self.pending_send = True
        else:
            self._actually_send_goal(pose)
            self.pending_send = True

    def _actually_send_goal(self, pose: PoseStamped):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_cb
        )
        send_future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        self.pending_send = False
        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error("No goal handle returned")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info("Nav2 goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_cb)

    def feedback_cb(self, feedback_msg):
        # keep quiet unless you want verbose logs
        pass

    def result_cb(self, future):
        try:
            result = future.result()
            self.get_logger().info(f"Nav2 goal finished with status: {result.status}")
        except Exception as e:
            self.get_logger().warn(f"Failed to get Nav2 result: {e}")
        finally:
            self.current_goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = TargetToNav2Follower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()