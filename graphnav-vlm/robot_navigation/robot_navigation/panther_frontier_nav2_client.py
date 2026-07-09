#!/usr/bin/env python3

import math
import threading
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener

from frontier_interfaces.srv import FrontierGoal


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def set_yaw(pose: PoseStamped, yaw: float) -> PoseStamped:
    pose.pose.orientation.x = 0.0
    pose.pose.orientation.y = 0.0
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose


@dataclass
class BlacklistedGoal:
    x: float
    y: float
    stamp_sec: float


class PantherFrontierNav2Client(Node):
    """Bridge adrian-soch/frontier_exploration /frontier_pose service to Nav2.

    This node does not command /cmd_vel directly. It only sends NavigateToPose goals,
    so your existing Nav2 config/remap to /panther/cmd_vel remains the safety boundary.
    """

    def __init__(self):
        super().__init__("panther_frontier_nav2_client")

        self.declare_parameter("frontier_service", "/frontier_pose")
        self.declare_parameter("nav2_action", "navigate_to_pose")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("auto_start", False)
        self.declare_parameter("timer_period_sec", 2.0)
        self.declare_parameter("goal_rank_min", 0)
        self.declare_parameter("goal_rank_max", 5)
        self.declare_parameter("min_goal_distance", 0.8)
        self.declare_parameter("same_goal_distance", 0.6)
        self.declare_parameter("blacklist_radius", 0.8)
        self.declare_parameter("blacklist_timeout_sec", 90.0)
        self.declare_parameter("nav_result_timeout_sec", 180.0)
        self.declare_parameter("startup_delay_sec", 0.0)
        self.declare_parameter("tf_timeout_sec", 1.0)
        self.declare_parameter("force_goal_yaw_from_robot", True)

        self.frontier_service = str(self.get_parameter("frontier_service").value)
        self.nav2_action = str(self.get_parameter("nav2_action").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.enabled = bool(self.get_parameter("auto_start").value)
        self.timer_period_sec = float(self.get_parameter("timer_period_sec").value)
        self.goal_rank_min = int(self.get_parameter("goal_rank_min").value)
        self.goal_rank_max = int(self.get_parameter("goal_rank_max").value)
        self.min_goal_distance = float(self.get_parameter("min_goal_distance").value)
        self.same_goal_distance = float(self.get_parameter("same_goal_distance").value)
        self.blacklist_radius = float(self.get_parameter("blacklist_radius").value)
        self.blacklist_timeout_sec = float(self.get_parameter("blacklist_timeout_sec").value)
        self.nav_result_timeout_sec = float(self.get_parameter("nav_result_timeout_sec").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.force_goal_yaw_from_robot = bool(self.get_parameter("force_goal_yaw_from_robot").value)
        self.start_time_sec = self.now_sec()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.frontier_client = self.create_client(FrontierGoal, self.frontier_service)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav2_action)

        self.latest_map: Optional[OccupancyGrid] = None
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, 10)

        self.enable_pub = self.create_publisher(Bool, "~/enabled", 1)
        self.current_goal_pub = self.create_publisher(PoseStamped, "~/current_goal", 1)
        self.create_service(SetBool, "~/set_enabled", self.set_enabled_cb)
        self.create_service(Trigger, "~/start", self.start_cb)
        self.create_service(Trigger, "~/stop", self.stop_cb)
        self.create_service(Trigger, "~/step_once", self.step_once_cb)

        self.goal_active = False
        self.current_goal: Optional[PoseStamped] = None
        self.blacklist: List[BlacklistedGoal] = []
        self.lock = threading.Lock()

        self.timer = self.create_timer(self.timer_period_sec, self.timer_cb)
        self.get_logger().info(
            f"Ready. enabled={self.enabled}. Frontier service={self.frontier_service}, "
            f"Nav2 action={self.nav2_action}, map={self.map_topic}, "
            f"frames={self.global_frame}->{self.robot_frame}, "
            f"startup_delay={self.startup_delay_sec:.1f}s"
        )

    def map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg

    def set_enabled_cb(self, req, resp):
        self.enabled = bool(req.data)
        resp.success = True
        resp.message = f"frontier exploration enabled={self.enabled}"
        self.publish_enabled()
        return resp

    def start_cb(self, _req, resp):
        self.enabled = True
        resp.success = True
        resp.message = "frontier exploration started"
        self.publish_enabled()
        return resp

    def stop_cb(self, _req, resp):
        self.enabled = False
        resp.success = True
        resp.message = "frontier exploration stopped; current Nav2 goal is not canceled by this node"
        self.publish_enabled()
        return resp

    def step_once_cb(self, _req, resp):
        if self.in_startup_delay():
            remaining = self.startup_delay_sec - (self.now_sec() - self.start_time_sec)
            resp.success = False
            resp.message = f"startup delay active for another {remaining:.1f}s"
            return resp
        if self.goal_active:
            resp.success = False
            resp.message = "a Nav2 goal is already active"
            return resp
        threading.Thread(target=self.find_and_send_goal, daemon=True).start()
        resp.success = True
        resp.message = "requested one frontier goal"
        return resp

    def publish_enabled(self):
        msg = Bool()
        msg.data = self.enabled
        self.enable_pub.publish(msg)

    def timer_cb(self):
        self.publish_enabled()
        if not self.enabled:
            return
        if self.in_startup_delay():
            return
        if self.goal_active:
            return
        threading.Thread(target=self.find_and_send_goal, daemon=True).start()

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def in_startup_delay(self) -> bool:
        return (self.now_sec() - self.start_time_sec) < self.startup_delay_sec

    def robot_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(f"TF unavailable {self.global_frame}->{self.robot_frame}: {exc}")
            return None

        pose = PoseStamped()
        pose.header = tf.header
        pose.pose.position.x = tf.transform.translation.x
        pose.pose.position.y = tf.transform.translation.y
        pose.pose.position.z = tf.transform.translation.z
        pose.pose.orientation = tf.transform.rotation
        return pose

    def dist_to_robot(self, pose: PoseStamped) -> Optional[float]:
        rp = self.robot_pose()
        if rp is None:
            return None
        return math.hypot(pose.pose.position.x - rp.pose.position.x, pose.pose.position.y - rp.pose.position.y)

    def prune_blacklist(self):
        t = self.now_sec()
        self.blacklist = [g for g in self.blacklist if (t - g.stamp_sec) < self.blacklist_timeout_sec]

    def is_blacklisted(self, pose: PoseStamped) -> bool:
        self.prune_blacklist()
        for g in self.blacklist:
            if math.hypot(pose.pose.position.x - g.x, pose.pose.position.y - g.y) <= self.blacklist_radius:
                return True
        return False

    def add_blacklist(self, pose: PoseStamped):
        self.blacklist.append(BlacklistedGoal(pose.pose.position.x, pose.pose.position.y, self.now_sec()))
        self.get_logger().warn(
            f"Blacklisted frontier around x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f} "
            f"for {self.blacklist_timeout_sec:.0f}s"
        )

    def wait_for_future(self, future, timeout_sec=None):
        event = threading.Event()
        future.add_done_callback(lambda _f: event.set())
        if not event.wait(timeout=timeout_sec):
            return None
        return future.result()

    def call_frontier(self, rank: int) -> Optional[PoseStamped]:
        if not self.frontier_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"Frontier service not available: {self.frontier_service}")
            return None

        req = FrontierGoal.Request()
        req.goal_rank = int(rank)
        result = self.wait_for_future(self.frontier_client.call_async(req), timeout_sec=5.0)
        if result is None:
            self.get_logger().warn(f"No response from {self.frontier_service} for rank {rank}")
            return None

        pose = result.goal_pose
        if not pose.header.frame_id:
            pose.header.frame_id = self.global_frame
        return pose

    def choose_frontier_goal(self) -> Optional[PoseStamped]:
        if self.latest_map is None:
            self.get_logger().warn(f"No {self.map_topic} received yet. Is slam_toolbox running?")
            return None

        robot = self.robot_pose()
        if robot is None:
            return None
        robot_yaw = yaw_from_quat(robot.pose.orientation)

        for rank in range(self.goal_rank_min, self.goal_rank_max + 1):
            goal = self.call_frontier(rank)
            if goal is None:
                continue

            if goal.header.frame_id != self.global_frame:
                self.get_logger().warn(
                    f"Frontier rank {rank} is in frame '{goal.header.frame_id}', expected '{self.global_frame}'. Skipping."
                )
                continue

            d = math.hypot(goal.pose.position.x - robot.pose.position.x, goal.pose.position.y - robot.pose.position.y)
            if d < self.min_goal_distance:
                self.get_logger().info(f"Skipping rank {rank}: too close ({d:.2f} m)")
                continue

            if self.is_blacklisted(goal):
                self.get_logger().info(f"Skipping rank {rank}: blacklisted")
                continue

            if self.force_goal_yaw_from_robot:
                # For exploration, yaw is not critical. Keeping current yaw avoids weird frontier orientations.
                goal = set_yaw(goal, robot_yaw)

            return goal

        return None

    def find_and_send_goal(self):
        with self.lock:
            if self.goal_active:
                return
            self.goal_active = True

        try:
            goal_pose = self.choose_frontier_goal()
            if goal_pose is None:
                self.get_logger().warn("No usable frontier goal found.")
                return

            self.current_goal = goal_pose
            self.current_goal_pub.publish(goal_pose)
            self.get_logger().info(
                f"Sending frontier goal: x={goal_pose.pose.position.x:.2f}, "
                f"y={goal_pose.pose.position.y:.2f}, frame={goal_pose.header.frame_id}"
            )

            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error(f"Nav2 action server not available: {self.nav2_action}")
                return

            nav_goal = NavigateToPose.Goal()
            nav_goal.pose = goal_pose
            nav_goal.pose.header.stamp = self.get_clock().now().to_msg()

            handle = self.wait_for_future(self.nav_client.send_goal_async(nav_goal), timeout_sec=10.0)
            if handle is None or not handle.accepted:
                self.get_logger().warn("Nav2 rejected frontier goal")
                self.add_blacklist(goal_pose)
                return

            result = self.wait_for_future(handle.get_result_async(), timeout_sec=self.nav_result_timeout_sec)
            if result is None:
                self.get_logger().warn("Nav2 result timeout; blacklisting this frontier")
                self.add_blacklist(goal_pose)
                return

            if result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info("Reached frontier goal.")
            else:
                self.get_logger().warn(f"Nav2 failed/aborted/canceled with status={result.status}")
                self.add_blacklist(goal_pose)
        finally:
            with self.lock:
                self.goal_active = False


def main(args=None):
    rclpy.init(args=args)
    node = PantherFrontierNav2Client()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
