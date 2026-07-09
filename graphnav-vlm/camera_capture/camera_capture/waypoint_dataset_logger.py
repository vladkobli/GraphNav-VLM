#!/usr/bin/env python3

import json
import math
import re
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from camera_capture.camera_capture import SnapshotCaptureNode


class NullSignals:
    class _Emitter:
        def emit(self, _text):
            pass

    status_update = _Emitter()


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    q = PoseStamped().pose.orientation
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WaypointDatasetLogger(SnapshotCaptureNode):
    def __init__(self):
        super().__init__(NullSignals(), node_name="waypoint_dataset_logger")

        self.declare_parameter("waypoint_topic", "/goal_pose")
        self.declare_parameter("start_topic", "/waypoint_logging/start")
        self.declare_parameter("reset_topic", "/waypoint_logging/reset")
        self.declare_parameter("remove_nearest_topic", "/waypoint_logging/remove_nearest")
        self.declare_parameter("remove_nearest_radius", 0.75)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("nodes_dir", "/rgbd_camera_intel_dev/src/nodes")
        self.declare_parameter("arrival_wait_sec", 2.4)
        self.declare_parameter("post_snapshot_wait_sec", 0.0)
        self.declare_parameter("snapshot_yaw_offsets_degrees", [0.0, 30.0, 60.0])
        self.declare_parameter("nav2_action_name", "navigate_to_pose")
        self.declare_parameter("tf_timeout_sec", 1.0)
        self.declare_parameter("odom_topic", "/odometry/local")
        self.declare_parameter("stopped_linear_velocity", 0.02)
        self.declare_parameter("stopped_angular_velocity", 0.03)
        self.declare_parameter("stop_wait_timeout_sec", 1.5)
        self.declare_parameter("stop_cmd_vel_topic", "/panther/cmd_vel")
        self.declare_parameter("stop_command_duration_sec", 0.25)

        self.waypoint_topic = self.get_parameter("waypoint_topic").value
        self.start_topic = self.get_parameter("start_topic").value
        self.reset_topic = self.get_parameter("reset_topic").value
        self.remove_nearest_topic = self.get_parameter("remove_nearest_topic").value
        self.remove_nearest_radius = float(
            self.get_parameter("remove_nearest_radius").value
        )
        self.global_frame = self.get_parameter("global_frame").value
        self.robot_frame = self.get_parameter("robot_frame").value
        self.nodes_dir = Path(self.get_parameter("nodes_dir").value)
        self.arrival_wait_sec = float(self.get_parameter("arrival_wait_sec").value)
        self.post_snapshot_wait_sec = float(
            self.get_parameter("post_snapshot_wait_sec").value
        )
        self.snapshot_yaw_offsets_degrees = [
            float(offset)
            for offset in self.get_parameter("snapshot_yaw_offsets_degrees").value
        ]
        action_name = self.get_parameter("nav2_action_name").value
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        odom_topic = self.get_parameter("odom_topic").value
        self.stopped_linear_velocity = float(
            self.get_parameter("stopped_linear_velocity").value
        )
        self.stopped_angular_velocity = float(
            self.get_parameter("stopped_angular_velocity").value
        )
        self.stop_wait_timeout_sec = float(
            self.get_parameter("stop_wait_timeout_sec").value
        )
        self.stop_cmd_vel_topic = self.get_parameter("stop_cmd_vel_topic").value
        self.stop_command_duration_sec = float(
            self.get_parameter("stop_command_duration_sec").value
        )

        self.nodes_dir.mkdir(parents=True, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, action_name)
        marker_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/waypoint_logging/markers",
            marker_qos,
        )
        self.stop_cmd_pub = self.create_publisher(
            TwistStamped,
            self.stop_cmd_vel_topic,
            10,
        )
        self.waypoints = []
        self.active_waypoints = []
        self.active_node_ids = []
        self.active_waypoint_index = None
        self.waypoints_lock = threading.Lock()
        self.running = False
        self.latest_odom = None
        self.latest_odom_lock = threading.Lock()

        self.create_subscription(PoseStamped, self.waypoint_topic, self.waypoint_cb, 10)
        self.create_subscription(Empty, self.start_topic, self.start_topic_cb, 10)
        self.create_subscription(Empty, self.reset_topic, self.reset_topic_cb, 10)
        self.create_subscription(
            PointStamped,
            self.remove_nearest_topic,
            self.remove_nearest_cb,
            10,
        )
        self.create_subscription(Odometry, odom_topic, self.odom_cb, 10)
        self.create_service(Trigger, "/waypoint_logging/start", self.start_srv_cb)
        self.create_service(Trigger, "/waypoint_logging/clear", self.clear_srv_cb)

        # Remove RealSense image sources for waypoint dataset logging.
        # This logger should only capture the four GMSL cameras.
        self.image_sources = [
            source
            for source in self.image_sources
            if not source["name"].startswith("realsense_")
        ]

        for sub in list(self.subscribers):
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
        self.subscribers = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for source in self.image_sources:
            name = source["name"]
            topic = source["topic"]
            self.get_logger().info(f"  {name}: {topic}")
            sub = self.create_subscription(
                Image,
                topic,
                lambda msg, source_name=name: self.image_callback(source_name, msg),
                qos,
            )
            self.subscribers.append(sub)

        self.get_logger().info(
            "Waypoint dataset logger ready. In RViz, use '2D Goal Pose' to add "
            f"waypoints on {self.waypoint_topic}. Start with: "
            f"ros2 topic pub --once {self.start_topic} std_msgs/msg/Empty {{}}. "
            "Use the Publish Point tool to remove the nearest queued waypoint. "
            f"Next dataset node will be {self.node_name(self.next_node_id())}."
        )

    def waypoint_cb(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose
        if not pose.header.frame_id:
            pose.header.frame_id = self.global_frame

        with self.waypoints_lock:
            self.waypoints.append(pose)
            count = len(self.waypoints)
            node_id = self.next_node_id() + count - 1

        yaw = yaw_from_quaternion(pose.pose.orientation)
        self.get_logger().info(
            f"Added {self.node_name(node_id)}: frame={pose.header.frame_id} "
            f"x={pose.pose.position.x:.3f} y={pose.pose.position.y:.3f} yaw={yaw:.3f}"
        )
        self.publish_current_markers()

    def start_topic_cb(self, _msg):
        self.start_run()

    def reset_topic_cb(self, _msg):
        self.clear_pending_waypoints()

    def remove_nearest_cb(self, msg):
        removed = self.remove_nearest_pending_waypoint(msg)
        if removed is None:
            self.get_logger().warn(
                "No queued waypoint close enough to remove near "
                f"x={msg.point.x:.3f} y={msg.point.y:.3f}."
            )
            return

        self.get_logger().info(
            f"Removed queued waypoint {removed} near x={msg.point.x:.3f} y={msg.point.y:.3f}."
        )

    def start_srv_cb(self, _request, response):
        accepted = self.start_run()
        response.success = accepted
        response.message = "Started waypoint dataset logging." if accepted else (
            "Could not start. Already running or no waypoints have been added."
        )
        return response

    def clear_srv_cb(self, _request, response):
        cleared = self.clear_pending_waypoints()
        response.success = True
        response.message = f"Cleared {cleared} waypoint(s)."
        self.get_logger().info(response.message)
        return response

    def start_run(self):
        with self.waypoints_lock:
            if self.running:
                self.get_logger().warn("Waypoint run is already active.")
                return False

            waypoints = list(self.waypoints)

            if not waypoints:
                self.get_logger().warn("No waypoints added yet.")
                return False

            self.waypoints.clear()
            self.active_waypoints = list(waypoints)
            first_node_id = self.next_node_id()
            self.active_node_ids = [
                first_node_id + offset for offset in range(len(waypoints))
            ]
            self.active_waypoint_index = None
            self.running = True

        self.publish_current_markers()
        thread = threading.Thread(
            target=self.run_waypoints,
            args=(waypoints, self.active_node_ids),
            daemon=True,
        )
        thread.start()
        return True

    def run_waypoints(self, waypoints, node_ids):
        run_manifest_path = None
        try:
            run_manifest_path = self.create_run_manifest(waypoints, node_ids)
            self.get_logger().info("Waiting for Nav2 navigate_to_pose action server...")
            if not self.nav_client.wait_for_server(timeout_sec=30.0):
                self.get_logger().error("Nav2 navigate_to_pose action server not available.")
                self.finalize_run_manifest(
                    run_manifest_path,
                    "failed",
                    "Nav2 navigate_to_pose action server not available.",
                )
                return

            for index, waypoint in enumerate(waypoints, start=1):
                node_id = node_ids[index - 1]
                with self.waypoints_lock:
                    self.active_waypoint_index = index
                self.publish_current_markers()

                yaw = yaw_from_quaternion(waypoint.pose.orientation)
                folder = self.make_node_folder(node_id)
                node_id = self.node_id_from_folder(folder)
                node_ids[index - 1] = node_id
                with self.waypoints_lock:
                    if index <= len(self.active_node_ids):
                        self.active_node_ids[index - 1] = node_id
                self.write_metadata(folder, node_id, waypoint, yaw)
                self.update_run_manifest_waypoint(
                    run_manifest_path,
                    index,
                    status="in_progress",
                    node_folder=str(folder),
                    node_id=node_id,
                )

                if index > 1:
                    self.get_logger().info(
                        f"Rotating in place toward {self.node_name(node_id)} yaw before driving."
                    )
                    if not self.rotate_in_place_to_goal_yaw(waypoint):
                        self.get_logger().error(
                            "Initial yaw alignment failed before "
                            f"{self.node_name(node_id)}; stopping run."
                        )
                        self.update_run_manifest_waypoint(
                            run_manifest_path,
                            index,
                            status="failed",
                        )
                        self.finalize_run_manifest(
                            run_manifest_path,
                            "failed",
                            f"Initial yaw alignment failed before {self.node_name(node_id)}.",
                        )
                        return

                self.get_logger().info(
                    f"Navigating to {self.node_name(node_id)} ({index}/{len(waypoints)})"
                )
                if not self.navigate_to_pose(waypoint):
                    self.get_logger().error(
                        f"Navigation failed at {self.node_name(node_id)}; stopping run."
                    )
                    self.update_run_manifest_waypoint(
                        run_manifest_path,
                        index,
                        status="failed",
                    )
                    self.finalize_run_manifest(
                        run_manifest_path,
                        "failed",
                        f"Navigation failed at {self.node_name(node_id)}.",
                    )
                    return

                for snapshot_index, yaw_offset in enumerate(
                    self.snapshot_yaw_offsets_degrees,
                    start=1,
                ):
                    if snapshot_index > 1:
                        rotated = self.rotated_pose(
                            waypoint,
                            math.radians(yaw_offset),
                        )
                        self.get_logger().info(
                            f"Rotating to +{yaw_offset:.1f} degrees at "
                            f"{self.node_name(node_id)}"
                        )
                        if not self.navigate_to_pose(rotated):
                            self.get_logger().error(
                                f"+{yaw_offset:.1f}-degree rotation failed at "
                                f"{self.node_name(node_id)}; stopping run."
                            )
                            self.update_run_manifest_waypoint(
                                run_manifest_path,
                                index,
                                status="failed",
                            )
                            self.finalize_run_manifest(
                                run_manifest_path,
                                "failed",
                                f"+{yaw_offset:.1f}-degree rotation failed at "
                                f"{self.node_name(node_id)}.",
                            )
                            return

                    reason = f"snapshot {snapshot_index}"
                    if not self.prepare_for_snapshot(run_manifest_path, index, reason):
                        return
                    self.get_logger().info(
                        f"Capturing {reason} for {self.node_name(node_id)} "
                        f"at +{yaw_offset:.1f} degrees"
                    )
                    filenames = self.save_snapshot_frames(
                        folder,
                        snapshot_index=snapshot_index - 1,
                    )
                    self.append_snapshot_metadata(
                        folder,
                        f"snapshot_{snapshot_index}",
                        filenames,
                        yaw_offset_degrees=yaw_offset,
                    )
                    self.wait_after_snapshot(index, snapshot_index)
                self.update_run_manifest_waypoint(
                    run_manifest_path,
                    index,
                    status="completed",
                )
                self.publish_current_markers()

            self.finalize_run_manifest(run_manifest_path, "completed")
            self.get_logger().info("Waypoint dataset logging run complete.")
        except Exception as exc:
            self.finalize_run_manifest(run_manifest_path, "failed", str(exc))
            raise
        finally:
            with self.waypoints_lock:
                self.running = False
                self.active_waypoints.clear()
                self.active_node_ids.clear()
                self.active_waypoint_index = None
            self.publish_current_markers()

    def navigate_to_pose(self, pose):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header = pose.header
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose = pose.pose

        goal_handle = self.wait_for_future(self.nav_client.send_goal_async(goal))

        if goal_handle is None or not goal_handle.accepted:
            return False

        result = self.wait_for_future(goal_handle.get_result_async())
        return result is not None and result.status == GoalStatus.STATUS_SUCCEEDED

    def rotated_pose(self, pose, yaw_delta):
        rotated = PoseStamped()
        rotated.header = deepcopy(pose.header)
        rotated.pose = deepcopy(pose.pose)
        yaw = normalize_angle(yaw_from_quaternion(pose.pose.orientation) + yaw_delta)
        rotated.pose.orientation = quaternion_from_yaw(yaw)
        return rotated

    def rotate_in_place_to_goal_yaw(self, goal_pose):
        current_pose = self.current_robot_pose_stamped()
        if current_pose is None:
            return False

        target = PoseStamped()
        target.header = current_pose.header
        target.pose.position = deepcopy(current_pose.pose.position)
        target.pose.orientation = deepcopy(goal_pose.pose.orientation)
        return self.navigate_to_pose(target)

    def current_robot_pose_stamped(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not look up robot TF pose for yaw alignment: {exc}")
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = transform.header.stamp
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def save_snapshot_frames(self, folder, snapshot_index: int = 0):
        folder.mkdir(parents=True, exist_ok=True)

        frame_mapping = {
            "gmsl_camera2_color_fullfov": 1,
            "gmsl_camera0_color_fullfov": 4,
            "gmsl_camera1_color_fullfov": 7,
            "gmsl_camera3_color_fullfov": 10,
        }

        filenames = []
        missing_sources = []

        with self.latest_lock:
            images_copy = dict(self.latest_images)

        for source_name, base_index in frame_mapping.items():
            frame_number = base_index + snapshot_index
            filename = folder / f"frame{frame_number}.png"

            if source_name not in images_copy:
                missing_sources.append(source_name)
                continue

            msg = images_copy[source_name]
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            image_to_save = self.prepare_image_for_saving(cv_img, msg.encoding)

            if not cv2.imwrite(str(filename), image_to_save):
                self.get_logger().error(f"Failed to write image for {source_name} to {filename}")
                continue

            filenames.append(str(filename.name))

        if missing_sources:
            self.get_logger().warn(
                "Missing images for sources: "
                + ", ".join(missing_sources)
                + ". Available sources: "
                + ", ".join(sorted(images_copy.keys()))
            )

        return filenames

    def wait_for_future(self, future):
        event = threading.Event()
        future.add_done_callback(lambda _future: event.set())
        event.wait()
        return future.result()

    def wait_after_snapshot(self, waypoint_index, snapshot_index):
        if self.post_snapshot_wait_sec <= 0.0:
            return

        self.get_logger().info(
            f"Waiting {self.post_snapshot_wait_sec:.1f}s after snapshot "
            f"{snapshot_index} at waypoint {waypoint_index}."
        )
        self.sleep_seconds(self.post_snapshot_wait_sec)

    def prepare_for_snapshot(self, run_manifest_path, waypoint_index, reason):
        self.publish_stop_command()
        if self.wait_for_stable_stop(reason):
            return True

        self.get_logger().warn(
            f"Robot did not settle before {reason} at waypoint {waypoint_index}; "
            "publishing one more stop command and capturing after the fixed settle delay."
        )
        self.publish_stop_command()
        self.sleep_seconds(self.arrival_wait_sec)
        return True

    def publish_stop_command(self):
        if self.stop_command_duration_sec <= 0.0:
            return

        self.get_logger().info(
            f"Publishing zero velocity on {self.stop_cmd_vel_topic} for "
            f"{self.stop_command_duration_sec:.2f}s before capture."
        )

        start_time = self.get_clock().now()
        while rclpy.ok():
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.robot_frame
            self.stop_cmd_pub.publish(msg)

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= self.stop_command_duration_sec:
                break

            self.sleep_seconds(0.05)

    def odom_cb(self, msg):
        with self.latest_odom_lock:
            self.latest_odom = msg

    def wait_for_stable_stop(self, reason):
        self.get_logger().info(
            f"Waiting for robot to be still for {self.arrival_wait_sec:.1f}s before {reason}."
        )

        start_time = self.get_clock().now()
        stable_since = None

        while rclpy.ok():
            now = self.get_clock().now()
            elapsed = (now - start_time).nanoseconds / 1e9
            if elapsed > self.stop_wait_timeout_sec:
                self.get_logger().warn(
                    "Timed out waiting for odometry to settle before "
                    f"{reason}. Current odometry is still above the stopped thresholds."
                )
                self.publish_stop_command()
                return False

            if self.robot_is_stopped():
                if stable_since is None:
                    stable_since = now
                stable_time = (now - stable_since).nanoseconds / 1e9
                if stable_time >= self.arrival_wait_sec:
                    return True
            else:
                stable_since = None

            self.sleep_seconds(0.05)

        return False

    def robot_is_stopped(self):
        with self.latest_odom_lock:
            odom = self.latest_odom

        if odom is None:
            return False

        twist = odom.twist.twist
        linear_speed = math.sqrt(
            twist.linear.x * twist.linear.x
            + twist.linear.y * twist.linear.y
            + twist.linear.z * twist.linear.z
        )
        angular_speed = math.sqrt(
            twist.angular.x * twist.angular.x
            + twist.angular.y * twist.angular.y
            + twist.angular.z * twist.angular.z
        )

        return (
            linear_speed <= self.stopped_linear_velocity
            and angular_speed <= self.stopped_angular_velocity
        )

    def publish_waypoint_markers(self, waypoints):
        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())

        first_node_id = self.next_node_id()
        for offset, pose in enumerate(waypoints):
            node_id = first_node_id + offset
            markers.markers.append(self.arrow_marker(node_id, pose, "queued"))
            markers.markers.append(self.text_marker(node_id, pose, "queued"))

        self.marker_pub.publish(markers)

    def publish_current_markers(self):
        with self.waypoints_lock:
            queued_waypoints = list(self.waypoints)
            active_waypoints = list(self.active_waypoints)
            active_node_ids = list(self.active_node_ids)
            active_waypoint_index = self.active_waypoint_index

        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())

        for index, pose in enumerate(active_waypoints, start=1):
            node_id = active_node_ids[index - 1] if index <= len(active_node_ids) else index
            status = "completed" if (
                active_waypoint_index is not None and index < active_waypoint_index
            ) else "active" if index == active_waypoint_index else "active_pending"
            markers.markers.append(self.arrow_marker(node_id, pose, status))
            markers.markers.append(self.text_marker(node_id, pose, status))

        first_queued_node_id = self.next_node_id()
        if active_node_ids:
            first_queued_node_id = max(first_queued_node_id, max(active_node_ids) + 1)
        for offset, pose in enumerate(queued_waypoints):
            node_id = first_queued_node_id + offset
            markers.markers.append(self.arrow_marker(node_id, pose, "queued"))
            markers.markers.append(self.text_marker(node_id, pose, "queued"))

        self.marker_pub.publish(markers)

    def clear_waypoint_markers(self):
        markers = MarkerArray()
        markers.markers.append(self.delete_all_marker())
        self.marker_pub.publish(markers)

    def delete_all_marker(self):
        marker = Marker()
        marker.action = Marker.DELETEALL
        return marker

    def arrow_marker(self, index, pose, status):
        marker = Marker()
        marker.header.frame_id = pose.header.frame_id or self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoint_arrows"
        marker.id = index
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = deepcopy(pose.pose)
        marker.pose.position.z = 0.18
        marker.scale.x = 0.65
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        self.apply_marker_color(marker, status)
        marker.color.a = 0.95
        return marker

    def text_marker(self, index, pose, status):
        marker = Marker()
        marker.header.frame_id = pose.header.frame_id or self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoint_labels"
        marker.id = index
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = pose.pose.position.x
        marker.pose.position.y = pose.pose.position.y
        marker.pose.position.z = 0.55
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.35
        self.apply_marker_color(marker, status)
        marker.color.a = 1.0
        marker.text = self.node_name(index)
        return marker

    def apply_marker_color(self, marker, status):
        if status == "active":
            marker.color.r = 1.0
            marker.color.g = 0.65
            marker.color.b = 0.0
        elif status == "completed":
            marker.color.r = 0.35
            marker.color.g = 0.9
            marker.color.b = 0.35
        elif status == "active_pending":
            marker.color.r = 0.55
            marker.color.g = 0.55
            marker.color.b = 0.55
        else:
            marker.color.r = 0.0
            marker.color.g = 0.85
            marker.color.b = 1.0

    def clear_pending_waypoints(self):
        with self.waypoints_lock:
            cleared = len(self.waypoints)
            self.waypoints.clear()

        self.publish_current_markers()
        return cleared

    def remove_nearest_pending_waypoint(self, point_msg):
        with self.waypoints_lock:
            if not self.waypoints:
                return None

            nearest_index = None
            nearest_distance = None
            for index, waypoint in enumerate(self.waypoints):
                distance = math.hypot(
                    waypoint.pose.position.x - point_msg.point.x,
                    waypoint.pose.position.y - point_msg.point.y,
                )
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_index = index

            if (
                nearest_index is None
                or nearest_distance is None
                or nearest_distance > self.remove_nearest_radius
            ):
                return None

            removed_index = nearest_index + 1
            self.waypoints.pop(nearest_index)

        self.publish_current_markers()
        return removed_index

    def next_node_id(self):
        highest = 0
        patterns = (
            re.compile(r"^n(\d+)$"),
            re.compile(r"^point_(\d+)$"),
        )
        for path in self.nodes_dir.iterdir():
            if not path.is_dir():
                continue
            for pattern in patterns:
                match = pattern.match(path.name)
                if match:
                    highest = max(highest, int(match.group(1)))
                    break
        return highest + 1

    def node_name(self, node_id):
        return f"n{node_id:02d}"

    def make_node_folder(self, node_id):
        while True:
            folder = self.nodes_dir / self.node_name(node_id)
            if not folder.exists():
                return folder
            node_id += 1

    def node_id_from_folder(self, folder):
        match = re.match(r"^n(\d+)$", folder.name)
        if match:
            return int(match.group(1))

        match = re.match(r"^point_(\d+)$", folder.name)
        if match:
            return int(match.group(1))
        return self.next_node_id()

    def write_metadata(self, folder, node_id, pose, yaw):
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {
            "node_id": node_id,
            "node_name": self.node_name(node_id),
            "created_at": datetime.now().isoformat(),
            "planned_frame_id": pose.header.frame_id,
            "planned_pose": {
                "x": pose.pose.position.x,
                "y": pose.pose.position.y,
                "z": pose.pose.position.z,
                "yaw": yaw,
                "orientation": {
                    "x": pose.pose.orientation.x,
                    "y": pose.pose.orientation.y,
                    "z": pose.pose.orientation.z,
                    "w": pose.pose.orientation.w,
                },
            },
            "snapshots": [],
        }
        with (folder / "metadata.json").open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def create_run_manifest(self, waypoints, node_ids):
        created_at = datetime.now()
        run_id = created_at.strftime("run_%Y%m%d_%H%M%S")
        manifest_path = self.nodes_dir / f"{run_id}_waypoints.json"
        manifest = {
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "updated_at": created_at.isoformat(),
            "status": "running",
            "waypoint_topic": self.waypoint_topic,
            "global_frame": self.global_frame,
            "robot_frame": self.robot_frame,
            "nodes_dir": str(self.nodes_dir),
            "waypoint_count": len(waypoints),
            "waypoints": [
                self.waypoint_manifest_entry(index, node_ids[index - 1], waypoint)
                for index, waypoint in enumerate(waypoints, start=1)
            ],
        }
        self.write_json(manifest_path, manifest)
        self.get_logger().info(f"Run waypoint manifest: {manifest_path}")
        return manifest_path

    def waypoint_manifest_entry(self, index, node_id, pose):
        yaw = yaw_from_quaternion(pose.pose.orientation)
        return {
            "id": index,
            "node_id": node_id,
            "node_name": self.node_name(node_id),
            "status": "pending",
            "frame_id": pose.header.frame_id or self.global_frame,
            "pose": {
                "x": pose.pose.position.x,
                "y": pose.pose.position.y,
                "z": pose.pose.position.z,
                "yaw": yaw,
                "orientation": {
                    "x": pose.pose.orientation.x,
                    "y": pose.pose.orientation.y,
                    "z": pose.pose.orientation.z,
                    "w": pose.pose.orientation.w,
                },
            },
        }

    def update_run_manifest_waypoint(
        self,
        manifest_path,
        index,
        status,
        node_folder=None,
        node_id=None,
    ):
        manifest = self.read_json(manifest_path)
        waypoint = manifest["waypoints"][index - 1]
        waypoint["status"] = status
        waypoint["updated_at"] = datetime.now().isoformat()
        if node_folder is not None:
            waypoint["node_folder"] = node_folder
        if node_id is not None:
            waypoint["node_id"] = node_id
            waypoint["node_name"] = self.node_name(node_id)
        manifest["updated_at"] = datetime.now().isoformat()
        self.write_json(manifest_path, manifest)

    def finalize_run_manifest(self, manifest_path, status, error=None):
        if manifest_path is None:
            return

        manifest = self.read_json(manifest_path)
        manifest["status"] = status
        manifest["finished_at"] = datetime.now().isoformat()
        manifest["updated_at"] = manifest["finished_at"]
        if error is not None:
            manifest["error"] = error
        self.write_json(manifest_path, manifest)

    def read_json(self, path):
        with path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file)

    def write_json(self, path, data):
        with path.open("w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=2)

    def append_snapshot_metadata(
        self,
        folder,
        label,
        filenames,
        yaw_offset_degrees=None,
    ):
        metadata_path = folder / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        snapshot = {
            "label": label,
            "captured_at": datetime.now().isoformat(),
            "filenames": filenames,
            "actual_robot_pose": self.lookup_robot_pose(),
        }
        if yaw_offset_degrees is not None:
            snapshot["yaw_offset_degrees"] = yaw_offset_degrees

        metadata["snapshots"].append(snapshot)

        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2)

    def lookup_robot_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not record robot TF pose: {exc}")
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        return {
            "frame_id": self.global_frame,
            "child_frame_id": self.robot_frame,
            "stamp": {
                "sec": transform.header.stamp.sec,
                "nanosec": transform.header.stamp.nanosec,
            },
            "x": t.x,
            "y": t.y,
            "z": t.z,
            "yaw": yaw_from_quaternion(q),
            "orientation": {
                "x": q.x,
                "y": q.y,
                "z": q.z,
                "w": q.w,
            },
        }

    def sleep_seconds(self, seconds):
        event = threading.Event()
        event.wait(seconds)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointDatasetLogger()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
