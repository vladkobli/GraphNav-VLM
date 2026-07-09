#!/usr/bin/env python3

import json
import math
import re
import threading
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import rclpy
from action_msgs.msg import GoalStatusArray
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener


class Phase(str, Enum):
    INITIALIZATION = "INITIALIZATION"
    EXPLORATION = "EXPLORATION"
    GOAL_NAVIGATION = "GOAL_NAVIGATION"
    FINISHED = "FINISHED"


STOP_RELATIVE_YAW_DEG = {
    "front": 0.0,
    "front_left": 45.0,
    "left": 90.0,
    "back_left": 135.0,
    "back": 180.0,
    "back_right": -135.0,
    "right": -90.0,
    "front_right": -45.0,
}

TASK_STOPWORDS = {
    "a", "an", "and", "around", "at", "by", "find", "for", "go", "in",
    "inside", "into", "is", "locate", "me", "near", "of", "on", "please",
    "robot", "see", "show", "take", "the", "to", "toward", "with",
}

OPEN_ROUTE_PHRASES = [
    "open doorway",
    "open door",
    "open passage",
    "open entrance",
    "clear path",
    "clear route",
    "clear corridor",
    "clear hallway",
    "can walk through",
    "walk through",
    "path through",
    "through or around",
    "ramp leading",
]

WEAK_ROUTE_PHRASES = [
    "hallway",
    "corridor",
    "walkway",
    "path",
    "ramp",
    "stairs",
    "passage",
    "doorway",
    "entrance",
]

CLOSED_ROUTE_PHRASES = [
    "closed door",
    "closed doorway",
    "closed doors",
    "closed wooden door",
    "closed entrance",
    "not confirmed traversable",
]

NO_ROUTE_PHRASES = [
    "no visible traversable",
    "no visible routes",
    "no visible traversable openings",
    "no visible traversable routes",
    "no clear navigation cue",
    "no clear open passage",
    "no clear visible route",
    "no clear traversable",
    "no clear path",
]

EXPECTED_MOONDREAM_PROMPT_MODE = "visual_summary_prompt_v6"


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value, low, high, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, value))


def extract_json_object(text):
    text = str(text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text}")
    return json.loads(match.group(0))


def short_text(text, max_chars=260):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def normalize_rad(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def bool_parameter(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def normalize_for_match(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def phrase_has_negating_prefix(text_l, start_idx):
    prefix = text_l[max(0, start_idx - 45):start_idx]
    return re.search(r"(no|not|without|cannot|can't|only)\s+[^.;,\n]{0,35}$", prefix) is not None


def positive_phrase_in_text(phrases, text_l):
    for phrase in phrases:
        phrase_l = normalize_for_match(phrase)
        if not phrase_l:
            continue
        for match in re.finditer(re.escape(phrase_l), text_l):
            if not phrase_has_negating_prefix(text_l, match.start()):
                return True
    return False


class GraphNavStateMachine(Node):
    def __init__(self):
        super().__init__("graph_nav_state_machine")

        self.declare_parameter("user_prompt", "Find the requested object.")
        self.declare_parameter("target_prompt", "person")
        self.declare_parameter("nodes_dir", "/rgbd_camera_intel_dev/src/nodes")
        self.declare_parameter("state_file_name", "graph_nav_state.json")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("tf_timeout_sec", 1.0)
        self.declare_parameter("phase_topic", "/graph_nav/phase")
        self.declare_parameter("finished_topic", "/graph_nav/finished")
        self.declare_parameter("target_visible_topic", "/graph_nav/target_visible")
        self.declare_parameter("goal_reached_topic", "/graph_nav/goal_reached")
        self.declare_parameter("node_reached_topic", "/graph_nav/node_reached")
        self.declare_parameter("target_point_topic", "/target_point")
        self.declare_parameter("target_visible_from_target_point", True)
        self.declare_parameter("target_visible_min_observations", 3)
        self.declare_parameter("target_visible_window_sec", 1.5)
        self.declare_parameter("target_visible_min_range_m", 0.2)
        self.declare_parameter("target_visible_max_range_m", 8.0)
        self.declare_parameter("target_visible_max_spread_m", 0.0)
        self.declare_parameter("stop_scan_on_target_visible", True)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("tick_sec", 1.0)
        self.declare_parameter("auto_scan_initial_node", True)
        self.declare_parameter("lerobot_server_url", "http://127.0.0.1:8765")
        self.declare_parameter(
            "camera_stop_names",
            ["back", "back_left", "left", "front_left", "front", "front_right", "right", "back_right"],
        )
        self.declare_parameter("settle_after_motion_sec", 0.5)
        self.declare_parameter("settle_before_sweep_sec", 2.0)
        self.declare_parameter("wait_for_fresh_frame_sec", 60.0)
        self.declare_parameter("fresh_frames_to_skip", 3)
        self.declare_parameter("wait_for_fresh_color_frame_sec", 5.0)
        self.declare_parameter("realsense_color_topic", "/camera/driver/color/image_raw")
        self.declare_parameter(
            "realsense_aligned_depth_topic",
            "/camera/driver/aligned_depth_to_color/image_raw",
        )
        self.declare_parameter(
            "realsense_depth_camera_info_topic",
            "/camera/driver/aligned_depth_to_color/camera_info",
        )
        self.declare_parameter("depth_save_formats", ["png", "npz", "preview_png"])
        self.declare_parameter("max_depth_preview_m", 5.0)
        self.declare_parameter("moondream_server_url", "http://127.0.0.1:8766")
        self.declare_parameter("require_moondream_descriptions", True)
        self.declare_parameter("moondream_overwrite_descriptions", False)
        self.declare_parameter("candidate_top_k", 3)
        self.declare_parameter("candidate_memory_nodes", 8)
        self.declare_parameter("next_node_goal_topic", "/graph_nav/next_node_goal_pose")
        self.declare_parameter("nav2_goal_pose_topic", "/goal_pose")
        self.declare_parameter("publish_nav2_goal_pose", False)
        self.declare_parameter("next_node_goal_distance_m", 2.0)
        self.declare_parameter("nav2_status_topic", "/navigate_to_pose/_action/status")
        self.declare_parameter("auto_scan_after_nav2_success", True)
        self.declare_parameter("auto_finish_on_goal_nav2_success", True)
        self.declare_parameter("nav2_success_min_wait_sec", 1.0)
        self.declare_parameter("llm_candidate_selection_enabled", True)
        self.declare_parameter("llm_model", "qwen2.5:7b")
        self.declare_parameter("ollama_base_url", "http://127.0.0.1:11434")
        self.declare_parameter("llm_request_timeout_sec", 60.0)
        self.declare_parameter("llm_candidate_num_predict", 700)

        self.user_prompt = str(self.get_parameter("user_prompt").value)
        self.target_prompt = str(self.get_parameter("target_prompt").value)
        self.nodes_dir = Path(str(self.get_parameter("nodes_dir").value))
        self.state_file = self.nodes_dir / str(self.get_parameter("state_file_name").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.tick_sec = float(self.get_parameter("tick_sec").value)
        self.lerobot_server_url = str(self.get_parameter("lerobot_server_url").value).rstrip("/")
        self.camera_stop_names = [str(v) for v in self.get_parameter("camera_stop_names").value]
        self.target_point_topic = str(self.get_parameter("target_point_topic").value)
        self.target_visible_from_target_point = bool_parameter(
            self.get_parameter("target_visible_from_target_point").value
        )
        self.target_visible_min_observations = int(
            self.get_parameter("target_visible_min_observations").value
        )
        self.target_visible_window_sec = float(
            self.get_parameter("target_visible_window_sec").value
        )
        self.target_visible_min_range_m = float(
            self.get_parameter("target_visible_min_range_m").value
        )
        self.target_visible_max_range_m = float(
            self.get_parameter("target_visible_max_range_m").value
        )
        self.target_visible_max_spread_m = float(
            self.get_parameter("target_visible_max_spread_m").value
        )
        self.stop_scan_on_target_visible = bool_parameter(
            self.get_parameter("stop_scan_on_target_visible").value
        )
        self.settle_after_motion_sec = float(self.get_parameter("settle_after_motion_sec").value)
        self.settle_before_sweep_sec = float(self.get_parameter("settle_before_sweep_sec").value)
        self.wait_for_fresh_frame_sec = float(self.get_parameter("wait_for_fresh_frame_sec").value)
        self.fresh_frames_to_skip = int(self.get_parameter("fresh_frames_to_skip").value)
        self.wait_for_fresh_color_frame_sec = float(
            self.get_parameter("wait_for_fresh_color_frame_sec").value
        )
        self.depth_save_formats = {
            str(v).strip().lower()
            for v in self.get_parameter("depth_save_formats").value
        }
        self.max_depth_preview_m = float(self.get_parameter("max_depth_preview_m").value)
        self.moondream_server_url = str(
            self.get_parameter("moondream_server_url").value
        ).rstrip("/")
        self.require_moondream_descriptions = bool(
            self.get_parameter("require_moondream_descriptions").value
        )
        self.moondream_overwrite_descriptions = bool(
            self.get_parameter("moondream_overwrite_descriptions").value
        )
        self.candidate_top_k = int(self.get_parameter("candidate_top_k").value)
        self.candidate_memory_nodes = int(self.get_parameter("candidate_memory_nodes").value)
        self.next_node_goal_topic = str(self.get_parameter("next_node_goal_topic").value)
        self.nav2_goal_pose_topic = str(self.get_parameter("nav2_goal_pose_topic").value)
        self.publish_nav2_goal_pose = bool_parameter(
            self.get_parameter("publish_nav2_goal_pose").value
        )
        self.next_node_goal_distance_m = float(
            self.get_parameter("next_node_goal_distance_m").value
        )
        self.nav2_status_topic = str(self.get_parameter("nav2_status_topic").value)
        self.auto_scan_after_nav2_success = bool_parameter(
            self.get_parameter("auto_scan_after_nav2_success").value
        )
        self.auto_finish_on_goal_nav2_success = bool_parameter(
            self.get_parameter("auto_finish_on_goal_nav2_success").value
        )
        self.nav2_success_min_wait_sec = float(
            self.get_parameter("nav2_success_min_wait_sec").value
        )
        self.llm_candidate_selection_enabled = bool_parameter(
            self.get_parameter("llm_candidate_selection_enabled").value
        )
        self.llm_model = str(self.get_parameter("llm_model").value)
        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).rstrip("/")
        self.llm_request_timeout_sec = float(
            self.get_parameter("llm_request_timeout_sec").value
        )
        self.llm_candidate_num_predict = int(
            self.get_parameter("llm_candidate_num_predict").value
        )

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_images = {}
        self.latest_receive_time = {}
        self.latest_depth_camera_info = None
        self.latest_lock = threading.Lock()
        self.scan_lock = threading.Lock()
        self.scan_running = False
        self.pending_scan = bool(self.get_parameter("auto_scan_initial_node").value)
        self.current_node_folder = None
        self.current_selected_candidates = []
        self.current_next_node_goal = None
        self.current_llm_view_decision = None
        self.waiting_for_next_node_goal_result = False
        self.next_node_goal_published_at_ns = None
        self.next_node_goal_seen_nav2_goal = False
        self.last_nav2_success_goal_id = None
        self.target_point_observations = deque()
        self.target_visible_confirmed_at_ns = None
        self.target_visible_source = None
        self.target_visible_observation_count = 0
        self.target_visible_last_point = None
        self.last_goal_navigation_success_goal_id = None

        self.target_visible = False
        self.goal_reached = False
        self.finished = False
        self.phase = Phase.INITIALIZATION

        self.phase_pub = self.create_publisher(String, str(self.get_parameter("phase_topic").value), 10)
        self.finished_pub = self.create_publisher(Bool, str(self.get_parameter("finished_topic").value), 10)
        self.cmd_vel_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        goal_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.next_node_goal_pub = self.create_publisher(PoseStamped, self.next_node_goal_topic, goal_qos)
        self.nav2_goal_pose_pub = self.create_publisher(PoseStamped, self.nav2_goal_pose_topic, 10)

        self.create_subscription(Bool, str(self.get_parameter("target_visible_topic").value), self.on_target_visible, 10)
        self.create_subscription(Bool, str(self.get_parameter("goal_reached_topic").value), self.on_goal_reached, 10)
        self.create_subscription(Bool, str(self.get_parameter("node_reached_topic").value), self.on_node_reached, 10)
        if self.target_visible_from_target_point:
            self.create_subscription(PointStamped, self.target_point_topic, self.on_target_point, 10)
        if self.auto_scan_after_nav2_success or self.auto_finish_on_goal_nav2_success:
            self.create_subscription(GoalStatusArray, self.nav2_status_topic, self.on_nav2_status, 10)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.image_sources = [
            {
                "name": "realsense_color",
                "topic": str(self.get_parameter("realsense_color_topic").value),
            },
            {
                "name": "realsense_aligned_depth",
                "topic": str(self.get_parameter("realsense_aligned_depth_topic").value),
            },
        ]
        for source in self.image_sources:
            self.create_subscription(
                Image,
                source["topic"],
                lambda msg, source_name=source["name"]: self.image_callback(source_name, msg),
                qos,
            )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("realsense_depth_camera_info_topic").value),
            self.depth_camera_info_callback,
            qos,
        )

        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info("========== GRAPH NAV: INITIALIZATION ==========")
        self.get_logger().info("Bringup is treated as complete by graph_nav_full.launch.py.")
        self.write_state(initialization_complete=True)
        self.transition_to(Phase.EXPLORATION)
        self.timer = self.create_timer(self.tick_sec, self.tick)

    def image_callback(self, source_name, msg):
        with self.latest_lock:
            self.latest_images[source_name] = msg
            self.latest_receive_time[source_name] = self.get_clock().now()

    def depth_camera_info_callback(self, msg):
        with self.latest_lock:
            self.latest_depth_camera_info = msg

    def on_target_visible(self, msg):
        if bool(msg.data):
            self.confirm_target_visible(source="target_visible_topic")
        elif self.phase == Phase.EXPLORATION:
            self.target_visible = False
            self.target_visible_confirmed_at_ns = None
            self.target_visible_source = None
            self.target_visible_observation_count = 0
            self.target_visible_last_point = None
            self.target_point_observations.clear()
            self.write_state(initialization_complete=True)

    def on_target_point(self, msg):
        if self.phase not in (Phase.EXPLORATION, Phase.GOAL_NAVIGATION):
            return

        point = msg.point
        coords = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(value) for value in coords):
            return

        distance = math.sqrt(coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2)
        if distance < self.target_visible_min_range_m or distance > self.target_visible_max_range_m:
            return

        now_ns = self.get_clock().now().nanoseconds
        observation = {
            "received_ns": now_ns,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": msg.header.frame_id,
            "point": {
                "x": coords[0],
                "y": coords[1],
                "z": coords[2],
            },
            "range_m": distance,
        }
        self.target_point_observations.append(observation)

        window_ns = max(0.1, self.target_visible_window_sec) * 1e9
        while self.target_point_observations:
            age_ns = now_ns - self.target_point_observations[0]["received_ns"]
            if age_ns <= window_ns:
                break
            self.target_point_observations.popleft()

        observation_count = len(self.target_point_observations)
        self.target_visible_observation_count = observation_count
        self.target_visible_last_point = observation
        if observation_count < max(1, self.target_visible_min_observations):
            return

        if not self.target_point_spread_ok():
            return

        self.confirm_target_visible(
            source="target_point",
            point=observation,
            observation_count=observation_count,
        )

    def confirm_target_visible(self, source, point=None, observation_count=None):
        if not self.target_visible:
            self.get_logger().info(
                f"Target visibility confirmed from {source}; entering GOAL_NAVIGATION."
            )
        self.target_visible = True
        self.target_visible_source = source
        self.target_visible_confirmed_at_ns = self.get_clock().now().nanoseconds
        if point is not None:
            self.target_visible_last_point = point
        if observation_count is not None:
            self.target_visible_observation_count = int(observation_count)
        if self.phase == Phase.EXPLORATION:
            self.pending_scan = False
            self.waiting_for_next_node_goal_result = False
            self.transition_to(Phase.GOAL_NAVIGATION)
        else:
            self.write_state(initialization_complete=True)

    def on_goal_reached(self, msg):
        self.goal_reached = bool(msg.data)
        if self.goal_reached and self.phase == Phase.GOAL_NAVIGATION:
            self.transition_to(Phase.FINISHED)

    def on_node_reached(self, msg):
        if bool(msg.data) and self.phase == Phase.EXPLORATION:
            self.get_logger().info("node_reached received; scheduling next graph node scan.")
            self.waiting_for_next_node_goal_result = False
            self.pending_scan = True

    def on_nav2_status(self, msg):
        if self.phase == Phase.GOAL_NAVIGATION and self.auto_finish_on_goal_nav2_success:
            self.maybe_finish_goal_navigation_from_nav2_status(msg)
            return

        if not self.waiting_for_next_node_goal_result or self.phase != Phase.EXPLORATION:
            return

        published_at_ns = self.next_node_goal_published_at_ns
        if published_at_ns is None:
            return

        elapsed_sec = (self.get_clock().now().nanoseconds - published_at_ns) / 1e9
        if elapsed_sec < self.nav2_success_min_wait_sec:
            return

        for status in msg.status_list:
            status_stamp_ns = self.stamp_to_nanoseconds(status.goal_info.stamp)
            if status_stamp_ns is not None and status_stamp_ns + 250_000_000 < published_at_ns:
                continue

            status_code = int(status.status)
            if status_code in (1, 2):
                self.next_node_goal_seen_nav2_goal = True

            if status_code == 4:
                goal_id = bytes(status.goal_info.goal_id.uuid).hex()
                if goal_id == self.last_nav2_success_goal_id:
                    continue
                self.last_nav2_success_goal_id = goal_id
                self.waiting_for_next_node_goal_result = False
                self.next_node_goal_seen_nav2_goal = False
                self.pending_scan = True
                self.get_logger().info(
                    "Nav2 reported next-node goal succeeded; scheduling next graph node scan."
                )
                self.write_state(initialization_complete=True)
                return

    def maybe_finish_goal_navigation_from_nav2_status(self, msg):
        if not self.target_visible or self.target_visible_confirmed_at_ns is None:
            return

        for status in msg.status_list:
            status_stamp_ns = self.stamp_to_nanoseconds(status.goal_info.stamp)
            if (
                status_stamp_ns is not None
                and status_stamp_ns + 250_000_000 < self.target_visible_confirmed_at_ns
            ):
                continue

            if int(status.status) != 4:
                continue

            goal_id = bytes(status.goal_info.goal_id.uuid).hex()
            if goal_id == self.last_goal_navigation_success_goal_id:
                continue

            self.last_goal_navigation_success_goal_id = goal_id
            self.goal_reached = True
            self.get_logger().info(
                "Nav2 reported target-following goal succeeded; marking goal reached."
            )
            self.transition_to(Phase.FINISHED)
            return

    def target_point_spread_ok(self):
        if self.target_visible_max_spread_m <= 0.0:
            return True
        if len(self.target_point_observations) < 2:
            return True

        points = [
            np.array([
                item["point"]["x"],
                item["point"]["y"],
                item["point"]["z"],
            ], dtype=np.float32)
            for item in self.target_point_observations
        ]
        center = np.mean(points, axis=0)
        max_spread = max(float(np.linalg.norm(point - center)) for point in points)
        return max_spread <= self.target_visible_max_spread_m

    def transition_to(self, next_phase):
        if self.phase == next_phase:
            return
        self.phase = next_phase
        self.get_logger().info(f"========== GRAPH NAV: {self.phase.value} ==========")
        if self.phase == Phase.FINISHED:
            self.finished = True
            self.stop_robot()
            self.get_logger().info("Goal reached. FINISHED flag set and robot stop command published.")
        self.publish_flags()
        self.write_state(initialization_complete=True)

    def tick(self):
        self.publish_flags()
        if self.phase == Phase.EXPLORATION:
            self.exploration_tick()
        elif self.phase == Phase.GOAL_NAVIGATION:
            self.goal_navigation_tick()
        elif self.phase == Phase.FINISHED:
            self.stop_robot()

    def exploration_tick(self):
        if self.pending_scan and not self.scan_running:
            self.pending_scan = False
            threading.Thread(target=self.create_node_and_scan_360, daemon=True).start()

    def create_node_and_scan_360(self):
        with self.scan_lock:
            if self.scan_running:
                return
            self.scan_running = True

        folder = None
        try:
            self.check_lerobot_server()
            if not self.wait_until_realsense_available(timeout_sec=120.0):
                raise RuntimeError("RealSense color/depth not available. Check topics and QoS.")

            node_id = self.next_node_id()
            folder = self.make_node_folder(node_id)
            node_id = self.node_id_from_folder(folder)
            self.current_node_folder = folder

            self.write_initial_metadata(folder, node_id)
            self.get_logger().info(f"create_node: created {folder}")

            self.scan_360(folder)
            if self.should_stop_scan_for_target():
                self.mark_metadata_status(folder, "target_visible_during_sweep")
                self.get_logger().info(
                    "Target became visible during scan_360; skipping descriptions, "
                    "candidate generation, and next-node exploration goal."
                )
                return

            self.generate_descriptions(folder)
            self.generate_candidates(folder)
            self.generate_next_node_goal(folder)
            self.mark_metadata_status(folder, "completed")
            self.get_logger().info(f"scan_360 complete: {folder}")

            # TODO analyze_node: ask an LLM if this node is promising for the task.
            # TODO navigate_to_local_goal: subscribe to next_node_goal_topic or enable publish_nav2_goal_pose.

        except Exception as exc:
            self.get_logger().error(f"create_node/scan_360 failed: {exc}")
            if folder is not None:
                self.mark_metadata_status(folder, "failed", error=str(exc))
        finally:
            with self.scan_lock:
                self.scan_running = False
            self.write_state(initialization_complete=True)

    def generate_descriptions(self, folder):
        url = f"{self.moondream_server_url}/describe_node"
        body = json.dumps({
            "node_dir": str(folder),
            "overwrite": self.moondream_overwrite_descriptions,
            "write_metadata": False,
            "task": self.user_prompt,
            "user_prompt": self.user_prompt,
            "target_prompt": self.target_prompt,
        }).encode("utf-8")

        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=300.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            message = (
                f"Could not generate Moondream descriptions at {url}: "
                f"HTTP {exc.code} {exc.reason}: {error_body}"
            )
            if self.require_moondream_descriptions:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            self.record_description_status(folder, "skipped", error=message)
            return None
        except Exception as exc:
            message = f"Could not generate Moondream descriptions at {url}: {exc}"
            if self.require_moondream_descriptions:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            self.record_description_status(folder, "skipped", error=str(exc))
            return None

        if not payload.get("ok", False):
            message = f"Moondream description server failed: {payload}"
            if self.require_moondream_descriptions:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            self.record_description_status(folder, "failed", error=str(payload))
            return payload

        prompt_mode = (payload.get("descriptions") or {}).get("prompt_mode")
        if prompt_mode and prompt_mode != EXPECTED_MOONDREAM_PROMPT_MODE:
            message = (
                "Moondream description server is using prompt_mode "
                f"{prompt_mode!r}, expected {EXPECTED_MOONDREAM_PROMPT_MODE!r}. "
                "Restart the add_description2.py server from the rebuilt source."
            )
            if self.require_moondream_descriptions:
                raise RuntimeError(message)
            self.get_logger().warn(message)
            self.record_description_status(folder, "failed", error=message)
            return payload

        self.apply_description_payload(folder, payload)
        self.get_logger().info(
            "generate_descriptions complete: "
            f"processed={payload.get('processed')}, "
            f"skipped={payload.get('skipped')}, "
            f"missing={payload.get('missing')}"
        )
        return payload

    def apply_description_payload(self, folder, payload):
        path = self.metadata_path(folder)
        if not path.exists():
            raise FileNotFoundError(f"metadata.json not found at {path}")

        metadata = self.read_json(path)
        snapshots = metadata.get("snapshots", [])
        snapshots_by_label = {
            snapshot.get("label"): snapshot
            for snapshot in snapshots
            if snapshot.get("label")
        }
        snapshots_by_stop = {
            (snapshot.get("stop_index"), snapshot.get("stop_name")): snapshot
            for snapshot in snapshots
        }

        for described in payload.get("described_snapshots", []):
            snapshot = snapshots_by_label.get(described.get("label"))
            if snapshot is None:
                snapshot = snapshots_by_stop.get((
                    described.get("stop_index"),
                    described.get("stop_name"),
                ))
            if snapshot is None:
                raise RuntimeError(f"Moondream returned unknown snapshot: {described}")

            snapshot["description"] = described["description"]
            snapshot["semantic_description"] = described["semantic_description"]
            snapshot["description_model"] = described.get(
                "description_model",
                "moondream-2b-2025-04-14-4bit",
            )
            if described.get("description_prompt_mode"):
                snapshot["description_prompt_mode"] = described["description_prompt_mode"]
            snapshot["described_at"] = described.get("described_at", datetime.now().isoformat())

        descriptions = payload.get("descriptions") or {
            "status": "completed" if not payload.get("errors") else "completed_with_errors",
            "updated_at": datetime.now().isoformat(),
            "model": "moondream-2b-2025-04-14-4bit",
            "processed": payload.get("processed"),
            "skipped": payload.get("skipped"),
            "missing": payload.get("missing"),
            "errors": payload.get("errors", []),
        }
        descriptions["server_url"] = self.moondream_server_url
        descriptions["server_wrote_metadata"] = bool(payload.get("wrote_metadata", False))
        metadata["descriptions"] = descriptions
        self.write_json(path, metadata)

    def generate_candidates(self, folder):
        path = self.metadata_path(folder)
        metadata = self.read_json(path)
        task_terms = self.extract_task_terms()
        previous_memory = self.load_previous_node_memory(metadata.get("node_name"))
        repeated_direction_counts = self.previous_selected_direction_counts(previous_memory)

        candidates = []
        for snapshot in sorted(metadata.get("snapshots", []), key=lambda s: s.get("stop_index", 999)):
            candidate = self.build_direction_candidate(
                snapshot=snapshot,
                task_terms=task_terms,
                repeated_direction_counts=repeated_direction_counts,
            )
            candidates.append(candidate)

        selectable = [
            candidate for candidate in candidates
            if candidate.get("visual_audit", {}).get("route_affordance") != "closed"
        ]
        if len(selectable) < self.candidate_top_k:
            selectable = list(candidates)

        ranked = sorted(
            selectable,
            key=lambda c: (
                c.get("selection_score", 0.0),
                c.get("visual_audit", {}).get("target_cue") == "direct",
            ),
            reverse=True,
        )
        selected = ranked[:max(1, self.candidate_top_k)]
        all_ranked = sorted(candidates, key=lambda c: c.get("selection_score", 0.0), reverse=True)
        deterministic_selected = list(selected)

        llm_selection = self.select_candidates_with_llm(
            metadata=metadata,
            all_ranked=all_ranked,
            deterministic_selected=deterministic_selected,
            previous_memory=previous_memory,
        )
        if llm_selection.get("status") == "completed":
            selected = llm_selection.get("selected_candidates", selected)

        now = datetime.now().isoformat()
        candidate_generation = {
            "status": "completed",
            "updated_at": now,
            "algorithm": (
                "llm_qwen_view_selector_v1"
                if llm_selection.get("status") == "completed"
                else "description_keyword_affordance_v1"
            ),
            "task": self.user_prompt,
            "target_prompt": self.target_prompt,
            "task_terms": task_terms,
            "top_k": self.candidate_top_k,
            "previous_nodes_considered": len(previous_memory),
            "previous_memory": previous_memory,
            "deterministic_selected_candidates": deterministic_selected,
            "llm_selection": llm_selection,
            "selected_candidates": selected,
            "all_candidates": all_ranked,
        }
        metadata["candidate_generation"] = candidate_generation
        metadata["analysis"] = {
            "status": "completed",
            "updated_at": now,
            "current_node_promising": bool(
                selected and selected[0].get("selection_score", 0.0) >= 4.0
            ),
            "best_direction": selected[0].get("stop_name") if selected else None,
            "best_score": selected[0].get("selection_score") if selected else None,
            "best_reason": (
                llm_selection.get("reason")
                or (selected[0].get("reason") if selected else "No candidates generated.")
            ),
            "best_decision_basis": (
                llm_selection.get("decision_basis")
                or (selected[0].get("decision_basis") if selected else None)
            ),
            "selection_source": (
                "llm"
                if llm_selection.get("status") == "completed"
                else "deterministic"
            ),
        }
        self.current_selected_candidates = selected
        self.current_llm_view_decision = llm_selection
        self.write_json(path, metadata)
        self.get_logger().info(
            "generate_candidates complete: "
            + ", ".join(
                f"{c.get('stop_name')}={c.get('selection_score')}"
                for c in selected
            )
        )
        return candidate_generation

    def select_candidates_with_llm(
        self,
        metadata,
        all_ranked,
        deterministic_selected,
        previous_memory,
    ):
        if not self.llm_candidate_selection_enabled:
            return {
                "status": "disabled",
                "model": self.llm_model,
                "reason": "llm_candidate_selection_enabled is false",
            }
        if not all_ranked:
            return {
                "status": "skipped",
                "model": self.llm_model,
                "reason": "No candidates available.",
            }

        prompt = self.build_llm_candidate_prompt(
            metadata=metadata,
            all_ranked=all_ranked,
            deterministic_selected=deterministic_selected,
            previous_memory=previous_memory,
        )
        try:
            decision = self.call_ollama_json(prompt)
            return self.apply_llm_candidate_decision(
                decision=decision,
                all_ranked=all_ranked,
                deterministic_selected=deterministic_selected,
                evidence_text=self.llm_candidate_evidence_text(all_ranked, previous_memory),
            )
        except Exception as exc:
            message = f"LLM candidate selection failed; using deterministic ranking: {exc}"
            self.get_logger().warn(message)
            return {
                "status": "failed",
                "model": self.llm_model,
                "error": str(exc),
                "reason": message,
            }

    def build_llm_candidate_prompt(
        self,
        metadata,
        all_ranked,
        deterministic_selected,
        previous_memory,
    ):
        candidates = []
        for candidate in all_ranked:
            audit = candidate.get("visual_audit", {}) or {}
            candidates.append({
                "stop_name": candidate.get("stop_name"),
                "relative_yaw_deg": candidate.get("relative_yaw_deg"),
                "heading_abs_deg": candidate.get("heading_abs_deg"),
                "deterministic_score": candidate.get("selection_score"),
                "deterministic_basis": candidate.get("decision_basis"),
                "deterministic_reason": short_text(candidate.get("reason"), 180),
                "scene_type": audit.get("scene_type", "unknown"),
                "scene": short_text(audit.get("scene"), 220),
                "visible_objects": short_text(audit.get("visible_objects"), 220),
                "navigation_cues": short_text(audit.get("navigation_cues"), 220),
                "traversable_openings": short_text(audit.get("traversable_openings"), 220),
                "open_traversable_routes": short_text(audit.get("open_traversable_routes"), 220),
                "blocked_or_unclear_routes": short_text(audit.get("blocked_or_unclear_routes"), 220),
                "direct_target_evidence": short_text(audit.get("direct_target_evidence"), 220),
                "useful_context_for_task": short_text(audit.get("useful_context_for_task"), 220),
                "navigation_recommendation": short_text(audit.get("navigation_recommendation"), 220),
                "task_relevance_score": audit.get("task_relevance_score"),
                "task_relevance_level": audit.get("task_relevance_level"),
                "best_direction_cue": short_text(audit.get("best_direction_cue"), 160),
                "best_direction_level": audit.get("best_direction_level"),
                "target_cue": audit.get("target_cue"),
                "target_cue_quote": audit.get("target_cue_quote"),
                "route_affordance": audit.get("route_affordance"),
                "route_quote": audit.get("route_quote"),
                "blocked_quote": audit.get("blocked_quote"),
                "repeated_direction_count": audit.get("repeated_direction_count", 0),
            })

        memory = []
        for item in previous_memory[-self.candidate_memory_nodes:]:
            memory.append({
                "node_name": item.get("node_name"),
                "selected_directions": item.get("selected_directions", []),
                "best_score": item.get("best_score"),
                "best_reason": short_text(item.get("best_reason"), 180),
            })

        payload = {
            "task": self.user_prompt,
            "target_prompt": self.target_prompt,
            "current_node": metadata.get("node_name"),
            "current_pose": metadata.get("planned_pose"),
            "deterministic_top_directions": [
                c.get("stop_name") for c in deterministic_selected
            ],
            "previous_node_memory": memory,
            "candidate_views": candidates,
            "allowed_stop_names": [c.get("stop_name") for c in all_ranked],
        }

        example = {
            "best_stop_name": "one allowed stop_name",
            "ranked_stop_names": ["best", "second", "third"],
            "confidence": 0.0,
            "decision_basis": (
                "direct_visual_evidence/contextual_visual_evidence/"
                "navigation_affordance/commonsense_prior/exploration/uncertain"
            ),
            "move_is_visually_justified": False,
            "visual_evidence_quote": "short exact phrase from candidate_views, or none",
            "affordance_quote": "short exact phrase about path/door/corridor, or none",
            "commonsense_assumption": "general assumption used, or none",
            "reason": "one short grounded sentence",
        }

        rules = """
Choose the best camera view/direction for the robot's next local navigation goal.

Rules:
- Return only valid JSON.
- best_stop_name must be one of allowed_stop_names.
- ranked_stop_names must contain only allowed stop names, best first.
- Use the user's task, but do not invent unseen rooms, signs, objects, or doors.
- Prefer direct visual evidence of the target if present.
- If the target is not visible, prefer contextual places where the target belongs.
- For object targets, a likely room or area is contextual evidence, not direct evidence.
- Prefer views with an open or weak traversable route over closed or unclear routes.
- Never choose a candidate whose route_affordance is "closed" unless direct target evidence is visible in that same candidate.
- A closed door is unavailable, not a promising route. Commonsense assumptions cannot override a closed route.
- Do not infer that a target is behind a door, wall, or unseen room unless the description explicitly says so.
- If the task is probably indoor, do not keep choosing outdoor-looking views unless they show an entrance, doorway, corridor, lobby, or indoor transition.
- If the task is probably outdoor, prefer outdoor exits or outdoor routes.
- Use previous_node_memory to avoid repeating directions that did not help.
- visual_evidence_quote and affordance_quote must be exact short phrases from candidate_views, or "none".
- confidence must be between 0.0 and 1.0.
""".strip()

        return (
            f"{rules}\n\n"
            f"Input:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
            f"Return JSON shaped like:\n{json.dumps(example, indent=2)}"
        )

    def call_ollama_json(self, prompt):
        url = f"{self.ollama_base_url}/api/chat"
        body = json.dumps({
            "model": self.llm_model,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the graph navigation view selector for a mobile robot. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "options": {
                "temperature": 0.0,
                "num_predict": self.llm_candidate_num_predict,
            },
        }).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.llm_request_timeout_sec) as response:
            payload = json.loads(response.read().decode("utf-8"))

        message = payload.get("message", {}) if isinstance(payload, dict) else {}
        content = message.get("content", "")
        return extract_json_object(content)

    def apply_llm_candidate_decision(
        self,
        decision,
        all_ranked,
        deterministic_selected,
        evidence_text,
    ):
        candidate_by_stop = {
            candidate.get("stop_name"): candidate
            for candidate in all_ranked
            if candidate.get("stop_name")
        }
        allowed = set(candidate_by_stop)
        stop_aliases = {
            self.normalized_stop_key(stop_name): stop_name
            for stop_name in allowed
        }

        def canonical_stop(value):
            value = str(value or "").strip()
            if value in allowed:
                return value
            return stop_aliases.get(self.normalized_stop_key(value))

        best_stop = canonical_stop(decision.get("best_stop_name"))
        raw_ranked = decision.get("ranked_stop_names", [])
        if isinstance(raw_ranked, str):
            raw_ranked = [raw_ranked]
        ranked_stops = [
            stop_name
            for stop_name in (canonical_stop(stop) for stop in raw_ranked)
            if stop_name in allowed
        ]
        if best_stop in allowed and best_stop not in ranked_stops:
            ranked_stops.insert(0, best_stop)
        if not ranked_stops:
            raise ValueError(f"LLM did not return an allowed stop: {decision}")

        for candidate in deterministic_selected:
            stop_name = candidate.get("stop_name")
            if stop_name in allowed and stop_name not in ranked_stops:
                ranked_stops.append(stop_name)
        for candidate in all_ranked:
            stop_name = candidate.get("stop_name")
            if stop_name in allowed and stop_name not in ranked_stops:
                ranked_stops.append(stop_name)

        original_ranked_stops = list(ranked_stops)
        override_reason = None
        if ranked_stops and self.llm_choice_needs_route_override(candidate_by_stop.get(ranked_stops[0])):
            unsafe_best = ranked_stops[0]
            safer_ranked = [
                stop_name
                for stop_name in ranked_stops
                if not self.llm_choice_needs_route_override(candidate_by_stop.get(stop_name))
            ]
            if safer_ranked:
                ranked_stops = safer_ranked + [
                    stop_name for stop_name in ranked_stops if stop_name not in safer_ranked
                ]
                override_reason = (
                    f"LLM chose {unsafe_best}, but that view is a closed route without "
                    f"direct target evidence; using {ranked_stops[0]} instead."
                )
                self.get_logger().warn(override_reason)

        selected = []
        for rank, stop_name in enumerate(ranked_stops[:max(1, self.candidate_top_k)], start=1):
            candidate = dict(candidate_by_stop[stop_name])
            candidate["llm_rank"] = rank
            candidate["llm_selected"] = True
            if rank == 1:
                candidate["llm_reason"] = decision.get("reason")
                candidate["llm_decision_basis"] = decision.get("decision_basis")
                candidate["llm_confidence"] = clamp(
                    decision.get("confidence"),
                    0.0,
                    1.0,
                    0.0,
                )
                if override_reason:
                    candidate["llm_override_reason"] = override_reason
            selected.append(candidate)

        visual_quote = self.validated_short_quote(
            decision.get("visual_evidence_quote"),
            evidence_text,
        )
        affordance_quote = self.validated_short_quote(
            decision.get("affordance_quote"),
            evidence_text,
        )
        basis = str(decision.get("decision_basis", "uncertain"))
        confidence = clamp(decision.get("confidence"), 0.0, 1.0, 0.0)
        commonsense_assumption = str(decision.get("commonsense_assumption", "none")).strip()
        if basis in ("direct_visual_evidence", "contextual_visual_evidence") and visual_quote == "none":
            basis = (
                "commonsense_prior"
                if commonsense_assumption and commonsense_assumption.lower() != "none"
                else "exploration"
            )
            confidence = min(confidence, 0.4)
        if basis == "navigation_affordance" and affordance_quote == "none":
            basis = "exploration"
            confidence = min(confidence, 0.3)
        if override_reason:
            confidence = min(confidence, 0.35)

        return {
            "status": "completed",
            "model": self.llm_model,
            "ollama_base_url": self.ollama_base_url,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "best_stop_name": selected[0].get("stop_name"),
            "ranked_stop_names": ranked_stops,
            "original_ranked_stop_names": original_ranked_stops,
            "planner_choice_overridden": bool(override_reason),
            "override_reason": override_reason,
            "confidence": confidence,
            "decision_basis": basis,
            "move_is_visually_justified": bool(decision.get("move_is_visually_justified", False)),
            "visual_evidence_quote": visual_quote,
            "affordance_quote": affordance_quote,
            "commonsense_assumption": commonsense_assumption or "none",
            "reason": short_text(decision.get("reason"), 300),
            "raw_decision": decision,
            "selected_candidates": selected,
        }

    @staticmethod
    def llm_choice_needs_route_override(candidate):
        if not candidate:
            return False
        audit = candidate.get("visual_audit", {}) or {}
        route_affordance = str(audit.get("route_affordance", "")).strip().lower()
        target_cue = str(audit.get("target_cue", "")).strip().lower()
        return route_affordance == "closed" and target_cue != "direct"

    @staticmethod
    def llm_candidate_evidence_text(all_ranked, previous_memory):
        parts = []
        for candidate in all_ranked:
            parts.append(candidate.get("description", ""))
            audit = candidate.get("visual_audit", {}) or {}
            parts.extend(str(value) for value in audit.values())
        for item in previous_memory:
            parts.append(item.get("best_reason", ""))
        return normalize_for_match("\n".join(str(part) for part in parts if part))

    @staticmethod
    def validated_short_quote(quote, evidence_text, max_words=14):
        quote = re.sub(r"\s+", " ", str(quote or "")).strip()
        if not quote or quote.lower() == "none":
            return "none"
        clipped = " ".join(quote.split()[:max_words])
        if normalize_for_match(clipped) in evidence_text:
            return clipped
        return "none"

    @staticmethod
    def normalized_stop_key(stop_name):
        return re.sub(r"[^a-z0-9]+", "_", str(stop_name or "").lower()).strip("_")

    @staticmethod
    def stamp_to_nanoseconds(stamp):
        if stamp is None:
            return None
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def generate_next_node_goal(self, folder):
        path = self.metadata_path(folder)
        metadata = self.read_json(path)
        selected = (metadata.get("candidate_generation") or {}).get("selected_candidates", [])
        if not selected:
            self.current_next_node_goal = None
            self.get_logger().warn("generate_next_node_goal skipped: no selected candidates")
            return None

        node_pose = metadata.get("planned_pose") or self.lookup_robot_pose()
        if node_pose is None:
            self.current_next_node_goal = None
            self.get_logger().warn("generate_next_node_goal skipped: no node pose available")
            return None

        candidate = selected[0]
        relative_yaw_deg = candidate.get("relative_yaw_deg")
        if relative_yaw_deg is None:
            relative_yaw_deg = self.relative_yaw_for_stop(candidate.get("stop_name"))
        if relative_yaw_deg is None:
            self.current_next_node_goal = None
            self.get_logger().warn(
                "generate_next_node_goal skipped: unknown stop direction "
                f"{candidate.get('stop_name')!r}"
            )
            return None

        distance_m = max(0.1, float(self.next_node_goal_distance_m))
        goal_yaw = normalize_rad(float(node_pose["yaw"]) + math.radians(float(relative_yaw_deg)))
        goal_x = float(node_pose["x"]) + distance_m * math.cos(goal_yaw)
        goal_y = float(node_pose["y"]) + distance_m * math.sin(goal_yaw)
        orientation = self.orientation_from_yaw(goal_yaw)

        goal = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame_id": metadata.get("planned_frame_id") or self.global_frame,
            "source_node": metadata.get("node_name", folder.name),
            "source_candidate": {
                "stop_name": candidate.get("stop_name"),
                "relative_yaw_deg": float(relative_yaw_deg),
                "selection_score": candidate.get("selection_score"),
                "reason": candidate.get("reason"),
            },
            "distance_m": distance_m,
            "pose": {
                "x": goal_x,
                "y": goal_y,
                "z": 0.0,
                "yaw": goal_yaw,
                "orientation": orientation,
            },
            "published_topics": {
                "graph_topic": self.next_node_goal_topic,
                "nav2_goal_pose_topic": self.nav2_goal_pose_topic
                if self.publish_nav2_goal_pose else None,
            },
        }

        metadata["next_node_goal"] = goal
        self.write_json(path, metadata)
        self.current_next_node_goal = goal
        self.publish_next_node_goal(goal)
        self.get_logger().info(
            "generate_next_node_goal complete: "
            f"{goal['source_candidate']['stop_name']} -> "
            f"x={goal_x:.2f}, y={goal_y:.2f}, yaw={math.degrees(goal_yaw):.1f} deg"
        )
        return goal

    def publish_next_node_goal(self, goal):
        msg = self.pose_stamped_from_goal(goal)
        self.next_node_goal_pub.publish(msg)
        if self.publish_nav2_goal_pose:
            self.waiting_for_next_node_goal_result = True
            self.next_node_goal_published_at_ns = self.get_clock().now().nanoseconds
            self.next_node_goal_seen_nav2_goal = False
            self.nav2_goal_pose_pub.publish(msg)

    def pose_stamped_from_goal(self, goal):
        msg = PoseStamped()
        msg.header.frame_id = str(goal.get("frame_id") or self.global_frame)
        msg.header.stamp = self.get_clock().now().to_msg()
        pose = goal.get("pose", {})
        msg.pose.position.x = float(pose.get("x", 0.0))
        msg.pose.position.y = float(pose.get("y", 0.0))
        msg.pose.position.z = float(pose.get("z", 0.0))
        orientation = pose.get("orientation", {})
        msg.pose.orientation.x = float(orientation.get("x", 0.0))
        msg.pose.orientation.y = float(orientation.get("y", 0.0))
        msg.pose.orientation.z = float(orientation.get("z", 0.0))
        msg.pose.orientation.w = float(orientation.get("w", 1.0))
        return msg

    def build_direction_candidate(self, snapshot, task_terms, repeated_direction_counts):
        stop_name = str(snapshot.get("stop_name", "unknown"))
        description = self.snapshot_description(snapshot)
        semantic = snapshot.get("semantic_description", {}) or {}
        text = self.snapshot_search_text(snapshot)
        text_l = normalize_for_match(text)
        matched_terms = self.matching_task_terms(text_l, task_terms)
        task_relevance_level = str(semantic.get("task_relevance_level", "none")).lower()
        if not self.non_template_text(semantic.get("task_relevance", "")):
            task_relevance_level = "none"
        best_direction_level = str(semantic.get("best_direction_level", "low")).lower()
        direct_target_visible = bool(semantic.get("direct_target_visible", False))
        task_relevance_score = self.safe_int(semantic.get("task_relevance_score"), 0)
        route_text = self.route_search_text(semantic)
        route_text_l = normalize_for_match(route_text)
        route_affordance = self.classify_route_affordance(route_text_l)
        route_quote = self.first_matching_quote(
            route_text,
            OPEN_ROUTE_PHRASES + WEAK_ROUTE_PHRASES,
        )
        blocked_quote = self.first_matching_quote(
            route_text,
            CLOSED_ROUTE_PHRASES + NO_ROUTE_PHRASES,
        )

        target_phrase = normalize_for_match(self.target_prompt)
        direct_target_text = self.direct_target_search_text(semantic)
        has_direct_target = bool(
            target_phrase and target_phrase in normalize_for_match(direct_target_text)
        )
        has_direct_evidence = direct_target_visible or (
            has_direct_target and self.has_non_template_target_quote(direct_target_text)
        )
        if has_direct_evidence:
            target_cue = "direct"
        elif task_relevance_score >= 3 or task_relevance_level in ("contextual", "weak") or matched_terms:
            target_cue = "contextual"
        else:
            target_cue = "none"
        target_quote = self.first_matching_quote(direct_target_text, [self.target_prompt] + matched_terms)

        score = 0.0
        if has_direct_evidence:
            score += 8.0
        elif task_relevance_score >= 4 or task_relevance_level == "contextual":
            score += 4.0
        elif task_relevance_score >= 1 or task_relevance_level == "weak":
            score += 1.5
        if not has_direct_evidence:
            score += min(6.0, 1.5 * len(matched_terms))

        if best_direction_level == "high":
            score += 3.0
        elif best_direction_level == "medium":
            score += 1.0

        if route_affordance == "open":
            score += 3.0
        elif route_affordance == "weak":
            score += 1.0
        elif route_affordance == "closed":
            score -= 3.0
        elif route_affordance == "blocked_or_unclear":
            score -= 2.0

        repeated_count = repeated_direction_counts.get(stop_name, 0)
        if repeated_count:
            score -= min(2.0, 0.5 * repeated_count)

        if not description:
            score -= 4.0

        rel_yaw = self.relative_yaw_for_stop(stop_name)
        pose = snapshot.get("actual_robot_pose") or {}
        heading_abs_deg = None
        if rel_yaw is not None and pose.get("yaw") is not None:
            heading_abs_deg = normalize_deg(math.degrees(float(pose["yaw"])) + rel_yaw)

        reason = self.candidate_reason(
            stop_name=stop_name,
            target_cue=target_cue,
            matched_terms=matched_terms,
            route_affordance=route_affordance,
            repeated_count=repeated_count,
        )
        visual_audit = {
            "scene": semantic.get("scene", ""),
            "scene_type": semantic.get("scene_type", "unknown"),
            "visible_objects": semantic.get("visible_objects", ""),
            "navigation_cues": semantic.get("navigation_cues", ""),
            "traversable_openings": semantic.get("traversable_openings", ""),
            "open_traversable_routes": semantic.get("open_traversable_routes", ""),
            "blocked_or_unclear_routes": semantic.get("blocked_or_unclear_routes", ""),
            "direct_target_evidence": semantic.get("direct_target_evidence", ""),
            "direct_target_visible": direct_target_visible,
            "useful_context_for_task": semantic.get("useful_context_for_task", ""),
            "navigation_recommendation": semantic.get("navigation_recommendation", ""),
            "task_relevance_score": task_relevance_score,
            "task_relevance": semantic.get("task_relevance", ""),
            "task_relevance_level": task_relevance_level,
            "best_direction_cue": semantic.get("best_direction_cue", ""),
            "best_direction_level": best_direction_level,
            "target_cue": target_cue,
            "target_cue_quote": target_quote,
            "matched_task_terms": matched_terms,
            "route_affordance": route_affordance,
            "route_quote": route_quote,
            "blocked_quote": blocked_quote,
            "repeated_direction_count": repeated_count,
        }
        return {
            "label": snapshot.get("label"),
            "stop_index": snapshot.get("stop_index"),
            "stop_name": stop_name,
            "direction": stop_name,
            "relative_yaw_deg": rel_yaw,
            "heading_abs_deg": round(heading_abs_deg, 1) if heading_abs_deg is not None else None,
            "image": self.color_image_from_snapshot(snapshot),
            "description": description,
            "selection_score": round(score, 2),
            "decision_basis": self.candidate_decision_basis(target_cue, route_affordance),
            "reason": reason,
            "visual_audit": visual_audit,
        }

    @staticmethod
    def safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def has_non_template_target_quote(text):
        text_l = normalize_for_match(text)
        template_fragments = (
            "direct/contextual/weak/none",
            "requested target",
            "direct target evidence",
            "target is directly visible",
            "target phrase appears",
        )
        return not any(fragment in text_l for fragment in template_fragments)

    @staticmethod
    def route_search_text(semantic):
        open_routes = GraphNavStateMachine.non_template_text(
            semantic.get("open_traversable_routes")
            or semantic.get("traversable_openings")
            or ""
        )
        blocked_routes = GraphNavStateMachine.non_template_text(
            semantic.get("blocked_or_unclear_routes") or ""
        )
        navigation = GraphNavStateMachine.non_template_text(
            semantic.get("navigation_cues") or ""
        )

        if open_routes or blocked_routes:
            return "\n".join([open_routes, blocked_routes])
        return navigation

    @staticmethod
    def direct_target_search_text(semantic):
        return "\n".join([
            GraphNavStateMachine.non_template_text(semantic.get("direct_target_evidence", "")),
            GraphNavStateMachine.non_template_text(semantic.get("visible_objects", "")),
            GraphNavStateMachine.non_template_text(semantic.get("distinctive_landmarks", "")),
            GraphNavStateMachine.non_template_text(semantic.get("scene", "")),
        ])

    @staticmethod
    def non_template_text(value):
        text = str(value or "")
        text_l = normalize_for_match(text)
        template_fragments = (
            "one short sentence",
            "comma-separated",
            "visible entrance, exit, doorway",
            "courtyard, or facade",
            "closed doors, walls, obstacles",
            "unclear paths, or none",
            "physically open visible routes",
            "direct/contextual/weak/none",
            "short reason using only visible evidence",
            "visible open doorway/corridor/hallway",
            "visible floor/path/door/corridor",
            "or none; say closed door",
            "no strong direction cue",
        )
        if any(fragment in text_l for fragment in template_fragments):
            return ""
        none_fragments = (
            "no visible",
            "no open",
            "no clear",
            "none visible",
            "not visible",
            "no blocked",
            "no unclear",
        )
        if text_l.strip(" .") in {"none", "unknown", "n/a", "not visible"}:
            return ""
        if any(fragment in text_l for fragment in none_fragments):
            return ""
        return text

    def extract_task_terms(self):
        terms = []
        raw_parts = [self.target_prompt, self.user_prompt]
        for raw in raw_parts:
            raw_l = normalize_for_match(raw)
            if raw_l and raw_l not in TASK_STOPWORDS:
                terms.append(raw_l)
            for token in re.findall(r"[a-z0-9_]+", raw_l):
                if len(token) < 3 or token in TASK_STOPWORDS:
                    continue
                terms.append(token)

        deduped = []
        seen = set()
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            deduped.append(term)
        return deduped[:16]

    @staticmethod
    def matching_task_terms(text_l, task_terms):
        matches = []
        for term in task_terms:
            term_l = normalize_for_match(term)
            if term_l and term_l in text_l:
                matches.append(term)
        return matches

    @staticmethod
    def classify_route_affordance(text_l):
        has_open = positive_phrase_in_text(OPEN_ROUTE_PHRASES, text_l)
        has_weak = positive_phrase_in_text(WEAK_ROUTE_PHRASES, text_l)
        has_closed = any(phrase in text_l for phrase in CLOSED_ROUTE_PHRASES)
        has_no_route = any(phrase in text_l for phrase in NO_ROUTE_PHRASES)

        if has_open:
            return "open"
        if has_closed:
            return "closed"
        if has_no_route:
            return "blocked_or_unclear"
        if has_weak:
            return "weak"
        return "unknown"

    @staticmethod
    def first_matching_quote(text, phrases, max_words=12):
        text = str(text)
        text_l = text.lower()
        for phrase in phrases:
            phrase_l = normalize_for_match(phrase)
            if not phrase_l:
                continue
            idx = text_l.find(phrase_l)
            if idx < 0:
                continue
            words = text[idx:idx + len(phrase_l)].strip().split()
            return " ".join(words[:max_words])
        return "none"

    @staticmethod
    def candidate_decision_basis(target_cue, route_affordance):
        if target_cue == "direct":
            return "direct_visual_evidence"
        if target_cue == "contextual":
            return "contextual_visual_evidence"
        if route_affordance in ("open", "weak"):
            return "navigation_affordance"
        if route_affordance in ("closed", "blocked_or_unclear"):
            return "insufficient_visual_evidence"
        return "exploration"

    @staticmethod
    def candidate_reason(stop_name, target_cue, matched_terms, route_affordance, repeated_count):
        parts = []
        if target_cue == "direct":
            parts.append("target phrase appears in this view")
        elif matched_terms:
            parts.append("task terms match this view")
        if route_affordance in ("open", "weak"):
            parts.append(f"{route_affordance} route cue is visible")
        elif route_affordance in ("closed", "blocked_or_unclear"):
            parts.append("route is blocked or unclear")
        if repeated_count:
            parts.append("direction was selected before")
        if not parts:
            parts.append("no strong task cue, kept as exploration option")
        return f"{stop_name}: " + "; ".join(parts) + "."

    @staticmethod
    def snapshot_description(snapshot):
        desc = str(snapshot.get("description", "")).strip()
        if desc:
            return desc

        sem = snapshot.get("semantic_description", {}) or {}
        if sem:
            return (
                f"Scene: {sem.get('scene', '')}\n"
                f"Visible objects: {sem.get('visible_objects', '')}\n"
                f"Distinctive landmarks: {sem.get('distinctive_landmarks', '')}\n"
                f"Navigation cues: {sem.get('navigation_cues', '')}\n"
                f"Traversable openings: {sem.get('traversable_openings', '')}\n"
                f"Scene type: {sem.get('scene_type', '')}"
            ).strip()

        return ""

    def snapshot_search_text(self, snapshot):
        sem = snapshot.get("semantic_description", {}) or {}
        return "\n".join([
            self.snapshot_description(snapshot),
            str(sem.get("scene", "")),
            str(sem.get("visible_objects", "")),
            str(sem.get("distinctive_landmarks", "")),
            str(sem.get("navigation_cues", "")),
            str(sem.get("traversable_openings", "")),
            str(sem.get("scene_type", "")),
        ])

    @staticmethod
    def color_image_from_snapshot(snapshot):
        image_stamps = snapshot.get("image_stamps", {})
        color_stamp = image_stamps.get("realsense_color", {})
        artifacts = color_stamp.get("artifacts", {})
        if artifacts.get("png"):
            return artifacts["png"]
        for filename in snapshot.get("filenames", []):
            lower_name = str(filename).lower()
            if lower_name.startswith("c") and lower_name.endswith((".png", ".jpg", ".jpeg")):
                return filename
        return None

    @staticmethod
    def relative_yaw_for_stop(stop_name):
        return STOP_RELATIVE_YAW_DEG.get(str(stop_name).strip().lower())

    @staticmethod
    def orientation_from_yaw(yaw):
        half_yaw = 0.5 * float(yaw)
        return {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(half_yaw),
            "w": math.cos(half_yaw),
        }

    def load_previous_node_memory(self, current_node_name):
        memories = []
        current_id = self.node_number_from_name(current_node_name)
        for folder in sorted(self.nodes_dir.iterdir(), key=lambda p: p.name):
            if not folder.is_dir() or folder.name == current_node_name:
                continue
            node_id = self.node_number_from_name(folder.name)
            if current_id is not None and node_id is not None and node_id >= current_id:
                continue
            metadata_path = self.metadata_path(folder)
            if not metadata_path.exists():
                continue
            try:
                metadata = self.read_json(metadata_path)
            except Exception:
                continue

            candidate_generation = metadata.get("candidate_generation", {})
            selected = candidate_generation.get("selected_candidates", [])
            memories.append({
                "node_name": metadata.get("node_name", folder.name),
                "status": metadata.get("status"),
                "selected_directions": [
                    c.get("stop_name") for c in selected if c.get("stop_name")
                ],
                "best_score": selected[0].get("selection_score") if selected else None,
                "best_reason": selected[0].get("reason") if selected else None,
            })

        if self.candidate_memory_nodes <= 0:
            return memories
        return memories[-self.candidate_memory_nodes:]

    @staticmethod
    def previous_selected_direction_counts(previous_memory):
        counts = {}
        for memory in previous_memory:
            for direction in memory.get("selected_directions", []):
                counts[direction] = counts.get(direction, 0) + 1
        return counts

    @staticmethod
    def node_number_from_name(name):
        match = re.match(r"^n(\d+)$", str(name or ""))
        return int(match.group(1)) if match else None

    def record_description_status(self, folder, status, error=None):
        path = self.metadata_path(folder)
        if not path.exists():
            return
        metadata = self.read_json(path)
        metadata["descriptions"] = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "server_url": self.moondream_server_url,
            "error": error,
        }
        self.write_json(path, metadata)

    def should_stop_scan_for_target(self):
        return (
            self.stop_scan_on_target_visible
            and self.target_visible
            and self.phase == Phase.GOAL_NAVIGATION
        )

    def record_scan_interrupted_by_target(self, folder, stop_index=None, stop_name=None):
        path = self.metadata_path(folder)
        if not path.exists():
            return
        metadata = self.read_json(path)
        metadata["scan_interrupted"] = {
            "reason": "target_visible",
            "interrupted_at": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase.value,
            "stop_index": stop_index,
            "stop_name": stop_name,
            "target_visible_source": self.target_visible_source,
            "target_visible_observation_count": self.target_visible_observation_count,
            "target_visible_last_point": self.target_visible_last_point,
            "return_home": "skipped_to_keep_camera_on_target",
        }
        self.write_json(path, metadata)

    def scan_360(self, folder):
        previous_tokens = None
        preposition_responses = self.move_through_stops([2, 1], reason=f"{folder.name}: pre-positioning")
        metadata = self.read_json(self.metadata_path(folder))
        metadata["preposition_to_start"] = {
            "created_at": datetime.now().isoformat(),
            "from_default_stop": 5,
            "to_first_capture_stop": 1,
            "path": [2, 1],
            "responses": preposition_responses,
            "settle_before_sweep_sec": self.settle_before_sweep_sec,
        }
        self.write_json(self.metadata_path(folder), metadata)

        if self.settle_before_sweep_sec > 0.0:
            self.get_logger().info(
                f"{folder.name}: waiting {self.settle_before_sweep_sec:.2f}s "
                "at back before starting image capture"
            )
            self.sleep_seconds(self.settle_before_sweep_sec)

        if self.should_stop_scan_for_target():
            self.record_scan_interrupted_by_target(folder)
            return

        for i, stop_name in enumerate(self.camera_stop_names, start=1):
            if self.should_stop_scan_for_target():
                self.record_scan_interrupted_by_target(folder, i, stop_name)
                return

            self.get_logger().info(f"{folder.name}: moving camera to stop {i} - {stop_name}")
            move_started_at = self.get_clock().now()
            move_response = self.move_lerobot_stop(i, stop_name)
            self.sleep_seconds(self.settle_after_motion_sec)

            if self.should_stop_scan_for_target():
                self.record_scan_interrupted_by_target(folder, i, stop_name)
                return

            if not self.wait_until_realsense_available(timeout_sec=self.wait_for_fresh_frame_sec):
                raise RuntimeError(f"Timed out waiting for RealSense frames at stop {i} - {stop_name}")

            if self.wait_for_fresh_color_frame_sec > 0.0:
                fresh = self.wait_for_fresh_color_frame_after(
                    after_time=move_started_at,
                    timeout_sec=self.wait_for_fresh_color_frame_sec,
                    frames_to_skip=self.fresh_frames_to_skip,
                )
                if not fresh:
                    self.get_logger().warn(
                        f"Timed out waiting for a fresh post-move color frame at "
                        f"stop {i} - {stop_name}; saving latest cached frame."
                    )

            if self.should_stop_scan_for_target():
                self.record_scan_interrupted_by_target(folder, i, stop_name)
                return

            current_tokens = self.current_frame_tokens()
            if previous_tokens is not None and current_tokens == previous_tokens:
                self.get_logger().warn("RealSense token did not change; saving latest cached frames anyway.")
            previous_tokens = current_tokens

            filenames, stamps = self.save_realsense_stop_snapshot(folder, i, stop_name)
            self.append_snapshot_metadata(
                folder=folder,
                label=f"camera_stop_{i}_{self.safe_name(stop_name)}",
                stop_index=i,
                stop_name=stop_name,
                move_response=move_response,
                filenames=filenames,
                image_stamps=stamps,
                frame_tokens=current_tokens,
            )

        metadata = self.read_json(self.metadata_path(folder))
        try:
            responses = self.move_through_stops([5], reason=f"{folder.name}: returning to default/front")
            metadata["return_home"] = {
                "returned_at": datetime.now().isoformat(),
                "ok": True,
                "default_stop_index": 5,
                "default_stop_name": "front",
                "path": [5],
                "responses": responses,
            }
        except Exception as exc:
            metadata["return_home"] = {
                "returned_at": datetime.now().isoformat(),
                "ok": False,
                "default_stop_index": 5,
                "default_stop_name": "front",
                "path": [5],
                "error": str(exc),
            }
        self.write_json(self.metadata_path(folder), metadata)

    def msg_stamp_tuple(self, msg):
        return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))

    def msg_stamp_ns(self, msg):
        sec, nanosec = self.msg_stamp_tuple(msg)
        return sec * 1_000_000_000 + nanosec

    def current_frame_tokens(self):
        with self.latest_lock:
            tokens = {}
            for name, msg in self.latest_images.items():
                recv_time = self.latest_receive_time.get(name)
                tokens[name] = {
                    "recv_ns": recv_time.nanoseconds if recv_time is not None else None,
                    "stamp": self.msg_stamp_tuple(msg),
                }
            return tokens

    def wait_until_realsense_available(self, timeout_sec=10.0):
        start = self.get_clock().now()
        required = {"realsense_color", "realsense_aligned_depth"}
        while rclpy.ok():
            with self.latest_lock:
                available = set(self.latest_images.keys())
            if required.issubset(available):
                return True
            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn(f"Available image sources: {sorted(available)}")
                return False
            self.sleep_seconds(0.05)

    def wait_for_fresh_color_frame_after(self, after_time, timeout_sec, frames_to_skip=0):
        start = self.get_clock().now()
        after_ns = after_time.nanoseconds
        required_fresh_frames = max(1, int(frames_to_skip) + 1)
        seen_stamps = set()
        fresh_count = 0

        while rclpy.ok():
            with self.latest_lock:
                msg = self.latest_images.get("realsense_color")
                recv_time = self.latest_receive_time.get("realsense_color")

            if msg is not None and recv_time is not None and recv_time.nanoseconds >= after_ns:
                stamp_ns = self.msg_stamp_ns(msg)
                stamp = self.msg_stamp_tuple(msg)
                if stamp not in seen_stamps and stamp_ns + 100_000_000 >= after_ns:
                    seen_stamps.add(stamp)
                    fresh_count += 1
                    if fresh_count >= required_fresh_frames:
                        return True

            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                return False
            self.sleep_seconds(0.02)

    @staticmethod
    def safe_name(name):
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def save_realsense_stop_snapshot(self, folder, stop_index, stop_name):
        safe_stop = self.safe_name(stop_name)
        with self.latest_lock:
            images_copy = dict(self.latest_images)
            depth_camera_info = self.latest_depth_camera_info

        filenames = []
        stamps = {}
        for source in self.image_sources:
            name = source["name"]
            if name not in images_copy:
                raise RuntimeError(f"Missing required image source: {name}")
            msg = images_copy[name]
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if name == "realsense_aligned_depth":
                artifacts = self.save_depth_artifacts(
                    folder=folder,
                    stop_index=stop_index,
                    stop_name=safe_stop,
                    depth_image=cv_img,
                    encoding=msg.encoding,
                )
                filenames.extend(artifacts.values())
            else:
                path = Path(folder) / f"c{stop_index}_{safe_stop}.png"
                if not cv2.imwrite(str(path), self.prepare_image_for_saving(cv_img, msg.encoding)):
                    raise RuntimeError(f"cv2.imwrite failed for {path}")
                artifacts = {"png": path.name}
                filenames.append(path.name)

            stamps[name] = {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
                "encoding": msg.encoding,
                "height": int(msg.height),
                "width": int(msg.width),
                "artifacts": artifacts,
            }
            if name == "realsense_aligned_depth":
                stamps[name]["depth_scale_to_meters"] = self.depth_scale_to_meters(msg.encoding)
                stamps[name]["camera_info"] = self.camera_info_to_dict(depth_camera_info)
        return filenames, stamps

    def save_depth_artifacts(self, folder, stop_index, stop_name, depth_image, encoding):
        folder = Path(folder)
        artifacts = {}

        if "png" in self.depth_save_formats:
            path = folder / f"d{stop_index}_{stop_name}.png"
            if not cv2.imwrite(str(path), self.prepare_image_for_saving(depth_image, encoding)):
                raise RuntimeError(f"cv2.imwrite failed for {path}")
            artifacts["png"] = path.name

        if "npz" in self.depth_save_formats:
            path = folder / f"dr{stop_index}_{stop_name}.npz"
            np.savez_compressed(
                path,
                depth=depth_image,
                encoding=np.array(str(encoding)),
                depth_scale_to_meters=np.array(self.depth_scale_to_meters(encoding), dtype=np.float32),
            )
            artifacts["npz"] = path.name

        if "preview_png" in self.depth_save_formats:
            path = folder / f"dv{stop_index}_{stop_name}.png"
            preview = self.depth_preview_image(depth_image, encoding)
            if not cv2.imwrite(str(path), preview):
                raise RuntimeError(f"cv2.imwrite failed for {path}")
            artifacts["preview_png"] = path.name

        return artifacts

    def depth_preview_image(self, depth_image, encoding):
        depth_m = self.depth_to_meters(depth_image, encoding)
        valid = np.isfinite(depth_m) & (depth_m > 0.0)

        preview = np.zeros(depth_m.shape, dtype=np.uint8)
        if np.any(valid):
            max_depth = max(0.1, self.max_depth_preview_m)
            scaled = np.clip(depth_m / max_depth, 0.0, 1.0)
            preview[valid] = (255.0 * (1.0 - scaled[valid])).astype(np.uint8)

        return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)

    def depth_to_meters(self, depth_image, encoding):
        scale = self.depth_scale_to_meters(encoding)
        depth = depth_image.astype(np.float32) * scale
        depth[depth <= 0.0] = np.nan
        return depth

    @staticmethod
    def depth_scale_to_meters(encoding):
        encoding = str(encoding).lower()
        if encoding in ("16uc1", "mono16"):
            return 0.001
        return 1.0

    @staticmethod
    def camera_info_to_dict(msg):
        if msg is None:
            return None

        return {
            "frame_id": msg.header.frame_id,
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "height": int(msg.height),
            "width": int(msg.width),
            "distortion_model": msg.distortion_model,
            "d": [float(v) for v in msg.d],
            "k": [float(v) for v in msg.k],
            "r": [float(v) for v in msg.r],
            "p": [float(v) for v in msg.p],
        }

    @staticmethod
    def prepare_image_for_saving(cv_img, encoding):
        encoding = str(encoding).lower()
        if encoding == "rgb8":
            return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
        return cv_img

    def check_lerobot_server(self):
        url = f"{self.lerobot_server_url}/health"
        try:
            with request.urlopen(url, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Could not reach LeRobot server at {url}: {exc}")
        if not payload.get("ok", False):
            raise RuntimeError(f"LeRobot server unhealthy: {payload}")

    def move_lerobot_stop(self, stop_index, stop_name):
        url = f"{self.lerobot_server_url}/move_stop"
        body = json.dumps({"stop_index": int(stop_index), "stop_name": str(stop_name)}).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=30.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"LeRobot HTTP error: {exc.code} {exc.reason}")
        except URLError as exc:
            raise RuntimeError(f"LeRobot URL error: {exc}")
        if not payload.get("ok", False):
            raise RuntimeError(f"LeRobot move failed: {payload}")
        return payload

    def move_through_stops(self, stop_indices, reason="moving"):
        responses = []
        for stop_index in stop_indices:
            stop_name = self.camera_stop_names[int(stop_index) - 1]
            self.get_logger().info(f"{reason}: stop {stop_index} - {stop_name}")
            responses.append(self.move_lerobot_stop(stop_index, stop_name))
            self.sleep_seconds(0.15)
        return responses

    def next_node_id(self):
        highest = 0
        patterns = (re.compile(r"^n(\d+)$"), re.compile(r"^point_(\d+)$"))
        for path in self.nodes_dir.iterdir():
            if not path.is_dir():
                continue
            for pattern in patterns:
                match = pattern.match(path.name)
                if match:
                    highest = max(highest, int(match.group(1)))
                    break
        return highest + 1

    @staticmethod
    def node_name(node_id):
        return f"n{node_id:02d}"

    def make_node_folder(self, node_id):
        while True:
            folder = self.nodes_dir / self.node_name(node_id)
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=False)
                return folder
            node_id += 1

    def node_id_from_folder(self, folder):
        match = re.match(r"^n(\d+)$", folder.name)
        return int(match.group(1)) if match else self.next_node_id()

    @staticmethod
    def metadata_path(folder):
        return Path(folder) / "metadata.json"

    def write_initial_metadata(self, folder, node_id):
        pose = self.lookup_robot_pose()
        metadata = {
            "node_id": node_id,
            "node_name": self.node_name(node_id),
            "created_at": datetime.now().isoformat(),
            "planned_frame_id": pose["frame_id"] if pose is not None else self.global_frame,
            "planned_pose": {
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "yaw": pose["yaw"],
                "orientation": pose["orientation"],
            } if pose is not None else None,
            "capture_mode": "graph_nav_lerobot_realsense_sweep",
            "lerobot_server_url": self.lerobot_server_url,
            "camera_stop_names": self.camera_stop_names,
            "realsense_color_topic": str(self.get_parameter("realsense_color_topic").value),
            "realsense_aligned_depth_topic": str(self.get_parameter("realsense_aligned_depth_topic").value),
            "realsense_depth_camera_info_topic": str(
                self.get_parameter("realsense_depth_camera_info_topic").value
            ),
            "depth_save_formats": sorted(self.depth_save_formats),
            "depth_scale_to_meters_by_encoding": {
                "16UC1": 0.001,
                "mono16": 0.001,
                "32FC1": 1.0,
            },
            "max_depth_preview_m": self.max_depth_preview_m,
            "status": "created",
            "snapshots": [],
        }
        self.write_json(self.metadata_path(folder), metadata)

    def append_snapshot_metadata(self, folder, label, stop_index, stop_name, move_response, filenames, image_stamps, frame_tokens=None):
        path = self.metadata_path(folder)
        metadata = self.read_json(path)
        metadata["snapshots"].append({
            "label": label,
            "captured_at": datetime.now().isoformat(),
            "filenames": filenames,
            "actual_robot_pose": self.lookup_robot_pose(),
            "stop_index": stop_index,
            "stop_name": stop_name,
            "lerobot_raw_targets": move_response.get("raw_targets"),
            "lerobot_actual_positions": move_response.get("actual_positions"),
            "lerobot_errors": move_response.get("errors"),
            "lerobot_move_response": move_response,
            "image_stamps": image_stamps,
            "frame_tokens": frame_tokens,
        })
        self.write_json(path, metadata)

    def mark_metadata_status(self, folder, status, error=None):
        path = self.metadata_path(folder)
        if not path.exists():
            return
        metadata = self.read_json(path)
        metadata["status"] = status
        metadata["updated_at"] = datetime.now().isoformat()
        if status in ("completed", "failed"):
            metadata["finished_at"] = metadata["updated_at"]
        if error is not None:
            metadata["error"] = error
        self.write_json(path, metadata)

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
                "sec": int(transform.header.stamp.sec),
                "nanosec": int(transform.header.stamp.nanosec),
            },
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
            "yaw": float(yaw_from_quaternion(q)),
            "orientation": {
                "x": float(q.x),
                "y": float(q.y),
                "z": float(q.z),
                "w": float(q.w),
            },
        }

    def goal_navigation_tick(self):
        if self.goal_reached:
            self.transition_to(Phase.FINISHED)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def publish_flags(self):
        phase_msg = String()
        phase_msg.data = self.phase.value
        self.phase_pub.publish(phase_msg)
        finished_msg = Bool()
        finished_msg.data = self.finished
        self.finished_pub.publish(finished_msg)

    def write_state(self, initialization_complete):
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase.value,
            "initialization_complete": bool(initialization_complete),
            "target_visible": self.target_visible,
            "target_visible_source": self.target_visible_source,
            "target_visible_observation_count": self.target_visible_observation_count,
            "target_visible_last_point": self.target_visible_last_point,
            "goal_reached": self.goal_reached,
            "finished": self.finished,
            "scan_running": self.scan_running,
            "pending_scan": self.pending_scan,
            "waiting_for_next_node_goal_result": self.waiting_for_next_node_goal_result,
            "current_node_folder": str(self.current_node_folder) if self.current_node_folder else None,
            "current_selected_candidates": self.current_selected_candidates,
            "current_llm_view_decision": self.current_llm_view_decision,
            "current_next_node_goal": self.current_next_node_goal,
            "user_prompt": self.user_prompt,
            "target_prompt": self.target_prompt,
        }
        self.state_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def read_json(path):
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def write_json(path, data):
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def sleep_seconds(seconds):
        threading.Event().wait(max(0.0, seconds))


def main(args=None):
    rclpy.init(args=args)
    node = GraphNavStateMachine()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        try:
            if rclpy.ok():
                node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
