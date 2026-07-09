#!/usr/bin/env python3

import json
import math
import re
import sys
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from camera_capture.camera_capture import SnapshotCaptureNode


class GuiSignals(QObject):
    status_update = pyqtSignal(str)


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    pose = PoseStamped()
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)
    return pose.pose.orientation


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ManualPoseSequenceLogger(SnapshotCaptureNode):
    """
    GUI-triggered dataset logger.

    Workflow:
      1. User manually drives the robot to a location while SLAM is running.
      2. User presses the GUI button.
      3. The node creates a new nXX folder.
      4. It captures the four GMSL cameras at yaw offsets [0, 30, 60] degrees.
      5. It stores 12 PNG files plus metadata.json with TF pose for every snapshot.

    If rotate_with_nav2=true, the node uses Nav2 NavigateToPose to rotate in place
    to current_yaw + offset at the current x/y position.
    If rotate_with_nav2=false, it captures the three snapshots without commanding motion.
    """

    def __init__(self, signals: GuiSignals):
        super().__init__(signals, node_name="manual_pose_sequence_logger")

        self.declare_parameter("nodes_dir", "/rgbd_camera_intel_dev/src/nodes")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("tf_timeout_sec", 1.0)
        self.declare_parameter("snapshot_yaw_offsets_degrees", [0.0, 30.0, 60.0])
        self.declare_parameter("rotate_with_nav2", True)
        self.declare_parameter("nav2_action_name", "navigate_to_pose")
        self.declare_parameter("settle_before_capture_sec", 1.0)
        self.declare_parameter("wait_for_fresh_frame_sec", 0.25)
        self.declare_parameter("stop_cmd_vel_topic", "/panther/cmd_vel")
        self.declare_parameter("stop_command_duration_sec", 0.25)

        self.nodes_dir = Path(self.get_parameter("nodes_dir").value)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.snapshot_yaw_offsets_degrees = [
            float(v) for v in self.get_parameter("snapshot_yaw_offsets_degrees").value
        ]
        self.rotate_with_nav2 = bool(self.get_parameter("rotate_with_nav2").value)
        self.nav2_action_name = str(self.get_parameter("nav2_action_name").value)
        self.settle_before_capture_sec = float(self.get_parameter("settle_before_capture_sec").value)
        self.wait_for_fresh_frame_sec = float(self.get_parameter("wait_for_fresh_frame_sec").value)
        self.stop_cmd_vel_topic = str(self.get_parameter("stop_cmd_vel_topic").value)
        self.stop_command_duration_sec = float(self.get_parameter("stop_command_duration_sec").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav2_action_name)
        self.stop_cmd_pub = self.create_publisher(TwistStamped, self.stop_cmd_vel_topic, 10)

        self.sequence_running = False
        self.sequence_lock = threading.Lock()

        # Keep only the four GMSL image sources inherited from SnapshotCaptureNode.
        self.image_sources = [
            source for source in self.image_sources if not source["name"].startswith("realsense_")
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
        self.get_logger().info("Manual pose sequence logger subscribing to GMSL topics:")
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

        self.signals.status_update.emit(
            "Ready.\nDrive the robot manually to a location, then press Capture Sequence.\n\n"
            f"Output folder:\n{self.nodes_dir}\n\n"
            f"Yaw offsets: {self.snapshot_yaw_offsets_degrees}\n"
            f"rotate_with_nav2={self.rotate_with_nav2}"
        )

    def request_capture_sequence(self):
        with self.sequence_lock:
            if self.sequence_running:
                self.get_logger().warn("Capture sequence is already running.")
                self.signals.status_update.emit("Capture sequence already running.")
                return
            self.sequence_running = True
        threading.Thread(target=self.run_capture_sequence, daemon=True).start()

    def set_save_dir(self, save_dir: str):
        self.nodes_dir = Path(save_dir)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = self.nodes_dir
        self.get_logger().info(f"nodes_dir changed to: {self.nodes_dir}")
        self.signals.status_update.emit(f"nodes_dir changed to:\n{self.nodes_dir}")

    def run_capture_sequence(self):
        folder = None
        try:
            start_pose = self.lookup_robot_pose_stamped()
            if start_pose is None:
                raise RuntimeError(
                    f"Could not get TF pose {self.global_frame} -> {self.robot_frame}. "
                    "Is SLAM/localization running?"
                )

            start_yaw = yaw_from_quaternion(start_pose.pose.orientation)
            node_id = self.next_node_id()
            folder = self.make_node_folder(node_id)
            node_id = self.node_id_from_folder(folder)
            node_name = self.node_name(node_id)

            self.write_initial_metadata(folder, node_id, start_pose, start_yaw)
            self.get_logger().info(f"Starting manual capture sequence for {node_name} in {folder}")
            self.signals.status_update.emit(f"Started sequence for {node_name}\nFolder:\n{folder}")

            if self.rotate_with_nav2:
                self.get_logger().info("Waiting for Nav2 NavigateToPose action server...")
                if not self.nav_client.wait_for_server(timeout_sec=10.0):
                    raise RuntimeError(f"Nav2 action server '{self.nav2_action_name}' not available.")

            for snapshot_index, yaw_offset_deg in enumerate(self.snapshot_yaw_offsets_degrees, start=1):
                target_yaw = normalize_angle(start_yaw + math.radians(yaw_offset_deg))

                if self.rotate_with_nav2:
                    target_pose = self.pose_with_yaw_at_same_position(start_pose, target_yaw)
                    self.get_logger().info(f"Rotating/correcting to yaw offset +{yaw_offset_deg:.1f} deg")
                    if not self.navigate_to_pose(target_pose):
                        raise RuntimeError(f"Nav2 rotation to +{yaw_offset_deg:.1f} deg failed.")
                else:
                    self.get_logger().info(
                        f"rotate_with_nav2=false; capturing offset label +{yaw_offset_deg:.1f} deg without commanding rotation."
                    )

                self.publish_stop_command()
                self.sleep_seconds(self.settle_before_capture_sec)
                self.sleep_seconds(self.wait_for_fresh_frame_sec)

                self.get_logger().info(f"Capturing snapshot_{snapshot_index} at +{yaw_offset_deg:.1f} deg")
                filenames = self.save_snapshot_frames(folder, snapshot_index=snapshot_index - 1)
                self.append_snapshot_metadata(
                    folder=folder,
                    label=f"snapshot_{snapshot_index}",
                    filenames=filenames,
                    yaw_offset_degrees=yaw_offset_deg,
                    target_yaw=target_yaw,
                )
                self.signals.status_update.emit(
                    f"{folder.name}: captured snapshot {snapshot_index}/{len(self.snapshot_yaw_offsets_degrees)}"
                )

            self.mark_metadata_status(folder, "completed")
            self.signals.status_update.emit(f"Sequence complete.\nFolder:\n{folder}")
            self.get_logger().info(f"Manual capture sequence complete: {folder}")

        except Exception as exc:
            self.get_logger().error(f"Manual capture sequence failed: {exc}")
            if folder is not None:
                self.mark_metadata_status(folder, "failed", error=str(exc))
            self.signals.status_update.emit(f"Sequence failed:\n{exc}")
        finally:
            with self.sequence_lock:
                self.sequence_running = False

    def save_snapshot_frames(self, folder, snapshot_index: int):
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
            message = (
                "Missing images for sources: "
                + ", ".join(missing_sources)
                + ". Available sources: "
                + ", ".join(sorted(images_copy.keys()))
            )
            if self.require_all_topics:
                raise RuntimeError(message)
            self.get_logger().warn(message)

        return filenames

    def lookup_robot_pose_stamped(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout_sec),
            )
        except Exception as exc:
            self.get_logger().warn(f"Could not look up robot TF pose: {exc}")
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = transform.header.stamp
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        return pose

    def pose_to_dict(self, pose):
        q = pose.pose.orientation
        p = pose.pose.position
        return {
            "frame_id": pose.header.frame_id,
            "stamp": {"sec": pose.header.stamp.sec, "nanosec": pose.header.stamp.nanosec},
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "yaw": yaw_from_quaternion(q),
            "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w},
        }

    def pose_with_yaw_at_same_position(self, pose, yaw):
        target = PoseStamped()
        target.header.frame_id = self.global_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position = deepcopy(pose.pose.position)
        target.pose.orientation = quaternion_from_yaw(yaw)
        return target

    def navigate_to_pose(self, pose):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal_handle = self.wait_for_future(self.nav_client.send_goal_async(goal))
        if goal_handle is None or not goal_handle.accepted:
            return False
        result = self.wait_for_future(goal_handle.get_result_async())
        return result is not None and int(result.status) == 4

    def publish_stop_command(self):
        if self.stop_command_duration_sec <= 0.0:
            return
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

    def wait_for_future(self, future):
        event = threading.Event()
        future.add_done_callback(lambda _future: event.set())
        event.wait()
        return future.result()

    def sleep_seconds(self, seconds):
        threading.Event().wait(max(0.0, seconds))

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

    def node_name(self, node_id):
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
        if match:
            return int(match.group(1))
        match = re.match(r"^point_(\d+)$", folder.name)
        if match:
            return int(match.group(1))
        return self.next_node_id()

    def metadata_path(self, folder):
        return Path(folder) / "metadata.json"

    def write_initial_metadata(self, folder, node_id, start_pose, start_yaw):
        metadata = {
            "node_id": node_id,
            "node_name": self.node_name(node_id),
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "capture_mode": "manual_pose_sequence",
            "global_frame": self.global_frame,
            "robot_frame": self.robot_frame,
            "rotate_with_nav2": self.rotate_with_nav2,
            "snapshot_yaw_offsets_degrees": self.snapshot_yaw_offsets_degrees,
            "start_robot_pose": self.pose_to_dict(start_pose),
            "start_yaw": start_yaw,
            "snapshots": [],
        }
        self.write_json(self.metadata_path(folder), metadata)

    def append_snapshot_metadata(self, folder, label, filenames, yaw_offset_degrees, target_yaw):
        path = self.metadata_path(folder)
        metadata = self.read_json(path)
        actual_pose = self.lookup_robot_pose_stamped()
        metadata["snapshots"].append({
            "label": label,
            "captured_at": datetime.now().isoformat(),
            "yaw_offset_degrees": yaw_offset_degrees,
            "target_yaw": target_yaw,
            "filenames": filenames,
            "actual_robot_pose": self.pose_to_dict(actual_pose) if actual_pose is not None else None,
        })
        metadata["updated_at"] = datetime.now().isoformat()
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

    def read_json(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, path, data):
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class ManualPoseSequenceGui(QWidget):
    def __init__(self, node: ManualPoseSequenceLogger, signals: GuiSignals):
        super().__init__()
        self.node = node
        self.signals = signals

        self.setWindowTitle("Manual Pose Sequence Logger")
        self.setMinimumWidth(850)

        self.title_label = QLabel("Manual Pose Sequence Logger")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.info_label = QLabel(
            "Drive the robot manually to a location while SLAM is running. "
            "Press the button to create one nXX folder and capture the four GMSL "
            "cameras at yaw offsets 0°, 30°, and 60°."
        )
        self.info_label.setWordWrap(True)

        topics_text = "Configured GMSL topics:\n" + "\n".join(
            [f"- {source['name']}: {source['topic']}" for source in self.node.image_sources]
        )
        self.topics_label = QLabel(topics_text)
        self.topics_label.setWordWrap(True)
        self.topics_label.setStyleSheet("background-color: #f4f4f4; padding: 10px; border-radius: 6px;")

        self.status_label = QLabel("Starting...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("background-color: #eeeeee; padding: 10px; border-radius: 6px;")

        self.capture_button = QPushButton("Capture 3-Pose Sequence")
        self.capture_button.setMinimumHeight(50)
        self.capture_button.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.capture_button.clicked.connect(self.node.request_capture_sequence)

        self.folder_button = QPushButton("Choose Nodes Folder")
        self.folder_button.clicked.connect(self.choose_folder)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.folder_button)

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.topics_label)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.signals.status_update.connect(self.update_status)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose nodes save folder", str(self.node.nodes_dir))
        if folder:
            self.node.set_save_dir(folder)

    def update_status(self, text: str):
        self.status_label.setText(text)


def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    signals = GuiSignals()
    node = ManualPoseSequenceLogger(signals)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    gui = ManualPoseSequenceGui(node, signals)
    gui.show()
    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
