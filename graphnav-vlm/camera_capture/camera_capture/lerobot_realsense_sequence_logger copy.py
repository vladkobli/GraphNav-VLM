#!/usr/bin/env python3

import json
import math
import re
import sys
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

import cv2
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_ros import Buffer, TransformListener

from sensor_msgs.msg import Image
from PyQt5.QtCore import Qt, pyqtSignal, QObject
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


class LeRobotRealsenseSequenceLogger(SnapshotCaptureNode):
    """
    GUI node for manually triggered LeRobot + RealSense dataset capture.

    Robust mode: the node does not abort the capture just because the RealSense
    frame stamp did not change quickly enough. It keeps the subscriptions alive,
    waits until color and aligned depth are available, then saves the latest
    received frames after each LeRobot movement. This matches the older behavior
    that was reliable for dataset capture.
    """

    def __init__(self, signals: GuiSignals):
        super().__init__(signals, node_name="lerobot_realsense_sequence_logger")

        self.declare_parameter("nodes_dir", "/rgbd_camera_intel_dev/src/nodes")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_frame", "panther/base_link")
        self.declare_parameter("tf_timeout_sec", 1.0)

        self.declare_parameter("lerobot_server_url", "http://172.17.0.1:8765")
        self.declare_parameter(
            "camera_stop_names",
            [
                "back",
                "back_left",
                "left",
                "front_left",
                "front",
                "front_right",
                "right",
                "back_right",
            ],
        )
        self.declare_parameter("settle_after_motion_sec", 1.0)

        # This was passed from your launch file but the old node did not use it.
        self.declare_parameter("wait_for_fresh_frame_sec", 60.0)

        # Optional extra safety: after detecting a first fresh frame, wait for this many
        # additional frame updates. This helps avoid saving the first frame immediately
        # after motion if the camera/USB pipeline is still settling.
        self.declare_parameter("fresh_frames_to_skip", 0)

        self.nodes_dir = Path(self.get_parameter("nodes_dir").value)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = self.nodes_dir

        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)

        self.lerobot_server_url = str(
            self.get_parameter("lerobot_server_url").value
        ).rstrip("/")
        self.camera_stop_names = [
            str(v) for v in self.get_parameter("camera_stop_names").value
        ]
        self.settle_after_motion_sec = float(
            self.get_parameter("settle_after_motion_sec").value
        )
        self.wait_for_fresh_frame_sec = float(
            self.get_parameter("wait_for_fresh_frame_sec").value
        )
        self.fresh_frames_to_skip = int(
            self.get_parameter("fresh_frames_to_skip").value
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sequence_running = False
        self.sequence_lock = threading.Lock()

        # Keep ONLY the RealSense sources from the old SnapshotCaptureNode.
        self.image_sources = [
            source
            for source in self.image_sources
            if source["name"] in ("realsense_color", "realsense_aligned_depth")
        ]

        # Destroy subscriptions created by parent, then recreate only RealSense ones.
        for sub in list(self.subscribers):
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass

        self.subscribers = []
        with self.latest_lock:
            self.latest_images = {}
            self.latest_receive_time = {}

        # RealSense image topics usually use sensor-data style QoS.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.get_logger().info("LeRobot RealSense sequence logger subscribing to:")
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
            "Ready.\n"
            "Press the button to rotate the LeRobot camera and capture fresh "
            "RealSense color + aligned depth at each stop.\n\n"
            f"Nodes folder:\n{self.nodes_dir}\n\n"
            f"LeRobot server:\n{self.lerobot_server_url}\n\n"
            f"Stops:\n{self.camera_stop_names}"
        )

    def request_capture_sequence(self):
        with self.sequence_lock:
            if self.sequence_running:
                self.signals.status_update.emit("Sequence already running.")
                return
            self.sequence_running = True

        threading.Thread(target=self.run_capture_sequence, daemon=True).start()

    def run_capture_sequence(self):
        folder = None

        try:
            self.check_lerobot_server()

            if not self.wait_until_realsense_available(timeout_sec=120.0):
                raise RuntimeError(
                    "RealSense color/depth not available in logger. "
                    "Check topic names and QoS."
                )

            node_id = self.next_node_id()
            folder = self.make_node_folder(node_id)
            node_id = self.node_id_from_folder(folder)

            self.write_initial_metadata(folder, node_id)
            self.signals.status_update.emit(f"Started sequence.\nFolder:\n{folder}")

            previous_tokens = None

            # Default/home is front = stop 5.
            # Dataset still starts at back = stop 1.
            # Move from front to back through intermediate stops, without saving images.
            preposition_responses = self.move_through_stops(
                [2, 1],
                reason=f"{folder.name}: pre-positioning to dataset start"
            )

            metadata = self.read_json(self.metadata_path(folder))
            metadata["preposition_to_start"] = {
                "created_at": datetime.now().isoformat(),
                "from_default_stop": 5,
                "to_first_capture_stop": 1,
                "path": [2, 1],
                "responses": preposition_responses,
            }
            self.write_json(self.metadata_path(folder), metadata)

            for i, stop_name in enumerate(self.camera_stop_names, start=1):
                self.signals.status_update.emit(
                    f"{folder.name}: moving camera to stop {i} - {stop_name} "
                    f"({i}/{len(self.camera_stop_names)})"
                )

                move_response = self.move_lerobot_stop(i, stop_name)

                # Let the arm/camera settle. Keep the ROS subscriptions alive.
                self.sleep_seconds(self.settle_after_motion_sec)

                # Do not destroy/recreate subscriptions here. On this setup DDS discovery
                # can take 15-25 seconds after resubscribe, which caused false failures.
                if not self.wait_until_realsense_available(timeout_sec=self.wait_for_fresh_frame_sec):
                    raise RuntimeError(
                        f"Timed out waiting for RealSense frames after "
                        f"moving to stop {i} - {stop_name}."
                    )

                current_tokens = self.current_frame_tokens()
                if previous_tokens is not None and current_tokens == previous_tokens:
                    self.get_logger().warn(
                        "RealSense token did not change since previous stop. "
                        "Saving latest cached frames anyway."
                    )
                previous_tokens = current_tokens

                filenames, stamps = self.save_realsense_stop_snapshot(
                    folder=folder,
                    stop_index=i,
                    stop_name=stop_name,
                )

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

            # After all dataset photos are captured, return to the normal/default
            # viewing direction: FRONT. This is NOT an extra dataset capture.
            default_stop_index = 5
            default_stop_name = "front"

            self.signals.status_update.emit(
                f"{folder.name}: returning camera to default/front position "
                f"stop {default_stop_index} - {default_stop_name}"
            )

            metadata = self.read_json(self.metadata_path(folder))

            try:
                return_home_responses = self.move_through_stops(
                    [5],
                    reason=f"{folder.name}: returning to default/front"
                )

                metadata["return_home"] = {
                    "returned_at": datetime.now().isoformat(),
                    "ok": True,
                    "default_stop_index": 5,
                    "default_stop_name": "front",
                    "path": [5],
                    "responses": return_home_responses,
                }

            except Exception as exc:
                self.get_logger().warn(f"Return-home failed, but capture is complete: {exc}")

                metadata["return_home"] = {
                    "returned_at": datetime.now().isoformat(),
                    "ok": False,
                    "default_stop_index": 5,
                    "default_stop_name": "front",
                    "path": [5],
                    "error": str(exc),
                }

            self.write_json(self.metadata_path(folder), metadata)

            self.mark_metadata_status(folder, "completed")
            self.signals.status_update.emit(
                f"Sequence complete and camera returned to front/default.\nFolder:\n{folder}"
            )

        except Exception as exc:
            self.get_logger().error(f"Sequence failed: {exc}")
            if folder is not None:
                self.mark_metadata_status(folder, "failed", error=str(exc))
            self.signals.status_update.emit(f"Sequence failed:\n{exc}")

        finally:
            with self.sequence_lock:
                self.sequence_running = False

    def msg_stamp_tuple(self, msg):
        return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))

    def current_frame_tokens(self):
        """Return receive-time and ROS-stamp tokens for the current latest frames."""
        with self.latest_lock:
            tokens = {}
            for name, msg in self.latest_images.items():
                recv_time = self.latest_receive_time.get(name)
                recv_ns = recv_time.nanoseconds if recv_time is not None else None
                tokens[name] = {
                    "recv_ns": recv_ns,
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

    def wait_for_fresh_realsense_frames(self, baseline_tokens, timeout_sec, extra_frames=0):
        required = ["realsense_color", "realsense_aligned_depth"]
        start = self.get_clock().now()
        last_seen = dict(baseline_tokens)
        fresh_count = {name: 0 for name in required}

        while rclpy.ok():
            all_fresh_enough = True

            with self.latest_lock:
                for name in required:
                    msg = self.latest_images.get(name)
                    recv_time = self.latest_receive_time.get(name)
                    if msg is None or recv_time is None:
                        all_fresh_enough = False
                        continue

                    current = {
                        "recv_ns": recv_time.nanoseconds,
                        "stamp": self.msg_stamp_tuple(msg),
                    }
                    previous = last_seen.get(name)
                    baseline = baseline_tokens.get(name)

                    changed_from_previous = previous is None or current != previous
                    changed_from_baseline = baseline is None or current != baseline

                    if changed_from_previous:
                        last_seen[name] = current
                        if changed_from_baseline:
                            fresh_count[name] += 1

                    if fresh_count[name] < (1 + max(0, extra_frames)):
                        all_fresh_enough = False

            if all_fresh_enough:
                return True

            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn(
                    f"Fresh-frame timeout. Counts={fresh_count}, "
                    f"baseline={baseline_tokens}, latest={self.current_frame_tokens()}"
                )
                return False

            self.sleep_seconds(0.02)

    @staticmethod
    def safe_name(name):
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    def save_realsense_stop_snapshot(self, folder, stop_index, stop_name):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        safe_stop_name = self.safe_name(stop_name)
        file_map = {
            "realsense_color": f"c{stop_index}_{safe_stop_name}.png",
            "realsense_aligned_depth": f"d{stop_index}_{safe_stop_name}.png",
        }

        with self.latest_lock:
            images_copy = dict(self.latest_images)

        filenames = []
        stamps = {}

        for source in self.image_sources:
            name = source["name"]
            if name not in file_map:
                continue
            if name not in images_copy:
                raise RuntimeError(f"Missing required image source: {name}")

            msg = images_copy[name]
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            image_to_save = self.prepare_image_for_saving(cv_img, msg.encoding)

            filename = file_map[name]
            path = folder / filename
            if not cv2.imwrite(str(path), image_to_save):
                raise RuntimeError(f"cv2.imwrite failed for {path}")

            filenames.append(filename)
            stamps[name] = {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
                "encoding": msg.encoding,
                "height": int(msg.height),
                "width": int(msg.width),
            }

        self.get_logger().info(
            f"Saved stop {stop_index}: {', '.join(filenames)}"
        )
        return filenames, stamps

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
        body = json.dumps({
            "stop_index": int(stop_index),
            "stop_name": str(stop_name),
        }).encode("utf-8")

        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=30.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise RuntimeError(f"LeRobot HTTP error: {exc.code} {exc.reason}")
        except URLError as exc:
            raise RuntimeError(f"LeRobot URL error: {exc}")

        if not payload.get("ok", False):
            raise RuntimeError(f"LeRobot move failed: {payload}")

        self.get_logger().info(f"LeRobot moved: {payload}")
        return payload

    def set_save_dir(self, save_dir):
        self.nodes_dir = Path(save_dir)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = self.nodes_dir
        self.signals.status_update.emit(f"nodes_dir changed to:\n{self.nodes_dir}")

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

    def move_through_stops(self, stop_indices, reason="moving"):
        responses = []

        for stop_index in stop_indices:
            stop_name = self.camera_stop_names[int(stop_index) - 1]

            self.signals.status_update.emit(
                f"{reason}: moving through stop {stop_index} - {stop_name}..."
            )

            response = self.move_lerobot_stop(stop_index, stop_name)
            responses.append(response)

            # Small mechanical settling only; no image saved here.
            self.sleep_seconds(0.15)

        return responses

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
        return self.next_node_id()

    def metadata_path(self, folder):
        return Path(folder) / "metadata.json"

    def write_initial_metadata(self, folder, node_id):
        pose = self.lookup_robot_pose()
        now = datetime.now().isoformat()

        planned_pose = None
        planned_frame_id = self.global_frame
        if pose is not None:
            planned_frame_id = pose["frame_id"]
            planned_pose = {
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"],
                "yaw": pose["yaw"],
                "orientation": pose["orientation"],
            }

        metadata = {
            "node_id": node_id,
            "node_name": self.node_name(node_id),
            "created_at": now,
            "planned_frame_id": planned_frame_id,
            "planned_pose": planned_pose,
            "capture_mode": "lerobot_realsense_sweep",
            "lerobot_server_url": self.lerobot_server_url,
            "camera_stop_names": self.camera_stop_names,
            "realsense_color_topic": self.get_parameter("realsense_color_topic").value,
            "realsense_aligned_depth_topic": self.get_parameter("realsense_aligned_depth_topic").value,
            "snapshots": [],
        }
        self.write_json(self.metadata_path(folder), metadata)

    def append_snapshot_metadata(self, folder, label, stop_index, stop_name, move_response, filenames, image_stamps, frame_tokens=None):
        path = self.metadata_path(folder)
        metadata = self.read_json(path)

        snapshot = {
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
        }
        metadata["snapshots"].append(snapshot)

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

    def read_json(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, path, data):
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def sleep_seconds(seconds):
        threading.Event().wait(max(0.0, seconds))


class LeRobotRealsenseSequenceGui(QWidget):
    def __init__(self, node, signals):
        super().__init__()

        self.node = node
        self.signals = signals

        self.setWindowTitle("LeRobot RealSense Sweep Logger")
        self.setMinimumWidth(850)

        self.title_label = QLabel("LeRobot RealSense Sweep Logger")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.info_label = QLabel(
            "Press the button to rotate the LeRobot-mounted RealSense camera "
            "to the configured stops and save fresh color + aligned depth at each stop."
        )
        self.info_label.setWordWrap(True)

        self.status_label = QLabel("Starting...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #eeeeee; padding: 10px; border-radius: 6px;"
        )

        self.capture_button = QPushButton("Capture RealSense Sweep")
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
        layout.addLayout(button_layout)
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.signals.status_update.connect(self.update_status)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose nodes save folder",
            str(self.node.nodes_dir),
        )
        if folder:
            self.node.set_save_dir(folder)

    def update_status(self, text):
        self.status_label.setText(text)


def main(args=None):
    rclpy.init(args=args)

    app = QApplication(sys.argv)
    signals = GuiSignals()
    node = LeRobotRealsenseSequenceLogger(signals)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    gui = LeRobotRealsenseSequenceGui(node, signals)
    gui.show()

    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()