#!/usr/bin/env python3

import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject


class GuiSignals(QObject):
    status_update = pyqtSignal(str)


class SnapshotCaptureNode(Node):
    def __init__(self, signals: GuiSignals, node_name: str = "snapshot_capture"):
        super().__init__(node_name)

        self.signals = signals
        self.bridge = CvBridge()

        # ----------------------------------------------------------------------
        # Parameters
        # ----------------------------------------------------------------------

        # RealSense uncompressed color + aligned depth
        self.declare_parameter(
            "realsense_color_topic",
            "/camera/driver/color/image_raw",
        )
        self.declare_parameter(
            "realsense_aligned_depth_topic",
            "/camera/driver/aligned_depth_to_color/image_raw",
        )

        # GMSL color camera topics
        self.declare_parameter("gmsl_cam0_topic", "/camera0/image_color")
        self.declare_parameter("gmsl_cam1_topic", "/camera1/image_color")
        self.declare_parameter("gmsl_cam2_topic", "/camera2/image_color")
        self.declare_parameter("gmsl_cam3_topic", "/camera3/image_color")

        self.declare_parameter(
            "save_dir",
            "/rgbd_camera_intel_dev/realsense_captures",
        )

        # If true, save only when all 6 topics have at least one frame.
        # If false, save whatever is available and report missing topics.
        self.declare_parameter("require_all_topics", False)

        self.save_dir = Path(self.get_parameter("save_dir").value)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.require_all_topics = bool(
            self.get_parameter("require_all_topics").value
        )

        self.declare_parameter("expected_gmsl_width", 1640)
        self.declare_parameter("expected_gmsl_height", 1232)
        self.declare_parameter("warn_on_unexpected_resolution", True)

        self.expected_gmsl_width = int(self.get_parameter("expected_gmsl_width").value)
        self.expected_gmsl_height = int(self.get_parameter("expected_gmsl_height").value)
        self.warn_on_unexpected_resolution = bool(
            self.get_parameter("warn_on_unexpected_resolution").value
        )

        self.image_sources = [
            {
                "name": "realsense_color",
                "topic": self.get_parameter("realsense_color_topic").value,
                "kind": "realsense_color",
                "expected_width": None,
                "expected_height": None,
            },
            {
                "name": "realsense_aligned_depth",
                "topic": self.get_parameter("realsense_aligned_depth_topic").value,
                "kind": "realsense_depth",
                "expected_width": None,
                "expected_height": None,
            },
            {
                "name": "gmsl_camera0_color_fullfov",
                "topic": self.get_parameter("gmsl_cam0_topic").value,
                "kind": "gmsl_color",
                "expected_width": self.expected_gmsl_width,
                "expected_height": self.expected_gmsl_height,
            },
            {
                "name": "gmsl_camera1_color_fullfov",
                "topic": self.get_parameter("gmsl_cam1_topic").value,
                "kind": "gmsl_color",
                "expected_width": self.expected_gmsl_width,
                "expected_height": self.expected_gmsl_height,
            },
            {
                "name": "gmsl_camera2_color_fullfov",
                "topic": self.get_parameter("gmsl_cam2_topic").value,
                "kind": "gmsl_color",
                "expected_width": self.expected_gmsl_width,
                "expected_height": self.expected_gmsl_height,
            },
            {
                "name": "gmsl_camera3_color_fullfov",
                "topic": self.get_parameter("gmsl_cam3_topic").value,
                "kind": "gmsl_color",
                "expected_width": self.expected_gmsl_width,
                "expected_height": self.expected_gmsl_height,
            },
        ]

        # ----------------------------------------------------------------------
        # Runtime state
        # ----------------------------------------------------------------------

        self.latest_images = {}
        self.latest_receive_time = {}
        self.latest_lock = threading.Lock()

        self.subscribers = []

        # For image topics, especially RealSense/GMSL, BEST_EFFORT is usually safer.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.get_logger().info("Snapshot capture node started.")
        self.get_logger().info("Subscribing to topics:")

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

        self.status_timer = self.create_timer(2.0, self.print_status)

        self.signals.status_update.emit(
            "Ready.\n"
            "Press the button to save the latest available frames.\n\n"
            f"Save directory:\n{self.save_dir}\n\n"
            "Waiting for image topics..."
        )

    def image_callback(self, source_name: str, msg: Image):
        with self.latest_lock:
            self.latest_images[source_name] = msg
            self.latest_receive_time[source_name] = self.get_clock().now()

    def print_status(self):
        with self.latest_lock:
            received_names = set(self.latest_images.keys())

        total = len(self.image_sources)
        received = len(received_names)

        missing = [
            source["name"]
            for source in self.image_sources
            if source["name"] not in received_names
        ]

        if missing:
            self.get_logger().warn(
                f"Received frames from {received}/{total} topics. "
                f"Missing: {', '.join(missing)}"
            )
        else:
            self.get_logger().info(
                f"Received frames from all {total} topics. Ready to capture."
            )

    def request_capture(self):
        self.get_logger().info("Snapshot requested.")

        self.signals.status_update.emit(
            "Snapshot requested.\n"
            "Saving latest available frames..."
        )

        # Save in a background thread so the GUI does not freeze.
        save_thread = threading.Thread(
            target=self.save_latest_snapshot,
            daemon=True,
        )
        save_thread.start()

    def set_save_dir(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"Save directory changed to: {self.save_dir}")
        self.signals.status_update.emit(f"Save directory changed to:\n{self.save_dir}")

    def save_latest_snapshot(self, capture_folder=None, filename_prefix: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if capture_folder is None:
            capture_folder = self.save_dir / f"snapshot_{timestamp}"
        else:
            capture_folder = Path(capture_folder)
        capture_folder.mkdir(parents=True, exist_ok=True)

        try:
            with self.latest_lock:
                images_copy = dict(self.latest_images)

            saved_info = []
            missing_info = []

            for source in self.image_sources:
                source_name = source["name"]
                topic = source["topic"]

                if source_name not in images_copy:
                    missing_info.append(f"{source_name}: {topic}")
                    continue

                msg = images_copy[source_name]

                cv_img = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )

                encoding = msg.encoding
                stamp_sec = msg.header.stamp.sec
                stamp_nanosec = msg.header.stamp.nanosec

                filename = capture_folder / f"{filename_prefix}{source_name}.png"

                image_to_save = self.prepare_image_for_saving(cv_img, encoding)

                success = cv2.imwrite(str(filename), image_to_save)

                if not success:
                    raise RuntimeError(f"cv2.imwrite failed for {source_name}")

                saved_info.append(
                    f"{source_name}: "
                    f"topic={topic}, "
                    f"stamp={stamp_sec}.{stamp_nanosec:09d}, "
                    f"encoding={encoding}, "
                    f"file={filename.name}"
                )

            if self.require_all_topics and missing_info:
                status = (
                    "Snapshot NOT saved completely because some topics are missing.\n\n"
                    f"Folder created:\n{capture_folder}\n\n"
                    "Missing topics:\n"
                    + "\n".join(missing_info)
                )

                self.get_logger().warn(status)
                self.signals.status_update.emit(status)
                return

            status_parts = [
                f"Snapshot folder:\n{capture_folder}",
            ]

            if saved_info:
                status_parts.append("Saved:\n" + "\n".join(saved_info))

            if missing_info:
                status_parts.append("Missing/no frame received:\n" + "\n".join(missing_info))

            status = "\n\n".join(status_parts)

            self.get_logger().info(status)
            self.signals.status_update.emit(status)
            return capture_folder

        except Exception as exc:
            error_msg = f"Failed to save snapshot: {exc}"
            self.get_logger().error(error_msg)
            self.signals.status_update.emit(error_msg)
            raise

    def prepare_image_for_saving(self, cv_img, encoding: str):
        encoding = encoding.lower()

        # RealSense color may be rgb8. OpenCV writes BGR correctly.
        if encoding == "rgb8":
            return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        if encoding == "rgba8":
            return cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)

        # RealSense depth 16UC1 and GMSL mono16 are preserved as 16-bit PNG.
        return cv_img


class SnapshotCaptureGui(QWidget):
    def __init__(self, node: SnapshotCaptureNode, signals: GuiSignals):
        super().__init__()

        self.node = node
        self.signals = signals

        self.setWindowTitle("Snapshot Capture")
        self.setMinimumWidth(850)

        self.title_label = QLabel("Snapshot Capture")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.info_label = QLabel(
            "Press the button to save the latest available frames from the "
            "RealSense camera and the four GMSL cameras."
        )
        self.info_label.setWordWrap(True)

        topics_text = "Configured topics:\n" + "\n".join(
            [f"- {source['name']}: {source['topic']}" for source in self.node.image_sources]
        )

        self.topics_label = QLabel(topics_text)
        self.topics_label.setWordWrap(True)
        self.topics_label.setStyleSheet(
            "background-color: #f4f4f4; padding: 10px; border-radius: 6px;"
        )

        self.status_label = QLabel("Starting...")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "background-color: #eeeeee; padding: 10px; border-radius: 6px;"
        )

        self.capture_button = QPushButton("Capture Snapshot")
        self.capture_button.setMinimumHeight(50)
        self.capture_button.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.capture_button.clicked.connect(self.node.request_capture)

        self.folder_button = QPushButton("Choose Save Folder")
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
        folder = QFileDialog.getExistingDirectory(
            self,
            "Choose snapshot save folder",
            str(self.node.save_dir),
        )

        if folder:
            self.node.set_save_dir(folder)

    def update_status(self, text: str):
        self.status_label.setText(text)


def main(args=None):
    rclpy.init(args=args)

    app = QApplication(sys.argv)

    signals = GuiSignals()
    node = SnapshotCaptureNode(signals)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    ros_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    ros_thread.start()

    gui = SnapshotCaptureGui(node, signals)
    gui.show()

    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
