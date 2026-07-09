#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

import matplotlib.patheffects as pe

from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe


class MapMarkerExporter(Node):
    def __init__(self):
        super().__init__("map_marker_exporter")

        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("marker_topic", "/metadata_node_markers2")
        self.declare_parameter("output_prefix", "map_export")
        self.declare_parameter("dpi", 600)

        self.map_msg = None
        self.marker_msg = None

        map_topic = self.get_parameter("map_topic").value
        marker_topic = self.get_parameter("marker_topic").value

        self.create_subscription(
            OccupancyGrid,
            map_topic,
            self.map_callback,
            10,
        )

        self.create_subscription(
            MarkerArray,
            marker_topic,
            self.marker_callback,
            10,
        )

        self.timer = self.create_timer(1.0, self.try_export_once)

        self.get_logger().info(f"Waiting for map on {map_topic}")
        self.get_logger().info(f"Waiting for markers on {marker_topic}")

    def map_callback(self, msg):
        self.map_msg = msg

    def marker_callback(self, msg):
        self.marker_msg = msg

    def try_export_once(self):
        if self.map_msg is None:
            return

        # MarkerArray is optional. Export map even if no markers arrived.
        self.export()
        self.get_logger().info("Export finished.")
        rclpy.shutdown()

    def export(self):
        output_prefix = self.get_parameter("output_prefix").value
        dpi = int(self.get_parameter("dpi").value)

        grid = self.map_msg
        width = grid.info.width
        height = grid.info.height
        resolution = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y

        data = np.array(grid.data, dtype=np.int16).reshape((height, width))

        # Convert occupancy values to display image:
        # unknown -1 -> light gray
        # free 0 -> white
        # occupied 100 -> black
        img = np.zeros((height, width), dtype=np.float32)
        img[data == -1] = 0.75
        img[data == 0] = 1.0
        img[data > 0] = 1.0 - (data[data > 0] / 100.0)

        extent = [
            origin_x,
            origin_x + width * resolution,
            origin_y,
            origin_y + height * resolution,
        ]

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.imshow(
            img,
            cmap="gray",
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )

        if self.marker_msg is not None:
            self.draw_markers(ax, self.marker_msg)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)

        plt.tight_layout()

        fig.savefig(f"{output_prefix}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{output_prefix}.tiff", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{output_prefix}.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")

        plt.close(fig)

    
    def draw_markers(self, ax, marker_array):
        # First pass: draw all non-text markers
        for marker in marker_array.markers:
            if marker.action == Marker.DELETE:
                continue

            if marker.type == Marker.TEXT_VIEW_FACING:
                continue

            if marker.type == Marker.SPHERE:
                self.draw_sphere(ax, marker)

            elif marker.type == Marker.CUBE:
                self.draw_cube(ax, marker)

            elif marker.type == Marker.LINE_STRIP:
                self.draw_line_strip(ax, marker)

            elif marker.type == Marker.LINE_LIST:
                self.draw_line_list(ax, marker)

            elif marker.type == Marker.ARROW:
                self.draw_arrow(ax, marker)

        # Second pass: draw text labels on top of everything
        for marker in marker_array.markers:
            if marker.action == Marker.DELETE:
                continue

            if marker.type == Marker.TEXT_VIEW_FACING:
                self.draw_text(ax, marker)

    def rgba(self, color):
        return (
            color.r,
            color.g,
            color.b,
            color.a if color.a > 0.0 else 1.0,
        )

    def marker_xy(self, marker):
        return marker.pose.position.x, marker.pose.position.y

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def draw_sphere(self, ax, marker):
        x, y = self.marker_xy(marker)
        radius = max(marker.scale.x, marker.scale.y) / 2.0
        color = self.rgba(marker.color)

        circle = Circle(
            (x, y),
            radius,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        )
        ax.add_patch(circle)

    def draw_cube(self, ax, marker):
        x, y = self.marker_xy(marker)
        sx = marker.scale.x
        sy = marker.scale.y
        color = self.rgba(marker.color)

        rect = Rectangle(
            (x - sx / 2.0, y - sy / 2.0),
            sx,
            sy,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        )
        ax.add_patch(rect)

    def draw_line_strip(self, ax, marker):
        if len(marker.points) < 2:
            return

        xs = [p.x for p in marker.points]
        ys = [p.y for p in marker.points]
        color = self.rgba(marker.color)
        linewidth = max(marker.scale.x, 0.01) * 10.0

        ax.plot(xs, ys, linewidth=linewidth, color=color)

    def draw_line_list(self, ax, marker):
        if len(marker.points) < 2:
            return

        color = self.rgba(marker.color)
        linewidth = max(marker.scale.x, 0.01) * 10.0

        points = marker.points
        for i in range(0, len(points) - 1, 2):
            p1 = points[i]
            p2 = points[i + 1]
            ax.plot(
                [p1.x, p2.x],
                [p1.y, p2.y],
                linewidth=linewidth,
                color=color,
            )

    def draw_arrow(self, ax, marker):
        color = self.rgba(marker.color)

        if len(marker.points) >= 2:
            start = marker.points[0]
            end = marker.points[1]
            x1, y1 = start.x, start.y
            x2, y2 = end.x, end.y
        else:
            x1, y1 = self.marker_xy(marker)
            yaw = self.yaw_from_quaternion(marker.pose.orientation)
            length = marker.scale.x if marker.scale.x > 0 else 0.5
            x2 = x1 + length * math.cos(yaw)
            y2 = y1 + length * math.sin(yaw)

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=7.0,
            linewidth=0.75,
            color=color,
        )
        ax.add_patch(arrow)

    def draw_text(self, ax, marker):
        x, y = self.marker_xy(marker)

        text = marker.text
        if not text:
            return

        # RViz TEXT_VIEW_FACING uses scale.z as text height in meters.
        # Tune fallback if your text marker scale.z is zero or weird.
        target_height_m = marker.scale.z if marker.scale.z > 0.0 else 0.45

        # Optional: clamp text size so labels don't explode
        target_height_m = min(max(target_height_m, 0.25), 0.75)

        font = FontProperties(weight="bold")

        # Create text as geometry, not as screen-font text
        tp = TextPath((0, 0), text, size=1.0, prop=font)
        bbox = tp.get_extents()

        if bbox.height <= 0:
            return

        scale = target_height_m / bbox.height

        # Center text on marker pose, like RViz-style marker text
        cx = (bbox.x0 + bbox.x1) / 2.0 - 0.4 * bbox.width  # slight horizontal offset for better centering
        cy = (bbox.y0 + bbox.y1) / 2.0 - 1.1 * bbox.height  # slight vertical offset for better centering

        transform = (
            Affine2D()
            .translate(-cx, -cy)
            .scale(scale)
            .translate(x, y)
            + ax.transData
        )

        patch = PathPatch(
            tp,
            transform=transform,
            facecolor="white",
            edgecolor="none",
            zorder=1000,
        )

        # Thin dark outline so white text is visible on white/gray map
        patch.set_path_effects([
            pe.Stroke(linewidth=0.5, foreground="black"),
            pe.Normal(),
        ])

        ax.add_patch(patch)


def main():
    rclpy.init()
    node = MapMarkerExporter()
    rclpy.spin(node)


if __name__ == "__main__":
    main()