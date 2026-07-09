#!/usr/bin/env python3

import threading

import rclpy
from rclpy.node import Node

from frontier_interfaces.srv import FrontierGoal


class PantherFrontierMarkerProbe(Node):
    """Refresh frontier detector visualization without sending navigation goals."""

    def __init__(self):
        super().__init__("panther_frontier_marker_probe")

        self.declare_parameter("frontier_service", "/frontier_pose")
        self.declare_parameter("refresh_period_sec", 2.0)
        self.declare_parameter("goal_rank", 0)
        self.declare_parameter("service_timeout_sec", 1.0)

        self.frontier_service = str(self.get_parameter("frontier_service").value)
        self.refresh_period_sec = float(self.get_parameter("refresh_period_sec").value)
        self.goal_rank = int(self.get_parameter("goal_rank").value)
        self.service_timeout_sec = float(self.get_parameter("service_timeout_sec").value)

        self.client = self.create_client(FrontierGoal, self.frontier_service)
        self.request_in_flight = False
        self.timer = self.create_timer(self.refresh_period_sec, self.timer_cb)

        self.get_logger().info(
            f"Refreshing frontier markers through {self.frontier_service} every "
            f"{self.refresh_period_sec:.1f}s"
        )

    def timer_cb(self):
        if self.request_in_flight:
            return

        threading.Thread(target=self.request_frontiers, daemon=True).start()

    def request_frontiers(self):
        self.request_in_flight = True
        try:
            if not self.client.wait_for_service(timeout_sec=self.service_timeout_sec):
                self.get_logger().warn(f"Frontier service not available: {self.frontier_service}")
                return

            req = FrontierGoal.Request()
            req.goal_rank = self.goal_rank
            future = self.client.call_async(req)
            future.add_done_callback(self.request_done)
        except Exception as exc:
            self.request_in_flight = False
            self.get_logger().warn(f"Failed to refresh frontier markers: {exc}")

    def request_done(self, future):
        self.request_in_flight = False
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warn(f"Frontier marker refresh failed: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = PantherFrontierMarkerProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
