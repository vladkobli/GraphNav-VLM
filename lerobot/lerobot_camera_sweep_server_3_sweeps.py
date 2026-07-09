#!/usr/bin/env python3
"""
LeRobot single-motor m5 pan server for RealSense 8-stop sweep capture.

Run this OUTSIDE Docker in the LeRobot venv, for example:

    /home/vladkobli/rocon-demos/lerobot/.venv/bin/python \
        /home/vladkobli/rocon-demos/lerobot/lerobot/lerobot_camera_sweep_server.py

The ROS2 logger inside Docker calls:

    POST http://<host>:8765/move_stop

with JSON:

    {"stop_index": 1}

Stops:

    1 back        -> m5=172
    2 back_left   -> m5=687
    3 left        -> m5=1203
    4 front_left  -> m5=1718
    5 front       -> m5=2233
    6 front_right -> m5=2748
    7 right       -> m5=3264
    8 back_right  -> m5=3779

Fixed arm posture:

    m1=2875, m2=1059, m3=3128, m4=1173
"""

import json
import math
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


PORT = "/dev/ttyACM0"
HOST = "0.0.0.0"
HTTP_PORT = 8765

FIXED_RAW = {
    "m1": 2875,
    "m2": 1059,
    "m3": 3128,
    "m4": 1100,
}

STOP_DEFS = [
    {"index": 1, "name": "back",        "m5": 172},
    {"index": 2, "name": "back_left",   "m5": 687},
    {"index": 3, "name": "left",        "m5": 1203},
    {"index": 4, "name": "front_left",  "m5": 1718},
    {"index": 5, "name": "front",       "m5": 2233},
    {"index": 6, "name": "front_right", "m5": 2748},
    {"index": 7, "name": "right",       "m5": 3264},
    {"index": 8, "name": "back_right",  "m5": 3779},
]

STOP_BY_INDEX = {s["index"]: s for s in STOP_DEFS}
STOP_BY_NAME = {s["name"]: s for s in STOP_DEFS}

# Default/non-dataset viewing position. Dataset logging still captures all 8 stops
# starting from stop 1 (back), but after a sequence the logger returns here.
DEFAULT_STOP_INDEX = 5
DEFAULT_STOP_NAME = "front"

# Faster than the previous 80 * 0.04 s fixed move.
# Increase MAX_TICKS_PER_STEP or decrease MOVE_DELAY to make it faster.
MAX_TICKS_PER_STEP = 30
MIN_MOVE_STEPS = 8
MAX_MOVE_STEPS = 90
MOVE_DELAY = 0.015
SETTLE_AFTER_MOVE = 0.20
POSITION_TOLERANCE = 150

motors = {
    "m1": Motor(1, "sts3215", MotorNormMode.RANGE_0_100),
    "m2": Motor(2, "sts3215", MotorNormMode.RANGE_0_100),
    "m3": Motor(3, "sts3215", MotorNormMode.RANGE_0_100),
    "m4": Motor(4, "sts3215", MotorNormMode.RANGE_0_100),
    "m5": Motor(5, "sts3215", MotorNormMode.RANGE_0_100),
    # m6 intentionally not used
}

bus = FeetechMotorsBus(port=PORT, motors=motors)
bus_lock = threading.Lock()
connected = False

FRONT_M5 = 2233
LEFT_M5 = 1203
RIGHT_M5 = 3264

TICKS_PER_DEG_LEFT = (FRONT_M5 - LEFT_M5) / 90.0
TICKS_PER_DEG_RIGHT = (RIGHT_M5 - FRONT_M5) / 90.0


def yaw_deg_to_m5(relative_yaw_deg):
    yaw = float(relative_yaw_deg)
    if yaw >= 0.0:
        return int(round(FRONT_M5 - yaw * TICKS_PER_DEG_LEFT))
    return int(round(FRONT_M5 - yaw * TICKS_PER_DEG_RIGHT))


def normalize_stop_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_stop(payload):
    if "relative_yaw_deg" in payload:
        yaw = float(payload["relative_yaw_deg"])
        m5 = yaw_deg_to_m5(yaw)
        return {
            "index": int(payload.get("stop_index", 0)),
            "name": str(payload.get("stop_name", f"yaw_{yaw:+.0f}")),
            "m5": m5,
            "relative_yaw_deg": yaw,
        }

    if "stop_index" in payload:
        stop_index = int(payload["stop_index"])
        if stop_index not in STOP_BY_INDEX:
            raise ValueError(f"Invalid stop_index={stop_index}. Valid: 1..{len(STOP_DEFS)}")
        return STOP_BY_INDEX[stop_index]

    if "stop_name" in payload:
        stop_name = normalize_stop_name(payload["stop_name"])
        if stop_name not in STOP_BY_NAME:
            raise ValueError(
                f"Invalid stop_name={stop_name!r}. Valid: {list(STOP_BY_NAME.keys())}"
            )
        return STOP_BY_NAME[stop_name]

    if "angle_deg" in payload:
        angle = float(payload["angle_deg"])
        if 1.0 <= angle <= float(len(STOP_DEFS)) and abs(angle - round(angle)) < 1e-6:
            return STOP_BY_INDEX[int(round(angle))]
        nearest_i = int(round((angle % 360.0) / 45.0)) + 1
        nearest_i = max(1, min(len(STOP_DEFS), nearest_i))
        return STOP_BY_INDEX[nearest_i]

    raise ValueError("Payload must include stop_index, stop_name, relative_yaw_deg, or angle_deg.")


def stop_to_targets(stop):
    return {
        "m1": FIXED_RAW["m1"],
        "m2": FIXED_RAW["m2"],
        "m3": FIXED_RAW["m3"],
        "m4": FIXED_RAW["m4"],
        "m5": int(stop["m5"]),
    }


def read_positions():
    return {
        name: int(pos)
        for name, pos in bus.sync_read("Present_Position", normalize=False).items()
    }


def read_goals():
    return {
        name: int(pos)
        for name, pos in bus.sync_read("Goal_Position", normalize=False).items()
    }


def read_torque():
    return {
        name: int(val)
        for name, val in bus.sync_read("Torque_Enable", normalize=False).items()
    }


def enable_torque_safely():
    """Set current pose as goal before torque enable, then enable all 5 used motors."""
    current = read_positions()
    bus.sync_write("Goal_Position", current, normalize=False)
    time.sleep(0.2)

    bus.sync_write("Torque_Enable", {name: 1 for name in motors.keys()}, normalize=False)
    time.sleep(0.4)

    current = read_positions()
    bus.sync_write("Goal_Position", current, normalize=False)
    time.sleep(0.2)


def move_to_stop(stop):
    """Move m5 to the requested stop while holding m1/m2/m3/m4 fixed."""
    targets = stop_to_targets(stop)
    start = read_positions()

    max_delta = max(abs(start[name] - targets[name]) for name in targets.keys())
    move_steps = int(math.ceil(max_delta / MAX_TICKS_PER_STEP))
    move_steps = max(MIN_MOVE_STEPS, min(MAX_MOVE_STEPS, move_steps))

    print(
        f"Moving to stop {stop['index']} {stop['name']} "
        f"with {move_steps} steps (max_delta={max_delta})"
    )

    for step_values in zip(
        np.linspace(start["m1"], targets["m1"], move_steps),
        np.linspace(start["m2"], targets["m2"], move_steps),
        np.linspace(start["m3"], targets["m3"], move_steps),
        np.linspace(start["m4"], targets["m4"], move_steps),
        np.linspace(start["m5"], targets["m5"], move_steps),
    ):
        pose = {
            "m1": int(round(step_values[0])),
            "m2": int(round(step_values[1])),
            "m3": int(round(step_values[2])),
            "m4": int(round(step_values[3])),
            "m5": int(round(step_values[4])),
        }
        bus.sync_write("Goal_Position", pose, normalize=False)
        time.sleep(MOVE_DELAY)

    bus.sync_write("Goal_Position", targets, normalize=False)
    time.sleep(SETTLE_AFTER_MOVE)

    actual = read_positions()
    goals = read_goals()
    errors = {name: abs(actual[name] - targets[name]) for name in targets.keys()}

    ok = all(error <= POSITION_TOLERANCE for error in errors.values())
    if not ok:
        raise RuntimeError(
            "Target not reached: "
            f"stop={stop}, targets={targets}, actual={actual}, errors={errors}"
        )

    return {
        "ok": True,
        "stop_index": int(stop["index"]),
        "stop_name": str(stop["name"]),
        "raw_targets": targets,
        "actual_positions": actual,
        "goal_positions": goals,
        "errors": errors,
        "fixed_raw": dict(FIXED_RAW),
        "m5_stop_positions": [int(s["m5"]) for s in STOP_DEFS],
        "stops": STOP_DEFS,
        "move_steps": move_steps,
        "max_delta": max_delta,
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("HTTP: " + fmt % args + "\n")

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            with bus_lock:
                payload = {
                    "ok": True,
                    "connected": connected,
                    "positions": read_positions() if connected else None,
                    "torque": read_torque() if connected else None,
                    "fixed_raw": dict(FIXED_RAW),
                    "default_stop": STOP_BY_INDEX[DEFAULT_STOP_INDEX],
                    "stops": STOP_DEFS,
                }
            self._send_json(200, payload)
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def do_POST(self):
        if self.path not in ("/move_stop", "/move_angle"):
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body) if body else {}
            stop = resolve_stop(payload)

            with bus_lock:
                response = move_to_stop(stop)

            self._send_json(200, response)
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


def shutdown(*_):
    global connected
    print("\nShutting down LeRobot m5 sweep server...")
    try:
        if connected:
            bus.disconnect()
            connected = False
    finally:
        sys.exit(0)


def main():
    global connected
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(f"Connecting to Feetech bus on {PORT}...")
    bus.connect()
    connected = True

    print("Current positions:")
    print(read_positions())

    print("Enabling torque safely...")
    enable_torque_safely()

    print("Server config:")
    print(json.dumps({
        "fixed_raw": FIXED_RAW,
        "default_stop": STOP_BY_INDEX[DEFAULT_STOP_INDEX],
        "stops": STOP_DEFS,
    }, indent=2))

    print(f"Listening on http://{HOST}:{HTTP_PORT}")
    httpd = HTTPServer((HOST, HTTP_PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()