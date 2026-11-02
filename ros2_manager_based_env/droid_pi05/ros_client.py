"""Terminal ROS diagnostics and optional native OpenPI policy connection.

Run in a ROS-enabled Python environment; Isaac Sim is not needed here.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

from .contract import CAMERA_KEYS, MODEL_HORIZON, decode_chunk


class Client(Node):
    def __init__(self, prefix):
        super().__init__("droid_pi05_terminal_client")
        self.images = {}
        self.state = None
        self.updated = {}
        self.status = {}
        self.publisher = self.create_publisher(Float32MultiArray, f"{prefix}/action_chunk", 1)
        self.create_subscription(Float32MultiArray, f"{prefix}/obs/state", self.on_state, qos_profile_sensor_data)
        self.create_subscription(String, f"{prefix}/status", self.on_status, 1)
        for k in CAMERA_KEYS:
            self.create_subscription(Image, f"{prefix}/obs/{k}", lambda msg, name=k: self.on_image(name, msg), qos_profile_sensor_data)

    def on_image(self, name, msg):
        if msg.encoding != "rgb8" or msg.height != 224 or msg.width != 224 or msg.step < msg.width * 3:
            raise ValueError(f"Unexpected image contract for {name}")
        self.images[name] = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3).copy()
        self.updated[name] = time.monotonic()

    def on_state(self, msg):
        state = np.asarray(msg.data, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Expected finite state [q1..q7,closure]")
        self.state = state
        self.updated["state"] = time.monotonic()

    def on_status(self, msg):
        self.status = json.loads(msg.data)
        self.updated["status"] = time.monotonic()

    def wait(self, predicate, timeout=30):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError("ROS data/action subscriber unavailable or stale; check ROS_DOMAIN_ID, DDS and topic prefix")
            rclpy.spin_once(self, timeout_sec=0.05)

    def fresh(self):
        return all(time.monotonic() - self.updated.get(k, 0) < 0.5 for k in (*CAMERA_KEYS, "state", "status"))

    def send(self, chunk):
        previous = self.status.get("received_chunks", 0)
        chunk = decode_chunk(np.asarray(chunk, dtype=np.float32).reshape(-1))
        msg = Float32MultiArray()
        h = len(chunk)
        if h:
            msg.layout.dim = [MultiArrayDimension(label="horizon", size=h, stride=h * 8),
                              MultiArrayDimension(label="action", size=8, stride=8)]
        msg.data = chunk.reshape(-1).tolist()
        self.publisher.publish(msg)
        return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "empty", "pulse", "policy"))
    parser.add_argument("--ros-prefix", default="/droid")
    parser.add_argument("--joint", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--command", type=float, default=0.02, help="normalized joint velocity, not rad/s")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default="pick up the red cube")
    parser.add_argument("--execute-steps", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--dump-images", help="optional output directory for RGB PPM images")
    args = parser.parse_args()
    if not 1 <= args.steps <= 15 or not 1 <= args.execute_steps <= 15:
        parser.error("steps and execute-steps must be 1..15")
    if not np.isfinite(args.command) or abs(args.command) > 1:
        parser.error("command must be finite and between -1 and 1")
    rclpy.init()
    client = Client(args.ros_prefix.rstrip("/"))
    try:
        client.wait(lambda: client.fresh() and client.publisher.get_subscription_count() > 0)
        if args.mode == "inspect":
            print("state [q1..q7 rad,closure]:", client.state.tolist())
            for k, image in client.images.items():
                print(k, image.shape, image.dtype, "min/max:", int(image.min()), int(image.max()))
                if args.dump_images:
                    folder = Path(args.dump_images)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{k}.ppm").write_bytes(b"P6\n224 224\n255\n" + image.tobytes())
            print("status:", client.status)
        elif args.mode == "empty":
            if client.status.get("paused"):
                raise RuntimeError("Environment is paused; buffer is already cleared. Resume to test empty-chunk idle.")
            previous = client.send(np.empty((0, 8)))
            client.wait(lambda: client.status.get("received_chunks", 0) > previous and client.status.get("idle") is True)
            print("Empty chunk sent; idle confirmed:", client.status)
        elif args.mode == "pulse":
            if client.status.get("paused"):
                raise RuntimeError("Resume the environment before sending a pulse")
            chunk = np.zeros((args.steps, 8), dtype=np.float32)
            chunk[:, args.joint - 1] = args.command
            chunk[:, 7] = float(client.state[7])
            before = client.state.copy()
            client.send(chunk)
            # Wait beyond all rows and the next idle tick, spinning to observe state.
            deadline = time.monotonic() + args.steps / 15 + 1.0
            while time.monotonic() < deadline:
                rclpy.spin_once(client, timeout_sec=0.05)
            print("Before:", before.tolist())
            print("After: ", client.state.tolist())
            print("Status:", client.status)
        else:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            policy = WebsocketClientPolicy(args.host, args.port)
            for cycle in range(args.cycles):
                client.wait(lambda: client.fresh() and client.status.get("idle") is True and not client.status.get("paused"))
                obs = {CAMERA_KEYS[k]: v.copy() for k, v in client.images.items()}
                obs["observation/joint_position"] = client.state[:7].copy()
                obs["observation/gripper_position"] = client.state[7:].copy()
                obs["prompt"] = args.task
                # Original OpenPI server owns padding/normalization/unnormalization.
                reply = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
                if reply.shape != (MODEL_HORIZON, 8):
                    raise ValueError(f"Expected official pi05_droid [15,8], got {reply.shape}; check server config/checkpoint")
                chunk = decode_chunk(reply.reshape(-1))[:args.execute_steps]
                print(f"Cycle {cycle + 1}: received {reply.shape}; executing {len(chunk)} rows at 15Hz", flush=True)
                previous = client.send(chunk)
                # Do not accept the pre-send idle status as execution completion.
                client.wait(lambda: client.status.get("received_chunks", 0) > previous and client.status.get("idle") is False)
                client.wait(lambda: client.fresh() and client.status.get("idle") is True)
    finally:
        if args.mode in ("pulse", "policy"):
            client.send(np.empty((0, 8)))
            rclpy.spin_once(client, timeout_sec=0.1)
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
