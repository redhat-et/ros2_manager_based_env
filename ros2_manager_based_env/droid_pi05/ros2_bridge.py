
    
  
"""ROS transport only. Isaac API calls stay on the simulator main thread."""
from __future__ import annotations

import json
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger

from ..bridge import BaseRos2Bridge, ActionChunkBuffer

from .contract import (
    decode_chunk,
    CAMERA_KEYS,
    CONTROL_HZ,
    ACTION_DIM,
    MODEL_HORIZON,
)


class DroidBridge(BaseRos2Bridge):
    def __init__(self, prefix="/droid"):
        self.prefix = prefix.rstrip("/")
        super().__init__("isaaclab_droid_pi05")
        self.buffer = ActionChunkBuffer(ACTION_DIM, max_horizon=MODEL_HORIZON,)
        self._received_chunks = 0
        self.images = {k: self.node.create_publisher(Image, f"{self.prefix}/obs/{k}", qos_profile_sensor_data)
                       for k in CAMERA_KEYS}
        self.infos = {k: self.node.create_publisher(CameraInfo, f"{self.prefix}/obs/{k}/camera_info", qos_profile_sensor_data)
                      for k in CAMERA_KEYS}
        self.state = self.node.create_publisher(Float32MultiArray, f"{self.prefix}/obs/state", qos_profile_sensor_data)
        self.status = self.node.create_publisher(String, f"{self.prefix}/status", 1)
        self.node.create_subscription(Float32MultiArray, f"{self.prefix}/action_chunk", self._on_chunk, 1)
        self._trigger_services = [
            self.node.create_service(
                Trigger, f"{self.prefix}/{name}", self._trigger(name),
                callback_group=self._control_group,
            )
            for name in ("pause", "resume", "reset", "step", "stop")
        ]
        self._decimation = 1
        self.start()

    def set_decimation(self, decimation):
        self._decimation = int(decimation)

    def _trigger(self, name):
        async def callback(request, response):
            if name == "reset":
                error = await self._submit("reset")
            elif name == "step":
                error = await self._submit("step", self._decimation)
            else:
                target = {"pause": self.PAUSED, "resume": self.PLAYING,
                          "stop": self.STOPPED}[name]
                error = await self._submit("state", target)
            response.success = error is None
            response.message = error or f"{name} completed"
            return response
        return callback

    def clear_action_buffer(self):
        self.buffer.clear()

    def _on_chunk(self, msg):
        try:
            chunk = decode_chunk(msg.data, tuple(d.size for d in msg.layout.dim), msg.layout.data_offset)
            with self._lock:
                self.buffer.replace(chunk)
                self._received_chunks += 1
        except (ValueError, TypeError) as exc:
            self.buffer.clear()
            self.node.get_logger().error(f"Rejected chunk and cleared buffer: {exc}")

    def action(self):
        row = self.buffer.pop()
        if row is None:
            return np.array([0, 0, 0, 0, 0, 0, 0, -1], dtype=np.float32), True
        # Internal -1 sentinel cannot be supplied through ROS.
        row[:7] = np.clip(row[:7], -1, 1)
        row[7] = np.clip(row[7], 0, 1)
        return row, False

    def publish(self, obs, env, sim_time, idle):
        """Use the ObservationManager result, not an alternate hidden state path."""
        state = np.concatenate([obs["joint_position"][0].detach().cpu().numpy(),
                                obs["gripper_position"][0].detach().cpu().numpy()]).astype(np.float32)
        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="q1..q7_rad,gripper_closure", size=8, stride=8)]
        msg.data = state.tolist()
        self.state.publish(msg)
        stamp = self.node.get_clock().now().to_msg()
        for name, publisher in self.images.items():
            rgb = obs[name][0].detach().cpu().numpy()[..., :3]
            if rgb.dtype != np.uint8:
                raise ValueError(f"Expected raw uint8 camera output for {name}, got {rgb.dtype}")
            rgb = np.ascontiguousarray(rgb)
            image = Image()
            image.header.stamp, image.header.frame_id = stamp, name
            image.height, image.width = rgb.shape[:2]
            image.encoding, image.step = "rgb8", rgb.shape[1] * 3
            image.data = rgb.tobytes()
            publisher.publish(image)
            info = CameraInfo()
            info.header = image.header
            info.height, info.width = image.height, image.width
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            k = env.scene[name].data.intrinsic_matrices[0].detach().cpu().numpy()
            info.k = k.reshape(-1).astype(float).tolist()
            info.r = np.eye(3).reshape(-1).tolist()
            info.p = np.column_stack([k, np.zeros(3)]).reshape(-1).astype(float).tolist()
            self.infos[name].publish(info)
        with self._lock:
            paused = self.is_paused
            received_chunks = self._received_chunks
        status = String()
        status.data = json.dumps({"sim_time": round(sim_time, 6), "control_hz": CONTROL_HZ,
                                  "remaining": self.buffer.remaining, "idle": idle, "paused": paused,
                                  "received_chunks": received_chunks})
        self.status.publish(status)



