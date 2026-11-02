"""custom_env_3_3 ROS transport using shared control and chunk buffer classes."""
from __future__ import annotations

import numpy as np
import torch
import json

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

from ..bridge import BaseRos2Bridge, ActionChunkBuffer
from .contract import (
    CONTROL_HZ, ACTION_DIM, MAX_CHUNK_HORIZON, INITIAL_GRIPPER_COMMAND,
    CAMERA_KEYS, STATE_TOPIC, ACTION_TOPIC, CAMERA_TOPIC_PREFIX, NODE_NAME,
    decode_chunk, idle_action, encode_state,
)
from .observations import eef_pose_axis_angle, gripper_pos


class Ros2VlaBridge(BaseRos2Bridge):
    """7D Cartesian actions; measured 8D eef/finger state.

    The public get_action(), publish_observations(), shutdown() interface is
    preserved. Simulation services and queue processing belong to the base.
    """

    def __init__(
        self,
        camera_names=None,
        env=None,
        state_topic=STATE_TOPIC,
        action_topic=ACTION_TOPIC,
        camera_topic_prefix=CAMERA_TOPIC_PREFIX,
        node_name=NODE_NAME,
        action_dim=ACTION_DIM,
    ):
        if action_dim != ACTION_DIM:
            raise ValueError(f'custom_env_3_3 requires action_dim={ACTION_DIM}')
        if env is not None and env.num_envs != 1:
            raise ValueError('Live ROS bridge requires num_envs=1')
        super().__init__(node_name)
        self._camera_names = list(CAMERA_KEYS) if camera_names is None else list(camera_names)
        self._action_dim = ACTION_DIM
        self._env = env
        self.buffer = ActionChunkBuffer(ACTION_DIM, max_horizon=MAX_CHUNK_HORIZON)
        self._last_gripper_command = INITIAL_GRIPPER_COMMAND

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._image_publishers = {
            name: self.node.create_publisher(Image, f'{camera_topic_prefix}{name}', sensor_qos)
            for name in self._camera_names
        }
        self._state_publisher = self.node.create_publisher(
            Float32MultiArray, state_topic, sensor_qos)
        action_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._action_subscriber = self.node.create_subscription(
            Float32MultiArray, action_topic, self._on_action_chunk_msg, action_qos)
        self._received_chunks = 0
        self._last_action_idle = True
        self.status = self.node.create_publisher(String, "/vla/status", 1)            
        self.start()

    def clear_action_buffer(self):
        with self._lock:
            self.buffer.clear()
            self._last_gripper_command = INITIAL_GRIPPER_COMMAND
            self._last_action_idle = True

    def _on_action_chunk_msg(self, msg):
        try:
            chunk = decode_chunk(
                msg.data,
                tuple(d.size for d in msg.layout.dim),
                msg.layout.data_offset,
            )
            with self._lock:
                if len(chunk) == 0:
                    self.clear_action_buffer()
                else:
                    self.buffer.replace(chunk)

                # Count every accepted message, including an empty chunk.
                self._received_chunks += 1

        except (ValueError, TypeError) as exc:
            self.node.get_logger().warning(
                f"Rejected action chunk: {exc}"
            )

    def _next_action_row(self):
        with self._lock:
            row = self.buffer.pop()
            self._last_action_idle = row is None

            if row is None:
                return idle_action(self._last_gripper_command)

            self._last_gripper_command = float(row[-1])
            return row

    def get_action(self):
        """Next action tensor [1,7]; after exhaustion hold only gripper command."""
        return torch.from_numpy(self._next_action_row()).unsqueeze(0)

    def publish_observations(self, env):
        """Publish cameras, measured state and chunk execution status."""
        stamp = self.node.get_clock().now().to_msg()

        for name in self._camera_names:
            self._publish_image(
                name,
                env.scene[name].data.output["rgb"][0],
                stamp,
            )

        pose = eef_pose_axis_angle(env)[0].detach().cpu().numpy()
        fingers = gripper_pos(env)[0].detach().cpu().numpy()
        state = encode_state(pose, fingers)

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label="state",
                size=len(state),
                stride=len(state),
            )
        ]
        msg.data = state.tolist()
        self._state_publisher.publish(msg)

        # Episode-local simulation time; returns to zero after env.reset().
        sim_time = float(env.episode_length_buf[0].item()) * env.step_dt

        with self._lock:
            payload = {
                "sim_time": round(sim_time, 6),
                "control_hz": CONTROL_HZ,
                "remaining": self.buffer.remaining,
                "idle": self._last_action_idle,
                "paused": self.is_paused,
                "received_chunks": self._received_chunks,
            }

        status = String()
        status.data = json.dumps(payload)
        self.status.publish(status)

    def _publish_image(self, name, img_tensor, stamp):
        rgb = img_tensor.detach().cpu().numpy()
        if rgb.dtype != np.uint8:
            # Preserve the previous bridge's assumption: floating RGB is [0,1].
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb[..., :3])
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = name
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = rgb.tobytes()
        self._image_publishers[name].publish(msg)
