# ros2_bridge.py
"""ROS2 bridge between this IsaacLab env and an external VLA policy.

Design (deliberately lives OUTSIDE the env's manager pipeline, i.e. not an
ObservationTerm/ActionTerm): the env's own ActionsCfg (arm_action +
gripper_action, see custom_actions_cfg.py) already knows how to turn a
7-dim [pos_delta(3), rot_delta_axisangle(3), gripper(1)] tensor into robot
joint efforts. This bridge's only job is supplying WHERE that 7-dim tensor
comes from (a ROS2 topic, instead of an in-process policy call or an
internally-generated command) and WHERE the resulting observations go (out
to ROS2 topics for the VLA to consume), each sim step.

The VLA publishes a CHUNK of actions per inference call (e.g. 8
consecutive [pos_delta, rot_delta, gripper] steps from one pi0.5 forward
pass), not one action per message -- get_action() walks through the
current chunk one step at a time across successive calls, only asking
for the next chunk once the current one runs out (or a new one arrives
early and replaces it -- see get_action()'s docstring for the exact
policy).

Topics/message types below are defaults, not a fixed contract -- change
them to match whatever your actual VLA-side ROS2 interface expects. Using
std_msgs/Float32MultiArray for state+action (no custom .msg package
needed) rather than a strongly-typed custom message, for simplicity. The
action topic expects a flattened (chunk_len, action_dim) array; set
msg.layout.dim[0].size = chunk_len on the publishing side if possible
(the bridge falls back to inferring chunk_len from len(data) /
action_dim if layout.dim isn't populated).

Usage (see run_ros2_rollout.py for the full loop):

    bridge = Ros2VlaBridge(camera_names=["wrist_cam", "table_cam"], env=env)
    ...
    while True:
        action = bridge.get_action()          # next step of the current VLA action chunk
        obs, ... = env.step(action)
        bridge.publish_observations(env)      # reads env.scene directly, not obs
"""

from __future__ import annotations

import threading

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from simulation_interfaces.srv import (
    GetSimulationState,
    ResetSimulation,
    SetSimulationState,
    StepSimulation,
)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from .observations import eef_pose_axis_angle, gripper_pos


class Ros2VlaBridge:
    """Publishes obs / subscribes to actions for one VLA-controlled env.

    Assumes num_envs == 1 (single live robot) -- this is a rollout/
    deployment bridge, not a training-time vectorized wrapper.

    Implements standard simulation_interfaces services for sim control:
    - /reset_simulation: Resets environment including action manager buffers
    - /set_simulation_state: Sets sim state (STOPPED/PLAYING/PAUSED)
    - /step_simulation: Steps simulation by N steps
    - /get_simulation_state: Returns current sim state
    """

    def __init__(
        self,
        camera_names: list[str],
        env,  # IsaacLab ManagerBasedEnv instance
        state_topic: str = "/vla/obs/state",
        action_topic: str = "/vla/action",
        camera_topic_prefix: str = "/vla/obs/",
        node_name: str = "isaaclab_vla_bridge",
        action_dim: int = 7,
    ):
        if not rclpy.ok():
            rclpy.init()

        self._camera_names = camera_names
        self._action_dim = action_dim
        self._env = env  # Store env reference for reset/step services

        self._node = Node(node_name)

        # Best-effort/keep-last QoS: for a real-time control loop we care
        # about the latest sample, not guaranteed delivery of every one.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._image_publishers = {
            name: self._node.create_publisher(Image, f"{camera_topic_prefix}{name}", qos)
            for name in camera_names
        }
        self._state_publisher = self._node.create_publisher(Float32MultiArray, state_topic, qos)

        # Action CHUNK buffer: the VLA publishes a (chunk_len, action_dim)
        # array per inference call (e.g. 8 consecutive actions from a
        # single pi0.5 forward pass), not one action per message.
        # get_action() walks through the current chunk one step per call;
        # a new message replaces the chunk and resets the walk-through
        # index, whether or not the previous chunk was fully consumed.
        self._action_lock = threading.Lock()
        self._current_chunk = torch.zeros(1, action_dim)  # (chunk_len, action_dim)
        self._chunk_step_index = 0
        self._has_received_chunk = False

        action_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._action_subscriber = self._node.create_subscription(
            Float32MultiArray, action_topic, self._on_action_chunk_msg, action_qos
        )

        # Standard simulation_interfaces service servers
        self._reset_service = self._node.create_service(
            ResetSimulation, "/reset_simulation", self._handle_reset_simulation
        )
        self._set_state_service = self._node.create_service(
            SetSimulationState, "/set_simulation_state", self._handle_set_simulation_state
        )
        self._step_service = self._node.create_service(
            StepSimulation, "/step_simulation", self._handle_step_simulation
        )
        self._get_state_service = self._node.create_service(
            GetSimulationState, "/get_simulation_state", self._handle_get_simulation_state
        )

        # Simulation state tracking
        self._sim_state_lock = threading.Lock()
        self._sim_state = 1  # STATE_PLAYING
        self._paused = False

        # Spin in a background thread so callbacks fire concurrently with
        # the sim loop, instead of blocking it.
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        rclpy.spin(self._node)

    def _handle_reset_simulation(
        self, request: ResetSimulation.Request, response: ResetSimulation.Response
    ):
        """Handle /reset_simulation service: fully resets env including action manager."""
        try:
            self._node.get_logger().info("ResetSimulation service called - resetting environment")

            # Reset the environment (clears action manager buffers!)
            self._env.reset()

            # Clear action buffer in bridge
            with self._action_lock:
                self._current_chunk = torch.zeros(1, self._action_dim)
                self._chunk_step_index = 0
                self._has_received_chunk = False

            # Transition to STOPPED so a subsequent Play can re-trigger PLAYING
            with self._sim_state_lock:
                self._sim_state = 0  # STATE_STOPPED
                self._paused = True

            self._node.get_logger().info("ResetSimulation completed successfully (state → STOPPED)")
        except Exception as e:
            self._node.get_logger().error(f"ResetSimulation failed: {e}")
            response.result.result = 1
            response.result.error_message = str(e)

        return response

    def _handle_set_simulation_state(
        self, request: SetSimulationState.Request, response: SetSimulationState.Response
    ):
        """Handle /set_simulation_state service: sets sim state (STOPPED/PLAYING/PAUSED)."""
        try:
            state_names = {0: "STOPPED", 1: "PLAYING", 2: "PAUSED", 3: "QUITTING"}
            state_name = state_names.get(request.state.state, f"UNKNOWN({request.state.state})")

            self._node.get_logger().info(f"SetSimulationState service called: {state_name}")

            with self._sim_state_lock:
                self._sim_state = request.state.state
                self._paused = request.state.state in (0, 2)  # STOPPED or PAUSED

                # STOPPED (0) = pause + reset
                if request.state.state == 0:
                    self._env.reset()
                    with self._action_lock:
                        self._current_chunk = torch.zeros(1, self._action_dim)
                        self._chunk_step_index = 0
                        self._has_received_chunk = False

            self._node.get_logger().info(f"SetSimulationState to {state_name} completed")
        except Exception as e:
            self._node.get_logger().error(f"SetSimulationState failed: {e}")
            response.result.result = 1
            response.result.error_message = str(e)

        return response

    def _handle_step_simulation(
        self, request: StepSimulation.Request, response: StepSimulation.Response
    ):
        """Handle /step_simulation service: step simulation by N steps.

        Note: This is a simplified implementation. For full stepping support,
        the calling code (run_ros2_rollout.py) would need to respect the
        paused state and only step when explicitly requested via this service.
        """
        try:
            steps = request.steps
            self._node.get_logger().info(f"StepSimulation service called: {steps} steps")

            # For manager-based envs, stepping is done by the rollout loop
            # This service acknowledges the request but doesn't directly step
            # (the rollout loop would need to check _paused and step accordingly)
            self._node.get_logger().info(f"StepSimulation acknowledged {steps} steps")
        except Exception as e:
            self._node.get_logger().error(f"StepSimulation failed: {e}")
            response.result.result = 1
            response.result.error_message = str(e)

        return response

    def _handle_get_simulation_state(
        self, request: GetSimulationState.Request, response: GetSimulationState.Response
    ):
        """Handle /get_simulation_state service: return current sim state."""
        with self._sim_state_lock:
            response.state.state = self._sim_state

        state_names = {0: "STOPPED", 1: "PLAYING", 2: "PAUSED"}
        self._node.get_logger().debug(
            f"GetSimulationState returned: {state_names.get(response.state.state, response.state.state)}"
        )
        return response

    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused (for rollout loop to respect)."""
        with self._sim_state_lock:
            return self._paused

    def _on_action_chunk_msg(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)

        # Prefer the publisher-declared shape (msg.layout.dim); fall back
        # to inferring chunk_len from total length if layout wasn't set
        # (easy to forget on the publishing side, so don't hard-require it).
        if len(msg.layout.dim) >= 1 and msg.layout.dim[0].size > 0:
            chunk_len = msg.layout.dim[0].size
        elif len(data) % self._action_dim == 0:
            chunk_len = len(data) // self._action_dim
        else:
            self._node.get_logger().warn(
                f"Received action chunk of length {len(data)}, not a multiple of "
                f"action_dim={self._action_dim} and no usable layout.dim -- ignoring"
            )
            return

        if chunk_len * self._action_dim != len(data):
            self._node.get_logger().warn(
                f"Declared chunk_len={chunk_len} * action_dim={self._action_dim} "
                f"!= data length {len(data)} -- ignoring"
            )
            return

        chunk = torch.tensor(data, dtype=torch.float32).reshape(chunk_len, self._action_dim)

        with self._action_lock:
            self._current_chunk = chunk
            self._chunk_step_index = 0
            self._has_received_chunk = True

    def get_action(self) -> torch.Tensor:
        """Returns the NEXT action to apply, shape (1, action_dim), walking
        one step further into the current chunk each call.

        - Before any chunk has been received: returns zeros (NOT a guess
          at a "safe" action -- review whether zeros is actually safe for
          your gripper_action's open/close convention).
        - Once the current chunk is fully consumed (chunk_step_index
          reaches its end) and no new chunk has arrived yet: holds the
          chunk's LAST action rather than erroring or reusing an
          arbitrary one -- if your VLA-side publishing cadence can't
          reliably keep up with chunk consumption, actions will visibly
          stall here instead of failing silently.
        - A new chunk arriving mid-consumption immediately replaces the
          old one and restarts from its first step -- the remainder of
          the old chunk is discarded, not queued after the new one
          (this is the standard receding-horizon-friendly behavior; if
          you instead want to always fully drain each chunk before
          accepting a new one, that's a different policy and would need
          an explicit "chunk done" gate here).
        """
        with self._action_lock:
            if not self._has_received_chunk:
                self._node.get_logger().warning(
                    "No action chunk received yet on the action topic -- publishing zeros", once=True
                )
                return self._current_chunk.clone()

            chunk_len = self._current_chunk.shape[0]
            if self._chunk_step_index >= chunk_len:
                self._node.get_logger().warning(
                    "Action chunk exhausted and no new chunk has arrived -- "
                    "sending zero arm delta while preserving the last gripper command",
                    throttle_duration_sec=1.0,
                )
                safe_action = torch.zeros(
                    (1, self._action_dim),
                    dtype=self._current_chunk.dtype,
                )
                # For the current 7D contract, the final component is gripper.
                safe_action[0, -1] = self._current_chunk[-1, -1]
                return safe_action

            step = self._chunk_step_index
            self._chunk_step_index += 1
            return self._current_chunk[step].unsqueeze(0).clone()

    def publish_observations(self, env) -> None:
        """Publishes camera images + proprioception, read directly from
        `env.scene` -- NOT from env.step()'s returned obs dict. This
        deliberately bypasses the ObservationManager/ObservationsCfg
        entirely: whether ObservationsCfg.PolicyCfg declares these terms,
        and whether concatenate_terms is True or False there, has no
        effect on what gets published here. Confirmed API:
        env.scene[name].data.output["rgb"], shape (num_envs, H, W, 3).
        """
        stamp = self._node.get_clock().now().to_msg()

        for name in self._camera_names:
            img_tensor = env.scene[name].data.output["rgb"][0]  # first (only) env
            self._publish_image(name, img_tensor, stamp)

        eef_pose = eef_pose_axis_angle(env)[0]  # (6,)
        gripper = gripper_pos(env)[0]  # (2,)
        state = torch.cat([eef_pose, gripper], dim=-1)  # (8,)

        msg = Float32MultiArray()
        msg.layout.dim = [MultiArrayDimension(label="state", size=state.shape[-1], stride=state.shape[-1])]
        msg.data = state.detach().cpu().numpy().astype(np.float32).tolist()
        self._state_publisher.publish(msg)

    def _publish_image(self, name: str, img_tensor: torch.Tensor, stamp) -> None:
        img_np = img_tensor.detach().cpu().numpy()
        if img_np.dtype != np.uint8:
            # Assumes [0, 1] float -- adjust if your camera output range differs.
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = name
        msg.height, msg.width = img_np.shape[0], img_np.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = img_np.tobytes()
        self._image_publishers[name].publish(msg)

    def shutdown(self):
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._spin_thread.join(timeout=2.0)
