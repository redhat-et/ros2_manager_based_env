"""custom_env_3_3 wire/control rules. No ROS, Isaac or Torch imports.

The wire state is eef pose + mirrored fingers, NOT seven joint positions.
"""
from __future__ import annotations

import numpy as np

# Shared simulation profile for both launchers, selected by the project.
PHYSICS_HZ = 150
CONTROL_HZ = 10 #(Freq of dataset used for finetune the checkpoint)
DECIMATION = PHYSICS_HZ // CONTROL_HZ
RENDER_INTERVAL = 1
CAMERA_UPDATE_PERIOD = 1.0 / CONTROL_HZ
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256

ACTION_DIM = 7
STATE_DIM = 8
# Target LeRobot policy; verified config revision 9f933e0.
POLICY_REPO_ID = "execbat/pi05-robot-finetuned"
MODEL_HORIZON = 50
MAX_CHUNK_HORIZON = MODEL_HORIZON
POSITION_SCALE = 1.0
ORIENTATION_SCALE = 1.0
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0
INITIAL_GRIPPER_COMMAND = 0.0

ASSET_NAME = 'robot'
EEF_BODY_NAME = 'panda_hand'
ARM_NAMES = tuple(f'panda_joint{i}' for i in range(1, 8))
FINGER_NAMES = ('panda_finger_joint1', 'panda_finger_joint2')
CAMERA_KEYS = {
    'wrist_cam': 'observation.images.image2',
    'table_cam': 'observation.images.image',
}
STATE_TOPIC = '/vla/obs/state'
ACTION_TOPIC = '/vla/action'
CAMERA_TOPIC_PREFIX = '/vla/obs/'
NODE_NAME = 'isaaclab_vla_bridge'
ACTION_COMPONENTS = ('dx', 'dy', 'dz', 'dax', 'day', 'daz', 'gripper_command')
STATE_COMPONENTS = ('x', 'y', 'z', 'ax', 'ay', 'az', 'finger1', 'mirrored_finger2')


def decode_chunk(data, shape=(), offset=0):
    """Decode Float32MultiArray: flat payload, no layout or [H] / [H,7].

    Empty payload cancels the chunk; no value scaling or clipping is applied.
    Accepts partial chunks up to the selected policy horizon (50).
    Invalid layout/offset is rejected.
    """
    values = np.asarray(data, dtype=np.float32)
    if offset != 0:
        raise ValueError('data_offset must be zero')
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('payload must be flat and finite')
    if values.size % ACTION_DIM:
        raise ValueError(f'payload length must be a multiple of {ACTION_DIM}')
    horizon = values.size // ACTION_DIM
    if shape and tuple(shape) not in ((horizon,), (horizon, ACTION_DIM)):
        raise ValueError(f'layout {tuple(shape)} disagrees with {(horizon, ACTION_DIM)}')
    if MAX_CHUNK_HORIZON is not None and horizon > MAX_CHUNK_HORIZON:
        raise ValueError(f'chunk horizon must not exceed {MAX_CHUNK_HORIZON}')
    return values.reshape(horizon, ACTION_DIM).copy()


def idle_action(gripper_command=INITIAL_GRIPPER_COMMAND):
    """Zero Cartesian arm delta, preserving the last executed gripper command.

    Initial/reset gripper command remains zero, as in the previous bridge.
    """
    if not np.isfinite(gripper_command):
        raise ValueError('gripper command must be finite')
    row = np.zeros(ACTION_DIM, dtype=np.float32)
    row[-1] = gripper_command
    return row


def encode_state(eef_pose, mirrored_finger_positions):
    """Pack measured root-frame [xyz,axis-angle] and already-mirrored fingers.

    gripper_pos(env) performs mirroring; do not negate its second value again.
    """
    pose = np.asarray(eef_pose, dtype=np.float32)
    fingers = np.asarray(mirrored_finger_positions, dtype=np.float32)
    if pose.shape != (6,) or fingers.shape != (2,):
        raise ValueError('expected eef pose [6] and mirrored finger positions [2]')
    state = np.concatenate((pose, fingers))
    if not np.isfinite(state).all():
        raise ValueError('state must be finite')
    return state
