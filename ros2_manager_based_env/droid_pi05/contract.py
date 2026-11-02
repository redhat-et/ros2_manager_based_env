"""DROID wire/control contract. No Isaac, Torch or ROS imports."""
from __future__ import annotations

import threading
import numpy as np

CONTROL_HZ = 15
PHYSICS_HZ = 150
DECIMATION = PHYSICS_HZ // CONTROL_HZ
ACTION_DIM = 8
MODEL_HORIZON = 15
MAX_JOINT_DELTA = 0.2  # rad per 15 Hz command; DROID RobotIKSolver, not rad/s.
ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_NAMES = ("panda_finger_joint1", "panda_finger_joint2")
CAMERA_KEYS = {
    "exterior_image_1_left": "observation/exterior_image_1_left",
    "wrist_image_left": "observation/wrist_image_left",
}


def closure_from_fingers(fingers, finger_open=0.04):
    """Measured Panda finger travel -> DROID closure: 0=open, 1=closed."""
    return np.clip(1.0 - np.asarray(fingers).mean(axis=-1) / finger_open, 0.0, 1.0)


def joint_targets(positions, command, lower, upper):
    """DROID clips normalized velocities, then targets current q + 0.2*u."""
    return np.clip(np.asarray(positions) + MAX_JOINT_DELTA *
                   np.clip(np.asarray(command), -1.0, 1.0), lower, upper)


def decode_chunk(data, shape=(), offset=0):
    """Validate Float32MultiArray payload; reject ambiguous/truncated chunks."""
    values = np.asarray(data, dtype=np.float32)
    if offset != 0:
        raise ValueError("data_offset must be zero")
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("payload must be flat and finite")
    if values.size == 0:
        return np.empty((0, ACTION_DIM), dtype=np.float32)
    if values.size % ACTION_DIM:
        raise ValueError("payload length must be a multiple of 8")
    h = values.size // ACTION_DIM
    if not 1 <= h <= MODEL_HORIZON:
        raise ValueError("chunk horizon must be between 1 and 15")
    if shape and tuple(shape) not in ((h,), (h, ACTION_DIM)):
        raise ValueError(f"layout {tuple(shape)} disagrees with payload {(h, ACTION_DIM)}")
    return values.reshape(h, ACTION_DIM).copy()


class ChunkBuffer:
    """Latest chunk wins. One pop per environment tick; never repeat last row."""

    def __init__(self):
        self._lock = threading.Lock()
        self._chunk = np.empty((0, ACTION_DIM), dtype=np.float32)
        self._index = 0

    def replace(self, chunk):
        array = np.asarray(chunk, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != ACTION_DIM:
            raise ValueError("chunk must have shape [H,8]")
        checked = decode_chunk(array.reshape(-1))
        with self._lock:
            self._chunk, self._index = checked, 0

    def clear(self):
        self.replace(np.empty((0, ACTION_DIM), dtype=np.float32))

    def pop(self):
        with self._lock:
            if self._index >= len(self._chunk):
                return None
            row = self._chunk[self._index].copy()
            self._index += 1
            return row

    @property
    def remaining(self):
        with self._lock:
            return len(self._chunk) - self._index
