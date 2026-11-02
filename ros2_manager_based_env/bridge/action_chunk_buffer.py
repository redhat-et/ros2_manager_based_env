"""Shared action buffers. No ROS, Isaac or PyTorch imports."""
from __future__ import annotations

from abc import ABC, abstractmethod
import threading

import numpy as np


class ChunkBuffer(ABC):
    """Replace the remaining chunk atomically; consume each row once.

    An empty buffer returns None. The environment defines what idle means.
    Implementations must protect shared state and return independent rows.
    """

    @abstractmethod
    def replace(self, chunk: np.ndarray) -> None:
        """Replace the old remainder with a new [H, action_dim] chunk."""

    @abstractmethod
    def pop(self) -> np.ndarray | None:
        """Return the next row, or None when exhausted."""

    @abstractmethod
    def clear(self) -> None:
        """Discard all remaining actions."""

    @property
    @abstractmethod
    def remaining(self) -> int:
        """Number of unread action rows."""


class ActionChunkBuffer(ChunkBuffer):
    """Thread-safe NumPy implementation, configurable for different environments.

    Shape/finiteness checks belong here. ROS layout, units, value ranges,
    clipping and idle commands belong to the environment-specific contract.
    Invalid replace() raises and leaves the previous buffer intact; the caller
    decides whether to clear on a rejected command.
    """

    def __init__(self, action_dim: int, *, max_horizon: int | None = None):
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        if max_horizon is not None and (
            isinstance(max_horizon, bool)
            or not isinstance(max_horizon, int)
            or max_horizon <= 0
        ):
            raise ValueError("max_horizon must be a positive integer or None")
        self._action_dim = action_dim
        self._max_horizon = max_horizon
        self._lock = threading.Lock()
        self._chunk = np.empty((0, action_dim), dtype=np.float32)
        self._index = 0

    def replace(self, chunk: np.ndarray) -> None:
        # Take ownership before validation; callers may reuse their input.
        array = np.array(chunk, dtype=np.float32, copy=True)
        if array.ndim != 2 or array.shape[1] != self._action_dim:
            raise ValueError(f"chunk must have shape [H,{self._action_dim}]")
        if not np.isfinite(array).all():
            raise ValueError("chunk must contain only finite values")
        if self._max_horizon is not None and len(array) > self._max_horizon:
            raise ValueError(f"chunk horizon must not exceed {self._max_horizon}")
        with self._lock:
            self._chunk = array
            self._index = 0

    def pop(self) -> np.ndarray | None:
        with self._lock:
            if self._index >= len(self._chunk):
                return None
            row = self._chunk[self._index].copy()
            self._index += 1
            return row

    def clear(self) -> None:
        with self._lock:
            self._chunk = np.empty((0, self._action_dim), dtype=np.float32)
            self._index = 0

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._chunk) - self._index
