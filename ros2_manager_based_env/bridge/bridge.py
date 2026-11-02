"""Shared ROS lifecycle and simulation control; Isaac operations run on the main thread.

Subclasses supply observation/action transport and clear_action_buffer().
Runners MUST call process_control_commands() on every iteration, including pause.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import queue
import threading

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future
from simulation_interfaces.msg import Result
from simulation_interfaces.srv import (
    GetSimulationState, ResetSimulation, SetSimulationState, StepSimulation,
)


class BaseRos2Bridge(ABC):
    STOPPED, PLAYING, PAUSED = 0, 1, 2
    MAX_MANUAL_CONTROL_TICKS = 15

    def __init__(self, node_name: str):
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self.node = Node(node_name)
        self._node = self.node  # Compatibility with the legacy bridge.
        self._lock = threading.RLock()
        self._sim_state = self.PLAYING
        self._closed = False
        self._commands = queue.Queue(maxsize=64)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = None
        self._control_group = ReentrantCallbackGroup()
        self._services = []
        for service_type, name, callback in (
            (GetSimulationState, '/get_simulation_state', self._get_state),
            (SetSimulationState, '/set_simulation_state', self._set_state),
            (ResetSimulation, '/reset_simulation', self._reset),
            (StepSimulation, '/step_simulation', self._step),
        ):
            self._services.append(self.node.create_service(
                service_type, name, callback, callback_group=self._control_group,
            ))

    @property
    def simulation_state(self):
        with self._lock:
            return self._sim_state

    @property
    def is_paused(self):
        return self.simulation_state != self.PLAYING

    def start(self):
        """Call after the subclass has finished creating its publishers/subscribers."""
        if self._spin_thread is None:
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

    def _get_state(self, request, response):
        response.state.state = self.simulation_state
        return response

    async def _submit(self, command, value=None):
        future = Future(executor=self._executor)
        with self._lock:
            if self._closed:
                return 'Bridge is closed'
            try:
                self._commands.put_nowait((command, value, future))
            except queue.Full:
                return 'Simulation control queue is full'
        # Yield to the ROS executor: action/status callbacks remain responsive.
        return await future

    @staticmethod
    def _result(response, error):
        response.result.result = (
            Result.RESULT_OK if error is None else Result.RESULT_OPERATION_FAILED
        )
        response.result.error_message = error or ''
        return response

    async def _set_state(self, request, response):
        state = int(request.state.state)
        if state not in (self.STOPPED, self.PLAYING, self.PAUSED):
            return self._result(response, 'Unsupported simulation state')
        return self._result(response, await self._submit('state', state))

    async def _reset(self, request, response):
        # Only a complete reset is implemented. Do not silently ignore scope.
        scope = int(getattr(request, 'scope', 0))
        full = int(getattr(type(request), 'SCOPE_ALL', 255))
        if scope not in (0, full):
            return self._result(response, 'Only a full reset is supported')
        return self._result(response, await self._submit('reset'))

    async def _step(self, request, response):
        return self._result(response, await self._submit('step', int(request.steps)))

    def process_control_commands(self, reset, step, decimation):
        """Main-thread API. reset() resets the env; step() executes ONE control tick.

        StepSimulation counts physics steps. Unsupported fractions of a control
        tick are rejected, never rounded or reported as successfully executed.
        Returns True when a command was processed, so a runner skips its normal
        tick in this iteration (avoids a second step immediately after Play).
        """
        try:
            command, value, future = self._commands.get_nowait()
        except queue.Empty:
            return False
        error = None
        operation_started = False
        try:
            if command == 'reset' or (command == 'state' and value == self.STOPPED):
                with self._lock:
                    self._sim_state = self.STOPPED
                    self.clear_action_buffer()
                operation_started = True
                reset()
            elif command == 'state':
                with self._lock:
                    # Preserve a buffered chunk on Play/Pause; reset/stop clear it.
                    self._sim_state = value
            elif command == 'step':
                if not self.is_paused:
                    raise ValueError('Pause or stop the simulation before stepping')
                decimation = int(decimation)
                if value <= 0 or value % decimation:
                    raise ValueError(f'steps must be a positive multiple of {decimation}')
                ticks = value // decimation
                if ticks > self.MAX_MANUAL_CONTROL_TICKS:
                    raise ValueError('At most 15 control ticks per request')
                operation_started = True
                for _ in range(ticks):
                    step()
                with self._lock:
                    self._sim_state = self.PAUSED
            else:
                raise ValueError(f'Unknown control command: {command}')
        except Exception as exc:
            error = str(exc)
            if operation_started:
                with self._lock:
                    self._sim_state = self.PAUSED
                    self.clear_action_buffer()
            self.node.get_logger().error(f'Simulation control failed: {exc}')
        finally:
            if not future.done():
                future.set_result(error)
            self._executor.wake()
        return True

    @abstractmethod
    def clear_action_buffer(self):
        """Clear environment-specific actions without calling any Isaac API."""

    def close(self):
        with self._lock:
            self._closed = True
            while True:
                try:
                    _, _, future = self._commands.get_nowait()
                except queue.Empty:
                    break
                if not future.done():
                    future.set_result('Bridge is closing')
        self._executor.wake()
        self._executor.shutdown(timeout_sec=2.0)
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        self.node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()

    def shutdown(self):
        self.close()
