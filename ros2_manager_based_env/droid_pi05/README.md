# DROID / pi0.5: independent Isaac Lab environment

This directory provides a standalone, inference-only **ManagerBasedRLEnv** for
testing the official OpenPI `pi05_droid` checkpoint with a simulated Franka
Panda and two RGB cameras. It does not train or fine-tune a model. It does not
inherit from, import, or change `custom_env_3_3`.

The model server and the simulator are separate processes. The environment has
no dependency on a model server to start: inspect observations, send a short
action chunk, or test idle behavior entirely from a terminal.

## Scope and embodiment

Target checkpoint:

```text
OpenPI config: pi05_droid
Checkpoint: gs://openpi-assets/checkpoints/pi05_droid
Normalization: the checkpoint's own assets/droid/norm_stats.json
```

This is a generalist model subsequently fine-tuned on DROID, not a model trained
only on this scene. The DROID embodiment is a **7-DoF Franka Panda with a Robotiq
parallel gripper**, originally recorded with two external ZED cameras and one
wrist ZED Mini. This policy's inference path consumes **one external RGB view
and one wrist RGB view**, not three stereo streams, and does not require depth.

This scene uses Isaac Lab's **Panda hand**, not a Robotiq asset. Finger travel is
mapped to DROID's normalized closure convention. This is a useful interface
approximation, **not an exact DROID digital twin**: gripper geometry, TCP,
contacts, camera calibration, controller response and sim-to-real appearance
differ. Loading the model successfully does not establish task success.
Do not interpret this setup as an already fine-tuned policy for another robot.

The checkpoint format/contract here is the **original OpenPI** policy. A
LeRobot export, base pi05, pi05-DROID joint-position variant, LIBERO model, or
DreamZero-DROID must not be substituted without rechecking its contract.

## Timing and chunk execution

| Component | Rate / size |
|---|---|
| Physics | 240 Hz (`dt=1/240 s`) |
| Control / `env.step` | **15 Hz** (`dt=1/15 s`) |
| Physics substeps per action | **16** |
| Camera update and render | 15 Hz in simulation time |
| ROS observations | Once per control tick; repeat last snapshot while paused |
| Official `pi05_droid` prediction | **`actions[15, 8]`**, 15 future control rows |
| Full prediction duration | 1.0 simulation second |
| Optional client receding horizon | First **8** rows, about 0.533 simulation seconds |

The current official OpenPI `pi05_droid` training config declares
`action_horizon=15`; `DroidOutputs` returns the first eight action dimensions.
The horizon is not the 50-row horizon of generic pi05 base. Check the actual
server config if using a different OpenPI revision; the included policy client
rejects responses other than `[15,8]` instead of guessing.

Rules enforced by `ChunkBuffer` and the runner:

1. At each **15 Hz environment tick**, pop exactly one row. The ActionManager
   processes it once, then holds its drive targets for 16 physics substeps.
2. Accept chunks `[H,8]` for `1 <= H <= 15`. A new chunk replaces any unconsumed
   rows of the old chunk. There is no FIFO backlog of old predictions.
3. **No chunk, empty chunk, or exhausted chunk:** stop issuing arm motion,
   cancel the previous target and hold the measured arm and finger positions
   captured on entry to idle. The last velocity row is never repeated.
4. A zero-sized ROS payload explicitly cancels the current chunk. A malformed
   or non-finite payload also clears the buffer and logs an error.
5. Reset clears the queue and returns the robot to its configured initial pose.
   Pause stops physics and clears actions; resume starts with an empty queue.

There is no magic zero command for the gripper: a valid all-zero policy row
means zero arm velocity **and OPEN gripper**. Empty-buffer handling is a
separate internal idle path (`gripper=-1`), not an all-zero policy row. Policy
and ROS commands are clipped to gripper `[0,1]` and cannot select that sentinel.
On idle entry, joint/finger velocities are zeroed once and position targets are
held; contact forces and numerical settling may still perturb a physics body.

The runner paces to at most 15 control ticks per wall-clock second. Slow rendering
can make wall time slower than simulation time; it never skips action rows or
repeats old rows to catch up. `--fast` disables this pacing only for diagnostics;
do not use it with the provided synchronous policy client.

## Observation format (before model normalization)

The ObservationManager exposes a dictionary under `obs["policy"]`, with a
leading environment dimension of 1. ROS publication uses that exact dictionary.

| Manager key | Shape / type | Meaning |
|---|---|---|
| `exterior_image_1_left` | `[1,224,224,3] uint8` | Fixed external camera, RGB `[0,255]` |
| `wrist_image_left` | `[1,224,224,3] uint8` | Camera rigidly attached to `panda_hand`, RGB |
| `joint_position` | `[1,7] float` | **Measured absolute** arm joint angles, radians, joint1 through joint7 |
| `gripper_position` | `[1,1] float` | **Measured absolute closure**, `0=open`, `1=closed` |

Panda closure is `clip(1 - mean(finger_positions)/0.04, 0, 1)`.
This is not TCP XYZ/orientation and not joint offsets from the default pose.

The policy client constructs this native OpenPI request:

```python
{
    "observation/exterior_image_1_left": external_rgb,  # uint8[224,224,3]
    "observation/wrist_image_left": wrist_rgb,         # uint8[224,224,3]
    "observation/joint_position": q,                 # float32[7], radians
    "observation/gripper_position": closure,         # float32[1], 0=open
    "prompt": "pick up the red cube",                # task string
}
```

The OpenPI server owns padding, tokenization, normalization, image masks and
action unnormalization. Do not normalize twice in the simulator. Its DROID
adapter masks the unused third model camera internally; no dummy third ROS
camera is published.

### Cameras and calibration

Names match the native DROID input keys above. The scene uses square 224x224
pinhole RGB cameras (18 mm focal length, 36 mm aperture, about 90-degree FOV).
These intrinsics and the example camera poses are **simulation choices**, not
fixed DROID dataset calibration. DROID includes varied external viewpoints;
there is no single mandatory exterior camera pose to copy.

Review both camera images with `inspect --dump-images` before running a policy.
Tune poses/FOV in `custom_scene_cfg.py` for visibility of the workspace and gripper.
The wrist camera can be occluded by the robot depending on the loaded asset;
the poses have not been visually validated in Isaac Sim here. CameraInfo
publishes the actual simulated K matrix; images are RGB, not BGR, with no depth.

## Action format and controller semantics

Model reply: `{"actions": array[15,8]}`. Each row:

```text
[u_joint1, u_joint2, u_joint3, u_joint4, u_joint5, u_joint6, u_joint7, closure]
```

* **Arm:** normalized **joint-velocity commands**, clipped to `[-1,1]`.
  They are not absolute joint angles, Cartesian pose deltas, or raw rad/s.
* **Gripper:** **absolute** closure `[0,1]`: `0=open`, `1=closed`.
  By default, commands are binarized at `>0.5`, matching OpenPI's DROID client.
  Measured observation closure remains continuous.

The custom `DroidAction` ActionTerm follows the DROID controller conversion:

```text
u = clip(action[:7], -1, 1)
q_target = measured_q_now + 0.2 * u
finger_target_m = 0.04 * (1 - binary_closure)
```

Arm targets are also clipped inside the Panda's joint position limits (0.01 rad
margin). `0.2` is the upstream DROID maximum joint delta **per control command**,
not a per-second velocity. For example, `u=0.02` requests a target increment of
0.004 rad for that tick. The equivalent nominal rate is `3*u rad/s` at 15 Hz,
but actual motion depends on the drive and contact dynamics.

Although the policy action space is velocity, DROID itself converts that command
to a measured-state-relative **position target** for its low-level controller.
This environment follows that design using Panda PD position drives. It does
not use the Cartesian OSC from the previous environment, and does not add
unnormalized velocities directly to positions. Targets are recomputed from the
measured state, not accumulated indefinitely from an old desired pose.

Arm gravity is disabled to approximate a gravity-compensated controller; objects
retain gravity. PD stiffness/damping are simulation approximations, not a copy
of the hardware controller. Empty-buffer position hold is deliberately separate
from the active velocity-command conversion.

## ROS 2 interface

Default prefix `/droid`; change it with `--ros-prefix` on both runner and client.

| Topic | Type | Payload |
|---|---|---|
| `/droid/obs/exterior_image_1_left` | `sensor_msgs/msg/Image` | `rgb8`, 224x224 |
| `/droid/obs/wrist_image_left` | `sensor_msgs/msg/Image` | `rgb8`, 224x224 |
| `<image_topic>/camera_info` | `sensor_msgs/msg/CameraInfo` | Actual simulated intrinsics |
| `/droid/obs/state` | `std_msgs/msg/Float32MultiArray` | `[q1..q7,closure]`, length 8 |
| `/droid/action_chunk` | `std_msgs/msg/Float32MultiArray` | Flattened row-major `[H,8]`; empty data cancels |
| `/droid/status` | `std_msgs/msg/String` | JSON: sim_time, control_hz, remaining, idle, paused, received_chunks |

Use layout `[horizon: H, action: 8]`, `data_offset=0`. Missing layout is accepted
if data length unambiguously identifies `[H,8]`. Maximum payload is 15x8.
Images/state use best-effort depth-1 sensor QoS. Actions use reliable depth-1.
Image headers use the node ROS clock; simulation time is explicit in status.
There is no global `/clock` publisher. Repeated paused observations are stale
physics snapshots, not new camera samples. Image/state topics have separate
delivery, so the simple client checks freshness but is not an exact-time
synchronizer.

Services use `std_srvs/srv/Trigger`:

* `/droid/pause`, `/droid/stop`: pause physics and clear actions.
* `/droid/resume`: resume with an empty queue.
* `/droid/reset`: queue reset on simulator main thread and remain paused.
* `/droid/step`: request one **control** step while paused (16 physics substeps).

Service replies acknowledge queued requests; status reflects main-thread
execution. These are local test services, not `simulation_interfaces` replacements.
Do not point the old Playground adapter at this environment unchanged: it must
preserve the DROID 8D action rows instead of converting them to 7D Cartesian
actions or treating gripper closure as meters.

## Launch and terminal tests

All installation, GUI/headless/container launch instructions, ROS terminal
checks and policy-server connection commands are maintained in the repository
[main README](../../README.md#run-droid_pi05). Start there.

## Tests and validation status

Tests that need only Python and NumPy:

```bash
python -m unittest ros2_manager_based_env.droid_pi05.test_contract -v
```

All 10 contract tests passed during authoring. They cover empty/exhausted chunks, replacement, ordered single-use rows, invalid
dimensions/NaN, DROID velocity scaling, gripper closure and rate consistency.
Python syntax can be checked without loading Isaac Sim.
**Isaac Sim, rclpy and a GPU runtime were not available in the authoring
environment. No simulator startup, camera image quality, ROS end-to-end or
closed-loop task success has been claimed.** Run the terminal tests above
before connecting the policy, and inspect camera views and joint motion.

## Source of the contract

Reviewed upstream source on 2026-11-02:

* [OpenPI config.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py): `pi05_droid`, 15-row horizon, DROID assets.
* [OpenPI droid_policy.py](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/droid_policy.py): input keys, joint+gripper state, image mask, 8 output dimensions.
* [OpenPI DROID client](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py): 15 Hz, joint-velocity mode, position gripper, 8-row receding horizon, gripper binarization. Its hard-coded historical shape assertion may differ from current config; this environment follows the named config above.
* [DROID robot_env.py](https://github.com/droid-dataset/droid/blob/main/droid/robot_env.py): control_hz and normalized command range.
* [DROID robot.py](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py): relative joint target and normalized absolute gripper closure.
* [DROID robot_ik_solver.py](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py): `max_joint_delta=0.2`, velocity-to-delta conversion.
* [DROID platform](https://droid-dataset.github.io/): embodiment and collection cameras.
* [Isaac Lab v2.3.0](https://github.com/isaac-sim/IsaacLab/tree/v2.3.0): manager API and Panda asset.

These URLs track upstream revisions; pin your OpenPI and Isaac Lab versions for
reproducible experiments. Recheck the contract if changing checkpoints.

## File naming

The environment follows the repository conventions:

* `custom_scene_cfg.py`: robot, table, object and two cameras.
* `custom_actions_cfg.py`: DROID velocity-command ActionTerm and ActionsCfg.
* `observations.py`: raw joint-angle and gripper-closure functions.
* `custom_observations_cfg.py`: ObservationManager configuration.
* `custom_velocity_env_cfg.py`: standalone ManagerBasedRLEnv configuration and rates.
* `run_ros2_rollout.py`: AppLauncher and the control loop.
* `ros2_bridge.py`: ROS transport and queued service requests.
* `contract.py`, `test_contract.py`: chunk contract and its tests.
* `ros_client.py`: optional terminal diagnostics and OpenPI client.

Unlike the RL example, this inference-only ManagerBasedRLEnv has no reward,
command, curriculum or termination managers. Empty placeholder configuration
files are not required. No file imports or inherits from `custom_env_3_3`.


## Launch lifecycle

The launcher follows `custom_env_3_3/run_ros2_rollout.py`: start AppLauncher,
import runtime modules, create ManagerBasedRLEnv and the ROS bridge, reset,
warm up two steps, re-author the wrist camera USD pose, then run the rollout.
The independent DROID scene, 8D actions, 15 Hz control, camera observations
and `/droid` ROS interface are retained. Reset events restore the default scene.
There are no automatic episode termination terms or reward terms.
