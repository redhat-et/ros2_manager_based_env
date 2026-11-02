# ros2_manager_based_env

Installable collection of ROS 2-enabled **Manager-Based** environments for
Isaac Lab.

## Layout

```text
ros2_manager_based_env/
├── pyproject.toml
├── README.md
├── config/
│   └── extension.toml
├── scripts/
│   └── run_custom_env_3_3_ros2.py
└── ros2_manager_based_env/
    ├── __init__.py
    └── custom_env_3_3/
        ├── __init__.py
        ├── custom_actions_cfg.py
        ├── custom_commands_cfg.py
        ├── custom_curriculum_cfg.py
        ├── custom_event_cfg.py
        ├── custom_observations_cfg.py
        ├── custom_rewards_cfg.py
        ├── custom_scene_cfg.py
        ├── custom_terminations_cfg.py
        ├── custom_velocity_env_cfg.py
        ├── observations.py
        ├── ros2_bridge.py
        ├── rough_env_3_3_cfg.py
        └── run_ros2_rollout.py
```

Each new environment is added as another package next to
`custom_env_3_3`, for example:

```text
ros2_manager_based_env/
├── custom_env_3_3/
├── custom_env_4/
└── pick_and_place_env/
```

Every environment should own its scene/config/MDP modules and launcher-specific
code. Code genuinely shared by several environments can later be placed in a
separate `ros2_manager_based_env/common/` package.

## Install locally

Run from the repository root in the Python environment used by Isaac Lab:

```bash
python -m pip install -e .
```

With Isaac Lab's launcher:

```bash
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e .
```

## Verify the lightweight package import

```bash
python -c "import ros2_manager_based_env; print(ros2_manager_based_env.__version__)"
```

## Run custom_env_3_3

From the repository root:

```bash
/path/to/IsaacLab/isaaclab.sh -p scripts/run_custom_env_3_3_ros2.py \
  --headless --enable_cameras
```

Or use the installed console command:

```bash
ros2-manager-based-env-3-3 --headless --enable_cameras
```

The launch process still requires the ROS 2 Python packages and message types
used by `ros2_bridge.py` to be visible in the same runtime.

## Import environment configuration from another repository

The configuration module should be imported only after `AppLauncher` has
started Isaac Sim:

```python
from ros2_manager_based_env.custom_env_3_3.rough_env_3_3_cfg import (
    G1RoughEnv3_3Cfg,
    G1RoughEnv3_3Cfg_ROS2,
)
```

## Add another environment

1. Create `ros2_manager_based_env/<new_env_name>/`.
2. Add `__init__.py`.
3. Keep imports inside that folder relative, for example
   `from .custom_scene_cfg import SceneCfg`.
4. Add a launcher under `scripts/` or a new `[project.scripts]` entry when the
   new environment needs its own command.
5. Reinstalling is not required in editable mode; the new Python package is
   discovered directly from the repository checkout.
   

# 🐳 Running `ros2_manager_based_env` Environments in an Isaac Lab Docker Container

This guide explains how to connect the custom environments from `ros2_manager_based_env` to an Isaac Lab Docker container and run training inside the container.

## ✅ Prerequisites
### Install: Docker + NVIDIA Container Toolkit. Please follow the instruction by the link below
```bash
https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html
```

---

## 1) Clone the repositories locally

### 1.1 Clone `ros2_manager_based_env`

```bash
git clone git@github.com:redhat-et/ros2_manager_based_env.git
```

### 1.2 Clone Isaac Lab into a different directory

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
```

> Keep `ros2_manager_based_env` and `IsaacLab` in separate folders.

---

## 2) 📀 Mount `ros2_manager_based_env` into the Isaac Lab Docker container

To make the `ros2_manager_based_env` repository available **inside the container**, create a Docker Compose patch file.

### 2.1 Create a file

Create this file:

`IsaacLab/docker/docker-compose.redhat.patch.yaml`

with the following content:

```yaml
services:
  isaac-lab-base:
    volumes:
      - YourPathTo_ros2_manager_based_env:/workspace/ros2_manager_based_env:rw
```

### Example

```yaml
services:
  isaac-lab-base:
    volumes:
      - /home/username/Documents/ros2_manager_based_env:/workspace/ros2_manager_based_env:rw
```

> Replace the path on the left with the **absolute path** to your local `vkus_grab_ext` repository.

---

## 3) 📦 Start the Isaac Lab container with the patch file

Go to the `IsaacLab` directory and run:

```bash
./docker/container.py start ros2 --file docker-compose.redhat.patch.yaml
```

Then enter the container:

```bash
./docker/container.py enter ros2
```

---


## 4) 🧱 Install `ros2_manager_based_env` inside the container

Inside the container, go to the mounted repository:

```bash
cd /workspace/ros2_manager_based_env
```

Install it in editable mode:

```bash
pip install -e .
```


## 5) 🚀 Test training run

Inside the container, go to the IsaacLab directory:

```bash
cd /workspace/isaaclab
```

Run a test training job:
# custom_env_3_3 (VLA Pi05 finetuned with Lerobot checkpoint)
```bash
./isaaclab.sh -p -m ros2_manager_based_env.custom_env_3_3.run_ros2_rollout \
  --rendering_mode performance \
  --enable_cameras
```

# droid_pi05
```bash
./isaaclab.sh -p -m ros2_manager_based_env.droid_pi05.run_ros2_rollout \
  --rendering_mode performance \
  --enable_cameras
```

---
   
