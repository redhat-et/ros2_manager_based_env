# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from .custom_velocity_env_cfg import CustomLocomotionVelocityRoughEnvCfg


@configclass
class G1RoughEnv3_3Cfg(CustomLocomotionVelocityRoughEnvCfg):
    """Franka Panda manipulation env for ROS2-driven VLA rollout."""

    # Used by observations.gripper_pos().
    gripper_joint_names = (
        "panda_finger_joint1",
        "panda_finger_joint2",
    )

    def __post_init__(self):
        super().__post_init__()

        robot_cfg = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        robot_cfg.actuators["panda_shoulder"].stiffness = 0.0
        robot_cfg.actuators["panda_shoulder"].damping = 0.0
        robot_cfg.actuators["panda_forearm"].stiffness = 0.0
        robot_cfg.actuators["panda_forearm"].damping = 0.0
        
        robot_cfg.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": 0.0,
            "panda_joint3": 0.0,
            "panda_joint4": -1.5,
            "panda_joint5": 0.0,
            "panda_joint6": 1.5,
            "panda_joint7": 0.0,
            "panda_finger_joint.*": 0.04,
        }

        self.scene.robot = robot_cfg
        
        
        # Set cameras
        # Set wrist camera
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.15,
            height=256,
            width=256,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(0.11062, 0.6984, 0.6984, 0.11062), convention="opengl"
            ),
        )

        # Set table view camera
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.15,
            height=256,
            width=256,
            data_types=["rgb"],
            #colorize_semantic_segmentation=True,
            #semantic_segmentation_mapping=SEMANTIC_MAPPING,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.5, 0.0, 0.4), rot=(0.55721, 0.43534, 0.43534, 0.55721), convention="opengl"
            ),
        )        
        
        

@configclass
class G1RoughEnv3_3Cfg_ROS2(G1RoughEnv3_3Cfg):
    """Single-environment configuration for live ROS2 testing."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
