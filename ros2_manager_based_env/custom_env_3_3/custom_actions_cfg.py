# custom_actions_cfg.py

from __future__ import annotations

from .contract import POSITION_SCALE, ORIENTATION_SCALE, FINGER_OPEN, FINGER_CLOSED
from isaaclab.utils import configclass
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    OperationalSpaceControllerActionCfg,
    BinaryJointPositionActionCfg,
)


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    7-dim action, matching lerobot/libero: [0:3] position delta, [3:6]
    orientation delta (axis-angle, radians), [6] gripper. arm_action
    consumes the first 6 (pose_rel), gripper_action consumes the 7th --
    ActionManager concatenates terms in declaration order.
    """

    arm_action = OperationalSpaceControllerActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_stiffness_task=100.0,
            motion_damping_ratio_task=1.0,
            motion_stiffness_limits_task=(50.0, 200.0),
            nullspace_control="position", # was "position"
        ),
        nullspace_joint_pos_target="center", # was "center"
        position_scale=POSITION_SCALE,
        orientation_scale=ORIENTATION_SCALE,
    )

    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": FINGER_OPEN},
        close_command_expr={"panda_finger_.*": FINGER_CLOSED},
    )
