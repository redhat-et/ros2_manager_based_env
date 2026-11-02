"""Velocity-command policy action implemented with DROID-style position targets."""
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .contract import ARM_NAMES, FINGER_NAMES, MAX_JOINT_DELTA


class DroidAction(ActionTerm):
    """7 normalized joint velocities + absolute gripper closure.

    As in DROID FrankaRobot.create_action_dict, velocities become a joint
    position target relative to the CURRENT measured state once per 15 Hz step.
    This is not an absolute joint-position policy and not Cartesian delta/IK.
    Internal gripper=-1 is reserved for idle, never accepted from policy/ROS.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("DROID live ROS bridge requires num_envs=1")
        self.arm_ids, _ = self._asset.find_joints(list(ARM_NAMES), preserve_order=True)
        self.finger_ids, _ = self._asset.find_joints(list(FINGER_NAMES), preserve_order=True)
        if len(self.arm_ids) != 7 or len(self.finger_ids) != 2:
            raise ValueError("Expected Panda arm (7 joints) and Panda hand (2 fingers)")
        self._raw = torch.zeros((env.num_envs, 8), device=env.device)
        self._processed = torch.zeros_like(self._raw)
        self._arm_target = self._asset.data.joint_pos[:, self.arm_ids].clone()
        self._finger_target = self._asset.data.joint_pos[:, self.finger_ids].clone()
        self._idle = True

    @property
    def action_dim(self):
        return 8

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        if actions.shape != self._raw.shape or not torch.isfinite(actions).all():
            raise ValueError("Expected finite actions [1,8]")
        self._raw.copy_(actions)
        idle = bool((actions[:, 7] < 0).all().item())
        if idle:
            if not self._idle:
                self._arm_target.copy_(self._asset.data.joint_pos[:, self.arm_ids])
                self._finger_target.copy_(self._asset.data.joint_pos[:, self.finger_ids])
                # Stop inertial motion once on entry to idle; keep physics active.
                self._asset.write_joint_velocity_to_sim(
                    torch.zeros_like(self._arm_target), joint_ids=self.arm_ids)
                self._asset.write_joint_velocity_to_sim(
                    torch.zeros_like(self._finger_target), joint_ids=self.finger_ids)
            self._processed.zero_()
            self._processed[:, 7] = -1
        else:
            velocity = actions[:, :7].clamp(-1, 1)
            limits = self._asset.data.soft_joint_pos_limits[:, self.arm_ids]
            target = self._asset.data.joint_pos[:, self.arm_ids] + MAX_JOINT_DELTA * velocity
            self._arm_target.copy_(torch.maximum(torch.minimum(target, limits[..., 1] - 0.01), limits[..., 0] + 0.01))
            closure = actions[:, 7].clamp(0, 1)
            if self.cfg.binarize_gripper:
                closure = (closure > 0.5).to(actions.dtype)
            self._finger_target[:] = (self.cfg.finger_open * (1 - closure))[:, None]
            self._processed[:, :7] = velocity
            self._processed[:, 7] = closure
        self._idle = idle

    def apply_actions(self):
        self._asset.set_joint_position_target(self._arm_target, joint_ids=self.arm_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self._arm_target), joint_ids=self.arm_ids)
        self._asset.set_joint_position_target(self._finger_target, joint_ids=self.finger_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self._finger_target), joint_ids=self.finger_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._processed[ids] = 0
        self._arm_target[ids] = self._asset.data.joint_pos[ids][:, self.arm_ids]
        self._finger_target[ids] = self._asset.data.joint_pos[ids][:, self.finger_ids]
        self._idle = True


@configclass
class DroidActionCfg(ActionTermCfg):
    class_type: type = DroidAction
    asset_name: str = "robot"
    finger_open: float = 0.04
    binarize_gripper: bool = True


@configclass
class ActionsCfg:
    droid = DroidActionCfg()
