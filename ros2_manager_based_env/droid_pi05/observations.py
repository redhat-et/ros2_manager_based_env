"""Raw DROID observation functions; imported after AppLauncher starts Isaac Sim."""
from .contract import ARM_NAMES, FINGER_NAMES


def joint_positions(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(ARM_NAMES), preserve_order=True)
    return robot.data.joint_pos[:, ids]


def gripper_closure(env):
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(FINGER_NAMES), preserve_order=True)
    return (1.0 - robot.data.joint_pos[:, ids].mean(dim=-1, keepdim=True) / 0.04).clamp(0, 1)


