"""ObservationManager terms for DROID camera and proprioception inputs."""
from isaaclab.envs.mdp.observations import image
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .observations import joint_positions, gripper_closure


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        exterior_image_1_left = ObservationTermCfg(func=image, params={
            "sensor_cfg": SceneEntityCfg("exterior_image_1_left"), "data_type": "rgb", "normalize": False})
        wrist_image_left = ObservationTermCfg(func=image, params={
            "sensor_cfg": SceneEntityCfg("wrist_image_left"), "data_type": "rgb", "normalize": False})
        joint_position = ObservationTermCfg(func=joint_positions)
        gripper_position = ObservationTermCfg(func=gripper_closure)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyCfg()
