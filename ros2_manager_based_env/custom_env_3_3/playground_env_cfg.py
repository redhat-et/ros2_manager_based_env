from isaaclab.utils import configclass

from .rough_env_3_3_cfg import G1RoughEnv3_3Cfg_ROS2


@configclass
class G1RoughEnv3_3PlaygroundCfg(G1RoughEnv3_3Cfg_ROS2):
    """Manager-Based environment configured for Robotics Playground."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1

        # Inherit 150 Hz physics / 15 Hz control / decimation=10 from
        # CustomLocomotionVelocityRoughEnvCfg and the shared contract.

        # Playground controls reset explicitly.
        self.episode_length_s = 24.0 * 60.0 * 60.0
        self.is_finite_horizon = False
        self.terminations.time_out = None

        self.observations.policy.enable_corruption = False
