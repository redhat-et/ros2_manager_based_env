from isaaclab.utils import configclass

from .rough_env_3_3_cfg import G1RoughEnv3_3Cfg_ROS2


@configclass
class G1RoughEnv3_3PlaygroundCfg(G1RoughEnv3_3Cfg_ROS2):
    """Manager-Based environment configured for Robotics Playground."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1

        # Physics: 150 Hz.
        self.sim.dt = 1.0 / 100.0

        # VLA actions: 10 Hz.
        self.decimation = 10

        # Playground controls reset explicitly.
        self.episode_length_s = 24.0 * 60.0 * 60.0
        self.is_finite_horizon = False
        self.terminations.time_out = None

        self.observations.policy.enable_corruption = False
