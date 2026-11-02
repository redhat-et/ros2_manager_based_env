"""DROID rollout config using the working custom_env_3_3 RL environment API."""
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from .custom_actions_cfg import ActionsCfg
from .custom_observations_cfg import ObservationsCfg
from .custom_scene_cfg import DroidSceneCfg
from .contract import PHYSICS_HZ, DECIMATION


@configclass
class EventsCfg:
    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    pass


@configclass
class DroidPi05EnvCfg(ManagerBasedRLEnvCfg):
    scene = DroidSceneCfg(num_envs=1, env_spacing=2.5)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    commands = None
    curriculum = None
    decimation = DECIMATION
    episode_length_s = 3600.0
    is_finite_horizon = False

    def __post_init__(self):
        self.sim.dt = 1.0 / PHYSICS_HZ
        self.sim.render_interval = 1
        self.rerender_on_reset = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
