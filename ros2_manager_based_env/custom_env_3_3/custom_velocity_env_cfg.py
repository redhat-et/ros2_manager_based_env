from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from .custom_observations_cfg import ObservationsCfg
from .custom_rewards_cfg import Rewards
from .custom_commands_cfg import CommandsCfg
from .custom_event_cfg import EventCfg
from .custom_scene_cfg import SceneCfg
from .custom_terminations_cfg import TerminationsCfg
from .custom_curriculum_cfg import CurriculumCfg
from .custom_actions_cfg import ActionsCfg

class CustomLocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    scene : SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    rewards: Rewards = Rewards()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    terminations : TerminationsCfg  = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    actions: ActionsCfg = ActionsCfg()
    
    def __post_init__(self):
        """Post initialization."""
        
        
        # general settings
        # Control: 150 / 10 = 15 Hz
        self.decimation = 10  # 10 physics steps per 1 control step


        self.episode_length_s = 10.0
        self.is_finite_horizon = True

        # simulation settings
        # Physics: 150 Hz
        self.sim.dt = 1 / 150 
        self.sim.render_interval = 1  # Render every physics step: 150 Hz
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
        

        


