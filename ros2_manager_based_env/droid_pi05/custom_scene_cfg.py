"""Standalone Panda scene; no imports or inheritance from another custom env."""
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from .contract import CONTROL_HZ


def panda():
    robot = FRANKA_PANDA_CFG.copy()
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    # Approximate gravity-compensated DROID position controller in simulation.
    robot.spawn.rigid_props.disable_gravity = True
    robot.init_state.joint_pos = {
        "panda_joint1": 0.0, "panda_joint2": -math.pi / 5,
        "panda_joint3": 0.0, "panda_joint4": -4 * math.pi / 5,
        "panda_joint5": 0.0, "panda_joint6": 3 * math.pi / 5,
        "panda_joint7": 0.0, "panda_finger_joint.*": 0.04,
    }
    robot.actuators["panda_shoulder"].stiffness = 400.0
    robot.actuators["panda_shoulder"].damping = 80.0
    robot.actuators["panda_forearm"].stiffness = 400.0
    robot.actuators["panda_forearm"].damping = 80.0
    return robot


def camera(path, pos, rot):
    return CameraCfg(
        prim_path=path, update_period=1.0 / CONTROL_HZ,
        height=224, width=224, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=36.0, vertical_aperture=36.0,
            focus_distance=1.0, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention="opengl"))


@configclass
class DroidSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -1.05)))
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0, 0), rot=(0.7071068, 0, 0, 0.7071068)))
    robot = panda()
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    red_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RedCube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.15, 0.03)),
        spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.05, 0.05))))
    exterior_image_1_left = camera(
        "{ENV_REGEX_NS}/ExteriorCamera", (1.3, 0.0, 0.55),
        (0.55721, 0.43534, 0.43534, 0.55721))
    wrist_image_left = camera(
        "{ENV_REGEX_NS}/Robot/panda_hand/WristCamera", (0.13, 0.0, -0.15),
        (0.11062, 0.6984, 0.6984, 0.11062))
