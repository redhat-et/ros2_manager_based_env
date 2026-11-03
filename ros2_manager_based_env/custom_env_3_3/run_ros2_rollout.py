"""Run the custom Isaac Lab environment with the local ROS 2 bridge."""

from __future__ import annotations

import argparse
import time


WRIST_CAMERA_PRIM_PATH = "/World/envs/env_0/Robot/panda_hand/wrist_cam"
WRIST_CAMERA_POSITION = (0.13, 0.0, -0.15)
WRIST_CAMERA_QUAT_WXYZ = (0.11062, 0.6984, 0.6984, 0.11062)


def main() -> None:
    """Launch Isaac Sim, create the manager-based environment, and bridge ROS 2."""
    # Importing AppLauncher itself is safe before Kit starts. Runtime Isaac Lab,
    # Omniverse, and task modules are imported only after the app is launched.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Run the custom Franka environment with ROS 2 I/O."
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Camera rendering is required even in headless mode.
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Isaac Lab/Isaac Sim imports must happen after AppLauncher.
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    import omni.kit.app
    import omni.usd
    from pxr import Gf, UsdGeom

    from .rough_env_3_3_cfg import G1RoughEnv3_3Cfg_ROS2
    from .ros2_bridge import Ros2VlaBridge
    from .contract import ACTION_DIM

    def quat_for_xform_op(
        op: UsdGeom.XformOp,
        quat_wxyz: tuple[float, float, float, float],
    ):
        """Create a quaternion matching the precision of an existing USD op."""
        w, x, y, z = quat_wxyz
        if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
            return Gf.Quatf(w, Gf.Vec3f(x, y, z))
        return Gf.Quatd(w, Gf.Vec3d(x, y, z))

    def force_wrist_camera_usd_pose(env: ManagerBasedRLEnv) -> None:
        """Re-author the local camera transform after the render product exists."""
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(WRIST_CAMERA_PRIM_PATH)
        if not prim.IsValid():
            raise RuntimeError(
                f"Wrist camera prim was not found: {WRIST_CAMERA_PRIM_PATH}"
            )

        xformable = UsdGeom.Xformable(prim)
        ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

        translate_op = ops.get("xformOp:translate")
        if translate_op is None:
            translate_op = xformable.AddTranslateOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            )

        orient_op = ops.get("xformOp:orient")
        if orient_op is None:
            orient_op = xformable.AddOrientOp(
                precision=UsdGeom.XformOp.PrecisionDouble
            )

        translate_op.Set(Gf.Vec3d(*WRIST_CAMERA_POSITION))

        # Force a USD change even if CameraCfg authored the same final value.
        orient_op.Set(quat_for_xform_op(orient_op, (1.0, 0.0, 0.0, 0.0)))
        omni.kit.app.get_app().update()
        env.sim.render()

        orient_op.Set(quat_for_xform_op(orient_op, WRIST_CAMERA_QUAT_WXYZ))
        omni.kit.app.get_app().update()

        for _ in range(3):
            env.sim.render()

        print(
            "[INFO] Re-authored wrist camera transform through USD:",
            WRIST_CAMERA_PRIM_PATH,
            prim.GetAttribute("xformOpOrder").Get(),
            flush=True,
        )

    env = None
    bridge = None
    try:
        env_cfg = G1RoughEnv3_3Cfg_ROS2()
        env_cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=env_cfg)

        bridge = Ros2VlaBridge(
            env=env,
            action_dim=ACTION_DIM,
        )

        env.reset()

        warmup_action = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
        for _ in range(2):
            env.step(warmup_action)

        force_wrist_camera_usd_pose(env)

        def reset_environment():
            with torch.inference_mode():
                env.reset()

        def step_environment():
            with torch.inference_mode():
                action = bridge.get_action().to(env.device)
                _, _, terminated, truncated, _ = env.step(action)

                if bool(terminated.any() or truncated.any()):
                    bridge.clear_action_buffer()
                    env.reset()

        while simulation_app.is_running():
            handled = bridge.process_control_commands(
                reset_environment, step_environment, env.cfg.decimation)
            if not bridge.is_paused and not handled:
                step_environment()
            else:
                env.sim.render()
                time.sleep(0.01)
            bridge.publish_observations(env)

    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if bridge is not None:
            bridge.shutdown()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
