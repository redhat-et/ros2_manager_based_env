"""Launch DROID using the working custom_env_3_3 application lifecycle."""
from __future__ import annotations

import argparse
import time
import traceback


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="DROID pi05 Panda environment")
    parser.add_argument("--ros-prefix", default="/droid")
    parser.add_argument("--fast", action="store_true", help="Disable wall-clock pacing")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Same import order and placement as custom_env_3_3: Kit first,
    # runtime imports next, environment/bridge construction inside try.
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    import omni.kit.app
    import omni.usd
    from pxr import Gf, UsdGeom

    from .contract import CONTROL_HZ, PHYSICS_HZ, MODEL_HORIZON
    from .custom_velocity_env_cfg import DroidPi05EnvCfg
    from .ros2_bridge import DroidBridge

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
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/panda_hand/WristCamera")
        if not prim.IsValid():
            raise RuntimeError(
                f"Wrist camera prim was not found: {'/World/envs/env_0/Robot/panda_hand/WristCamera'}"
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

        translate_op.Set(Gf.Vec3d(*(0.13, 0.0, -0.15)))

        # Force a USD change even if CameraCfg authored the same final value.
        orient_op.Set(quat_for_xform_op(orient_op, (1.0, 0.0, 0.0, 0.0)))
        omni.kit.app.get_app().update()
        env.sim.render()

        orient_op.Set(quat_for_xform_op(orient_op, (0.11062, 0.6984, 0.6984, 0.11062)))
        omni.kit.app.get_app().update()

        for _ in range(3):
            env.sim.render()

        print(
            "[INFO] Re-authored wrist camera transform through USD:",
            "/World/envs/env_0/Robot/panda_hand/WristCamera",
            prim.GetAttribute("xformOpOrder").Get(),
            flush=True,
        )

    env = None
    bridge = None
    try:
        print("[INFO] Creating DROID Manager-Based environment...", flush=True)
        env_cfg = DroidPi05EnvCfg()
        env_cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=env_cfg)

        print("[INFO] Creating DROID ROS2 bridge...", flush=True)
        bridge = DroidBridge(args_cli.ros_prefix)
        bridge.set_decimation(env.cfg.decimation)
        env.reset()

        # Like custom_env_3_3, step twice before publishing camera data.
        # DROID's -1 gripper sentinel holds the initial pose during warmup.
        warmup_action = torch.zeros(env.num_envs, 8, device=env.device)
        warmup_action[:, 7] = -1
        for _ in range(2):
            env.step(warmup_action)
        force_wrist_camera_usd_pose(env)
        obs = env.observation_manager.compute()

        sim_time = 0.0
        idle = True
        bridge.publish(obs["policy"], env, sim_time, idle)
        print(f"DROID READY: physics={PHYSICS_HZ}Hz control={CONTROL_HZ}Hz "
              f"model_chunk={MODEL_HORIZON}x8 prefix={bridge.prefix}", flush=True)

        def reset_environment():
            nonlocal obs, sim_time, idle

            with torch.inference_mode():
                obs, _ = env.reset()

            sim_time = 0.0
            idle = True

        def step_environment():
            nonlocal obs, sim_time, idle
            row, idle = bridge.action()
            with torch.inference_mode():
                obs, _, terminated, truncated, _ = env.step(
                    torch.as_tensor(row, device=env.device).unsqueeze(0))
            sim_time += 1.0 / CONTROL_HZ
            if bool(terminated.any() or truncated.any()):
                bridge.clear_action_buffer()
                reset_environment()

        while simulation_app.is_running():
            start = time.monotonic()
            handled = bridge.process_control_commands(
                reset_environment, step_environment, env.cfg.decimation)
            running = not bridge.is_paused
            if running and not handled:
                step_environment()
            else:
                env.sim.render()
            bridge.publish(obs["policy"], env, sim_time, idle)
            if not args_cli.fast or not running:
                time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    except Exception:
        # Print before Kit shutdown, which can terminate the interpreter.
        traceback.print_exc()
        raise
    finally:
        try:
            if bridge is not None:
                bridge.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                simulation_app.close()


if __name__ == "__main__":
    main()
