"""Run the Manager-Based environment for Robotics Playground."""

from __future__ import annotations

import argparse
import time


def main() -> None:
    # AppLauncher должен стартовать Isaac Sim ДО импортов runtime Isaac Lab.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Run Manager-Based Franka environment for Robotics Playground."
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Камеры нужны даже в headless.
    args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Эти импорты только после запуска Isaac Sim.
    from isaaclab.envs import ManagerBasedRLEnv

    from .playground_env_cfg import G1RoughEnv3_3PlaygroundCfg
    from .ros2_bridge import Ros2VlaBridge
    from .contract import ACTION_DIM

    env = None
    bridge = None

    try:
        print("[INFO] Creating Manager-Based Playground environment...", flush=True)

        env_cfg = G1RoughEnv3_3PlaygroundCfg()
        env_cfg.sim.device = args_cli.device
        env = ManagerBasedRLEnv(cfg=env_cfg)

        print("[INFO] Creating ROS2 VLA bridge...", flush=True)

        bridge = Ros2VlaBridge(
            env=env,
            action_dim=ACTION_DIM,
        )

        env.reset()

        print(
            "[INFO] Manager-Based environment ready. "
            "ROS2 topics active.",
            flush=True,
        )

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

    finally:
        if bridge is not None:
            bridge.shutdown()

        if env is not None:
            env.close()

        simulation_app.close()


if __name__ == "__main__":
    main()
