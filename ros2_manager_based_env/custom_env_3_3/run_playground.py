"""Run the Manager-Based environment for Robotics Playground."""

from __future__ import annotations

import argparse


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

    env = None
    bridge = None

    try:
        print("[INFO] Creating Manager-Based Playground environment...", flush=True)

        env_cfg = G1RoughEnv3_3PlaygroundCfg()
        env = ManagerBasedRLEnv(cfg=env_cfg)

        print("[INFO] Creating ROS2 VLA bridge...", flush=True)

        bridge = Ros2VlaBridge(
            camera_names=[
                "wrist_cam",
                "table_cam",
            ],
            env=env,
            state_topic="/vla/obs/state",
            action_topic="/vla/action",
            camera_topic_prefix="/vla/obs/",
            action_dim=7,
        )

        env.reset()

        print(
            "[INFO] Manager-Based environment ready. "
            "ROS2 topics active.",
            flush=True,
        )

        while simulation_app.is_running():
            if bridge.is_paused:
                import time
                time.sleep(0.01)
                continue

            # One row [1, 7] from current VLA action chunk.
            action = bridge.get_action().to(env.device)

            _, _, terminated, truncated, _ = env.step(action)

            # Ros2VlaBridge reads cameras/state directly from env.scene.
            bridge.publish_observations(env)

            if bool(terminated.any() or truncated.any()):
                env.reset()

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
