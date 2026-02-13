#!/usr/bin/env python3
"""
Record videos of trained PPO policies on Humanoid tasks.

Loads a saved PPO checkpoint, runs the policy in the corresponding humanoid task
environment, and records videos to verify expected behavior.

Usage:
  # Record 3 episodes for task "walk_forward" using checkpoint
  python scripts/record_humanoid_ppo_videos.py --checkpoint models/mujoco-humanoid/walk_forward/ppo/seed_0 --task walk_forward

  # Infer task from checkpoint config (must have been trained with env.task set)
  python scripts/record_humanoid_ppo_videos.py --checkpoint models/mujoco-humanoid/stand/ppo/seed_0

  # Save to a specific folder, 5 episodes
  python scripts/record_humanoid_ppo_videos.py --checkpoint path/to/checkpoint --task run_forward --video-dir ./eval_videos --episodes 5

  # Live window instead of saving video
  python scripts/record_humanoid_ppo_videos.py --checkpoint path/to/checkpoint --task stand --mode human

  On headless machines (no display), video mode sets MUJOCO_GL=osmesa for software rendering.
  If that fails, try: MUJOCO_GL=egl python scripts/record_humanoid_ppo_videos.py ...
  or install libOSMesa (e.g. libosmesa6 on Debian/Ubuntu).

  Episodes can end before max_steps (default 1000) when the env terminates: Humanoid-v5
  terminates if the robot is "unhealthy" (e.g. torso height leaves healthy_z_range, i.e. it fell).
  So short videos mean the policy fell; the script does not impose a low step limit.
"""

from __future__ import annotations

import os
import sys

# Project root for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tyro
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class Config:
    """Record PPO policy videos on Humanoid task env."""

    checkpoint: str = field(
        metadata={"help": "Path to PPO checkpoint (directory with checkpoint.pt or .pt file)"}
    )
    task: Optional[str] = field(
        default=None,
        metadata={
            "help": "Humanoid task name (e.g. stand, walk_forward, run_forward). "
            "If not set, inferred from checkpoint config."
        },
    )
    video_dir: str = field(
        default="videos",
        metadata={"help": "Directory to save video files"},
    )
    episodes: int = field(
        default=3,
        metadata={"help": "Number of episodes to record"},
    )
    max_steps: int = field(
        default=1000,
        metadata={"help": "Max steps per episode (truncation)"},
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed for env reset"},
    )
    device: str = field(
        default="cpu",
        metadata={"help": "Device for policy (cpu or cuda)"},
    )
    mode: Literal["video", "human"] = field(
        default="video",
        metadata={"help": "video: save MP4 to video_dir; human: live window"},
    )


def main(config: Config) -> None:
    # Use headless/software rendering for video when no display (must be before mujoco is loaded)
    if config.mode == "video":
        os.environ.setdefault("MUJOCO_GL", "osmesa")

    import numpy as np
    import torch

    from crl.envs import make_env
    from crl.envs.tasks import HUMANOID_TASK_BY_NAME
    from crl.algos.ppo import PPO

    # Load agent from checkpoint
    agent, payload = PPO.from_checkpoint(config.checkpoint, device=config.device)
    agent.actor.eval()

    # Infer task from checkpoint if not provided
    task = config.task
    if task is None:
        cfg = payload.get("config", {})
        env_cfg = cfg.get("env") or {}
        if isinstance(env_cfg, dict):
            task = env_cfg.get("task", "stand")
        else:
            task = getattr(env_cfg, "task", "stand")
        print(f"Inferred task from checkpoint: {task}")

    if task not in HUMANOID_TASK_BY_NAME:
        known = ", ".join(sorted(HUMANOID_TASK_BY_NAME.keys()))
        raise ValueError(f"Unknown humanoid task {task!r}. Known: {known}")

    # Create env with same task (factory applies HumanoidTaskWrapper)
    render_mode = "human" if config.mode == "human" else "rgb_array"
    env = make_env(
        "mujoco/humanoid",
        task=task,
        seed=config.seed,
        render_mode=render_mode,
    )

    if config.mode == "video":
        from gymnasium.wrappers import RecordVideo

        os.makedirs(config.video_dir, exist_ok=True)
        # Record first N episodes; name includes task for clarity
        episode_trigger = lambda ep: ep < config.episodes
        env = RecordVideo(
            env,
            config.video_dir,
            episode_trigger=episode_trigger,
            name_prefix=f"humanoid_{task}",
        )

    obs, info = env.reset(seed=config.seed)
    episode = 0
    steps_in_episode = 0
    total_reward = 0.0

    while episode < config.episodes:
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            action = agent.act(obs_t, values=False)
            action_np = action.cpu().numpy()[0]

        obs, reward, terminated, truncated, info = env.step(action_np)
        steps_in_episode += 1
        total_reward += reward

        if terminated or truncated or steps_in_episode >= config.max_steps:
            reason = (
                "unhealthy (fell)" if terminated
                else "time limit (truncated)" if truncated
                else "max_steps reached"
            )
            print(
                f"Episode {episode + 1}/{config.episodes} ended after {steps_in_episode} steps "
                f"(reason: {reason}) reward={total_reward:.1f} task={info.get('task_name', task)}"
            )
            obs, info = env.reset()
            steps_in_episode = 0
            total_reward = 0.0
            episode += 1

    env.close()

    if config.mode == "video":
        print(f"Videos saved under {os.path.abspath(config.video_dir)}")


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
