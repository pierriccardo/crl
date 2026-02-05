#!/usr/bin/env python3
"""
Record or display a video of the Humanoid environment.

Usage:
  # Record one episode to videos/ (default)
  python scripts/record_humanoid_video.py

  # Live window (render_mode="human")
  python scripts/record_humanoid_video.py --mode human

  # Record with task wrapper (e.g. velocity or pose task)
  python scripts/record_humanoid_video.py --task

  # Custom video folder and record first 3 episodes
  python scripts/record_humanoid_video.py --video-dir ./my_videos --episodes 3

Requires: gymnasium, mujoco; for recording, gymnasium's RecordVideo (built-in).
If you see ModuleNotFoundError: numpy.typing, upgrade numpy or use the conda env (crl310).
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Record or display Humanoid video.")
    parser.add_argument("--mode", choices=("video", "human"), default="video",
                        help="video: save MP4 to disk; human: live window")
    parser.add_argument("--video-dir", default="videos", help="Folder for saved videos (default: videos)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (for video mode, record all up to this)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode (truncation)")
    parser.add_argument("--task", action="store_true", help="Wrap with HumanoidTaskWrapper using first task spec")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    import gymnasium as gym
    import numpy as np

    render_mode = "human" if args.mode == "human" else "rgb_array"
    env = gym.make("Humanoid-v5", render_mode=render_mode)

    if args.task:
        # Ensure project root on path for test.HumanoidTaskWrapper
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from crl.envs.tasks import HUMANOID_TASKS_SPECS
        from test import HumanoidTaskWrapper
        task_spec = HUMANOID_TASKS_SPECS["tasks"][0]
        env = HumanoidTaskWrapper(env, task_spec)

    if args.mode == "video":
        from gymnasium.wrappers import RecordVideo
        os.makedirs(args.video_dir, exist_ok=True)
        # Record first N episodes
        episode_trigger = lambda ep: ep < args.episodes
        env = RecordVideo(env, args.video_dir, episode_trigger=episode_trigger)

    obs, info = env.reset(seed=args.seed)
    steps = 0
    episode = 0

    while episode < args.episodes:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated or steps >= args.max_steps:
            obs, info = env.reset()
            steps = 0
            episode += 1
            if args.mode == "human" and episode >= args.episodes:
                break

    env.close()
    if args.mode == "video":
        print(f"Video(s) saved under {os.path.abspath(args.video_dir)}")


if __name__ == "__main__":
    main()
