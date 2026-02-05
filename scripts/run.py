#!/usr/bin/env python3
"""Build N commands (seeds x algos) and run them all in parallel."""

import subprocess
import sys
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("fb_cpr", "varibad", "ptdqn")


def build_args(env_name: str, task_list: str, seed: int, env_seed: int, device: str, num_episodes: int, task_switch_prob: float, max_episode_steps: int) -> dict[str, list[str]]:
    common = [
        f"--env.domain_name={env_name}",
        f"--env.task_list={task_list}",
        f"--env.max_episode_steps={max_episode_steps}",
        f"--env.task_switch_prob={task_switch_prob}",
        f"--env.seed={env_seed}",
        f"--num_episodes={num_episodes}",
    ]
    return {
        "varibad": [str(ROOT / "crl" / "algos" / "varibad.py"), *common, f"--device={device}", f"--seed={seed}"],
        "fb_cpr": [
            str(ROOT / "crl" / "algos" / "fb_cpr.py"),
            *common,
            f"--model.device={device}",
            "--expl.epsilon=0.2",
            f"--seed={seed}",
        ],
        "ptdqn": [str(ROOT / "crl" / "algos" / "ptdqn.py"), *common, f"--device={device}", f"--seed={seed}"],
    }


def main(
    env_name: str = "highway_parking",
    task_list: str = "full",
    algos: list[str] | None = None,
    seeds: list[int] = [0],
    env_seed: int = 0,
    device: str = "cpu",
    num_episodes: int = 10000,
    task_switch_prob: float = 0.01,
    max_episode_steps: int = 1000,
) -> None:
    algo_list = algos or list(ALGOS)
    commands = []
    for seed in seeds:
        argv = build_args(env_name, task_list, seed, env_seed, device, num_episodes, task_switch_prob, max_episode_steps)
        for name in algo_list:
            if name not in argv:
                continue
            commands.append((seed, name, [sys.executable] + argv[name]))

    if not commands:
        return
    print(f"[run] Starting {len(commands)} jobs in parallel", flush=True)
    for seed, name, cmd in commands:
        print(f"  {env_name} | {task_list} | {name} | seed={seed}", flush=True)

    procs = []
    for _, _, cmd in commands:
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    max_code = max(p.wait() for p in procs)
    if max_code != 0:
        sys.exit(max_code)


if __name__ == "__main__":
    tyro.cli(main)
