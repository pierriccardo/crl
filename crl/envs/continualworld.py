"""Continual World protocol helpers for Meta-World v3.

This module keeps benchmark-specific logic separate from env factory plumbing:
- canonical CW10/CW20 task presets (defined on v1 names, mapped to v3)
- creation of single-task Meta-World envs with MT1 task binding
"""

from __future__ import annotations

from typing import Iterable

import gymnasium as gym
import metaworld
from metaworld.env_dict import ALL_V3_ENVIRONMENTS


# Canonical task presets from Continual World (v1 naming in upstream).
CW10_TASKS_V1: tuple[str, ...] = (
    "hammer-v1",
    "push-wall-v1",
    "faucet-close-v1",
    "push-back-v1",
    "stick-pull-v1",
    "handle-press-side-v1",
    "push-v1",
    "shelf-place-v1",
    "window-close-v1",
    "peg-unplug-side-v1",
)

# CW20 is CW10 repeated twice.
CW20_TASKS_V1: tuple[str, ...] = CW10_TASKS_V1 + CW10_TASKS_V1


def v1_to_v3_task_name(task_name: str) -> str:
    """Convert a Continual World v1 task name to Meta-World v3 naming."""
    if not task_name.endswith("-v1"):
        raise ValueError(f"Expected a v1 task name ending with '-v1', got: {task_name!r}")
    converted = f"{task_name[:-3]}-v3"
    if converted not in ALL_V3_ENVIRONMENTS:
        raise ValueError(
            f"Task {task_name!r} maps to {converted!r}, which is unavailable in metaworld v3."
        )
    return converted


def map_v1_sequence_to_v3(tasks_v1: Iterable[str]) -> list[str]:
    return [v1_to_v3_task_name(task) for task in tasks_v1]


def cw10_v3() -> list[str]:
    return map_v1_sequence_to_v3(CW10_TASKS_V1)


def cw20_v3() -> list[str]:
    return map_v1_sequence_to_v3(CW20_TASKS_V1)


def make_continualworld_env(
    task: str,
    seed: int = 0,
    max_episode_steps: int = 1000,
) -> gym.Env:
    """Create a single-task Meta-World v3 env following CW-style MT1 task binding."""
    if task not in ALL_V3_ENVIRONMENTS:
        known = ", ".join(sorted(ALL_V3_ENVIRONMENTS))
        raise ValueError(f"Unknown ContinualWorld task {task!r}. Known: {known}")

    benchmark = metaworld.MT1(task, seed=seed)
    env = ALL_V3_ENVIRONMENTS[task]()
    env.set_task(benchmark.train_tasks[0])

    # Force a common horizon from the outer benchmark config.
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    return env
