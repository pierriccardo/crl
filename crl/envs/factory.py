# crl/envs/factory.py
from __future__ import annotations

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

# Environments, adapters and wrappers
from crl.envs.gridenv import GoalEnv
from crl.envs.adapters import MiniAtarEnv
from crl.envs.wrappers import ImageWrapper, ActionWrapper, ChannelStandardizationWrapper

from typing import Callable, Dict, Any, Optional


# --- registry of callables returning a fresh env (unwrapped) ---
_REGISTRY: Dict[str, Callable[..., gym.Env]] = {}

# --- registry of task sequences ---
_TASK_SEQUENCES: Dict[str, Dict[str, list[str]]] = {}


def _register(name: str):
    def deco(fn: Callable[..., gym.Env]):
        _REGISTRY[name.lower()] = fn
        return fn
    return deco


def _register_task_sequence(env_name: str, sequence_name: str, task_list: list[str]):
    """Register a named task sequence for a specific environment."""
    env_key = env_name.lower()
    sequence_key = sequence_name.lower()

    if env_key not in _TASK_SEQUENCES:
        _TASK_SEQUENCES[env_key] = {}

    _TASK_SEQUENCES[env_key][sequence_key] = task_list


# ============================================
# Single task environments
# ============================================

@_register("goalenv")
def _goal_env(**kwargs) -> gym.Env:
    """Single task goal environment with configurable task."""
    # Extract task parameter
    task = kwargs.pop('task', 'north')  # Default to 'north' task

    # Create GoalEnv with specified task
    env = GoalEnv(task=task, **kwargs)

    # Apply standard wrappers
    env = ImageWrapper(env)
    env = ActionWrapper(env=env, action_subset=[0, 1, 2])
    return env


@_register("minatar")
def _minatar_env(**kwargs) -> gym.Env:
    """MinAtar environment with configurable task (game)."""
    # Extract task parameter (which corresponds to the game name)
    task = kwargs.pop('task', 'breakout')  # Default to 'breakout' game
    target_channels = kwargs.pop('target_channels', 4)  # Use minimum channels for efficiency

    # Validate task is a valid MinAtar game
    valid_games = ['asterix', 'breakout', 'freeway', 'seaquest', 'space_invaders']
    if task not in valid_games:
        raise ValueError(f"Invalid MinAtar task '{task}'. Valid games: {valid_games}")

    # Create MiniAtarEnv with specified game
    env = MiniAtarEnv(game=task, **kwargs)

    env = ChannelStandardizationWrapper(env, target_channels=target_channels)
    return env


# ============================================
# Task sequence registrations
# ============================================

# GoalEnv sequences
_register_task_sequence("goalenv", "cardinal", ["north", "south", "east", "west"])
_register_task_sequence("goalenv", "horizontal", ["east", "west"])
_register_task_sequence("goalenv", "vertical", ["north", "south"])
_register_task_sequence("goalenv", "clockwise", ["north", "east", "south", "west"])
_register_task_sequence("goalenv", "counter_clockwise", ["north", "west", "south", "east"])

# MinAtar sequences
_register_task_sequence("minatar", "all", ["asterix", "breakout", "freeway", "seaquest", "space_invaders"])
_register_task_sequence("minatar", "classic", ["breakout", "space_invaders", "asterix"])
_register_task_sequence("minatar", "navigation", ["freeway", "seaquest"])
_register_task_sequence("minatar", "action", ["asterix", "space_invaders"])
_register_task_sequence("minatar", "puzzle", ["breakout", "freeway"])


# ============================================
# Exposed public methods
# ============================================

def make_env(
    env_id: str,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    record_stats: bool = True,
    wrappers: Optional[list[Callable[[gym.Env], gym.Env]]] = None,
    **env_kwargs: Any,
) -> gym.Env:
    """
    Create a single environment by id with consistent seeding & optional wrappers.
    - env_id: one of the keys in _REGISTRY (case-insensitive), e.g. "minatar/breakout"
    - seed: base seed; also used to seed action/obs spaces
    - render_mode: forwarded to the underlying env if supported
    - record_stats: attaches RecordEpisodeStatistics
    - wrappers: list of callables env->env to apply after creation
    - env_kwargs: forwarded to the underlying constructor
    """
    key = env_id.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown env_id '{env_id}'. Known: {known}")

    # Forward render_mode if constructor accepts it
    if "render_mode" not in env_kwargs:
        env_kwargs["render_mode"] = render_mode

    env = _REGISTRY[key](**env_kwargs)

    # Consistent seeding
    if seed is not None:
        env.reset(seed=seed)
        # Also seed spaces when available
        try:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        except Exception:
            pass

    # Optional per-project standard wrapper
    if record_stats:
        env = RecordEpisodeStatistics(env)

    # User-specified wrapper chain
    if wrappers:
        for wrap in wrappers:
            env = wrap(env)

    return env


def get_task_sequence(env_name: str, sequence_name: str) -> list[str]:
    """Retrieve a task sequence by environment and sequence name."""
    env_key = env_name.lower()
    sequence_key = sequence_name.lower()

    if env_key not in _TASK_SEQUENCES:
        raise ValueError(f"No task sequences registered for environment '{env_name}'")

    if sequence_key not in _TASK_SEQUENCES[env_key]:
        available = ", ".join(sorted(_TASK_SEQUENCES[env_key].keys()))
        raise ValueError(f"Unknown sequence '{sequence_name}' for env '{env_name}'. Available: {available}")

    return _TASK_SEQUENCES[env_key][sequence_key]


def list_task_sequences(env_name: Optional[str] = None) -> Dict[str, Dict[str, list[str]]]:
    """List all available task sequences, optionally filtered by environment."""
    if env_name is None:
        return _TASK_SEQUENCES.copy()

    env_key = env_name.lower()
    if env_key not in _TASK_SEQUENCES:
        return {}

    return {env_key: _TASK_SEQUENCES[env_key].copy()}


def list_envs() -> list[str]:
    """
    List all available environment names.

    Returns:
        List of registered environment names
    """
    return sorted(_REGISTRY.keys())


def make_vec_env(
    env_id: str,
    num_envs: int,
    seed: Optional[int] = None,
    async_mode: bool = False,
    start_index: int = 0,
    **kwargs: Any,
):
    """
    Vectorized envs with unique seeds: seed+i
    Usage:
        venv = make_vec_env("minatar/breakout", 8, seed=42, async_mode=True)
    """
    def thunk(i: int):
        def _th():
            return make_env(env_id, seed=None if seed is None else seed + i, **kwargs)
        return _th

    thunks = [thunk(i + start_index) for i in range(num_envs)]
    Vec = AsyncVectorEnv if async_mode else SyncVectorEnv
    return Vec(thunks)


if __name__ == "__main__":

    print("Available envs:")

    all_sequences = list_task_sequences()
    for env_name, sequences in all_sequences.items():
        print(f"   {env_name}:")
        for seq_name, tasks in sequences.items():
            print(f"     - {seq_name}: {tasks}")

    # List available environments
    envs = list_envs()
    print(f"Available environments: {envs}")

    # Get cardinal sequence for goalenv
    cardinal_tasks = get_task_sequence("goalenv", "cardinal")
    print(f"goalenv cardinal: {cardinal_tasks}")

    # Get all MinAtar games
    all_minatar = get_task_sequence("minatar", "all")
    print(f"minatar all: {all_minatar}")

    # Get classic MinAtar games
    classic_minatar = get_task_sequence("minatar", "classic")
    print(f"minatar classic: {classic_minatar}")
