# crl/envs/factory.py
from __future__ import annotations

import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers import RecordEpisodeStatistics

# Environments, adapters and wrappers
from crl.envs.gridenv import GoalEnv
from crl.envs.adapters import DMControlEnv, ContinualWorldEnv
from crl.envs.wrappers import ImageWrapper, ActionWrapper, ContinualEpisodicWrapper

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


@_register("dmcontrol")
def _dmcontrol_env(**kwargs) -> gym.Env:
    """DeepMind Control Suite environment with configurable domain and task."""
    # Extract domain and task parameters
    # Task format can be "domain/task" or separate domain and task params
    task = kwargs.pop('task', None)
    domain = kwargs.pop('domain', None)

    # Parse task if it's in "domain/task" format
    if task and '/' in task:
        domain, task = task.split('/', 1)

    # Set defaults
    if domain is None:
        domain = 'cartpole'
    if task is None:
        task = 'swingup'

    # Create DMControlEnv with specified domain and task
    env = DMControlEnv(domain=domain, task=task, **kwargs)
    return env


@_register("continualworld")
def _continualworld_env(**kwargs) -> gym.Env:
    """Continual World (MetaWorld) environment with configurable task."""
    # Extract task parameter
    task = kwargs.pop('task', 'hammer-v1')  # Default to hammer-v1

    # Create ContinualWorldEnv with specified task
    env = ContinualWorldEnv(task=task, **kwargs)
    return env


# ============================================
# Task sequence registrations
# ============================================
#
# _register_task_sequence(ENV_NAME, SEQ_NAME, TASKS)
#


# GoalEnv sequences
_register_task_sequence("goalenv", "cardinal", ["north", "south", "east", "west"])
_register_task_sequence("goalenv", "horizontal", ["east", "west"])
_register_task_sequence("goalenv", "vertical", ["north", "south"])
_register_task_sequence("goalenv", "clockwise", ["north", "east", "south", "west"])
_register_task_sequence("goalenv", "counter_clockwise", ["north", "west", "south", "east"])

# DMControl sequences
_register_task_sequence("dmcontrol", "walker_basic", ["walker/stand", "walker/walk"])
_register_task_sequence("dmcontrol", "walker_full", ["walker/stand", "walker/walk", "walker/run"])
_register_task_sequence("dmcontrol", "cartpole_basic", ["cartpole/swingup", "cartpole/balance"])
_register_task_sequence("dmcontrol", "cartpole_full", [
    "cartpole/swingup", "cartpole/balance", "cartpole/balance_sparse"
])

# Continual World (MetaWorld) sequences
# CW20: The standard 20-task sequence from Continual World benchmark
_register_task_sequence("continualworld", "cw20", [
    "hammer-v1",
    "push-v1",
    "door-open-v1",
    "drawer-close-v1",
    "drawer-open-v1",
    "reach-v1",
    "push-wall-v1",
    "shelf-place-v1",
    "button-press-v1",
    "button-press-topdown-v1",
    "button-press-topdown-wall-v1",
    "peg-insert-side-v1",
    "window-open-v1",
    "window-close-v1",
    "door-close-v1",
    "reach-wall-v1",
    "push-back-v1",
    "lever-pull-v1",
    "box-close-v1",
    "hand-insert-v1",
])

# CW10: A subset of 10 tasks from CW20
_register_task_sequence("continualworld", "cw10", [
    "hammer-v1",
    "push-v1",
    "door-open-v1",
    "drawer-close-v1",
    "drawer-open-v1",
    "reach-v1",
    "push-wall-v1",
    "shelf-place-v1",
    "button-press-v1",
    "button-press-topdown-v1",
])


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
    - env_id: one of the keys in _REGISTRY (case-insensitive), e.g. "goalenv"
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
        venv = make_vec_env("goalenv", 8, seed=42, async_mode=True)
    """
    def thunk(i: int):
        def _th():
            return make_env(env_id, seed=None if seed is None else seed + i, **kwargs)
        return _th

    thunks = [thunk(i + start_index) for i in range(num_envs)]
    Vec = AsyncVectorEnv if async_mode else SyncVectorEnv
    return Vec(thunks)


def make_continual_episodic_env(
    env_id: str,
    task_list: list[str],
    episode_length: int = 100,
    task_switch_prob: float = 1.0,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    record_stats: bool = True,
    wrappers: Optional[list[Callable[[gym.Env], gym.Env]]] = None,
    **env_kwargs: Any,
) -> gym.Env:
    """
    Create a continual episodic environment where:
    - Each episode lasts for exactly N timesteps (episode_length)
    - A task is selected at the start of each episode with a given probability
    - Tasks can persist across multiple episodes, allowing the agent to face
      the same task for consecutive episodes

    Args:
        env_id: Environment identifier (e.g., "goalenv", "continualworld")
        task_list: List of task identifiers to randomly sample from
        episode_length: Fixed number of timesteps per episode (N)
        task_switch_prob: Probability of switching to a new task at each episode reset.
                         Default 1.0 means task changes every episode.
                         Lower values (e.g., 0.1) allow tasks to persist across episodes.
        seed: Random seed for task selection and environment
        render_mode: Forwarded to underlying environment
        record_stats: Attaches RecordEpisodeStatistics wrapper
        wrappers: Additional wrappers to apply before ContinualEpisodicWrapper
        env_kwargs: Additional arguments forwarded to make_env

    Returns:
        Wrapped environment with continual episodic behavior

    Example:
        env = make_continual_episodic_env(
            "goalenv",
            task_list=["north", "south", "east", "west"],
            episode_length=200,
            task_switch_prob=0.1,  # 10% chance to switch task each episode
            seed=42
        )
    """

    # Create initial environment (will be replaced on first reset)
    initial_task = task_list[0]
    base_env = make_env(
        env_id,
        task=initial_task,
        seed=None,  # Don't seed here, will be seeded in wrapper
        render_mode=render_mode,
        record_stats=False,  # Apply after continual wrapper
        wrappers=wrappers,
        **env_kwargs
    )

    # Create factory function for creating new environments with different tasks
    def env_factory(task: str) -> gym.Env:
        return make_env(
            env_id,
            task=task,
            seed=None,
            render_mode=render_mode,
            record_stats=False,
            wrappers=wrappers,
            **env_kwargs
        )

    # Wrap with ContinualEpisodicWrapper
    env = ContinualEpisodicWrapper(
        env=base_env,
        task_list=task_list,
        episode_length=episode_length,
        env_factory=env_factory,
        task_switch_prob=task_switch_prob,
        seed=seed
    )

    # Apply record_stats wrapper if requested
    if record_stats:
        env = RecordEpisodeStatistics(env)

    return env


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

    # Get DMControl sequences
    walker_tasks = get_task_sequence("dmcontrol", "walker_basic")
    print(f"dmcontrol walker_basic: {walker_tasks}")

    # Test dm_control environment
    print("\n" + "="*50)
    print("Testing DMControl environment...")
    print("="*50)
    try:
        # Test with cartpole (simple, fast)
        print("\nTesting cartpole/swingup...")
        env = make_env("dmcontrol", task="cartpole/swingup", seed=42)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial observation range: [{obs.min():.3f}, {obs.max():.3f}]")

        # Run a few steps
        total_reward = 0.0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"  Step {step+1}: reward={reward:.3f}, "
                  f"terminated={terminated}, truncated={truncated}")
            if terminated or truncated:
                obs, info = env.reset()
                print(f"    Reset: new obs shape={obs.shape}")

        print(f"  Total reward over 5 steps: {total_reward:.3f}")
        env.close()
        print("  ✓ cartpole/swingup test passed!")

        # Test with walker (more complex)
        print("\nTesting walker/walk...")
        env = make_env("dmcontrol", task="walker/walk", seed=42)
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        obs, info = env.reset()
        print(f"  Initial observation shape: {obs.shape}")

        # Run a few steps
        for step in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {step+1}: reward={reward:.3f}, obs_shape={obs.shape}")

        env.close()
        print("  ✓ walker/walk test passed!")

        print("\n" + "="*50)
        print("All DMControl tests passed! ✓")
        print("="*50)

    except Exception as e:
        print(f"\n✗ DMControl test failed with error: {e}")
        import traceback
        traceback.print_exc()
