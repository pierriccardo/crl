# crl/envs/factory.py
from __future__ import annotations
import itertools
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from minigrid.wrappers import FlatObsWrapper
from minigrid.core.actions import Actions

from minihack import RewardManager
from minihack.reward_manager import Event
from crl.envs.adapters import DenseCoordEvent, lvl_gen

import metaworld
from metaworld.envs import (
    SawyerReachEnvV3,
    SawyerDrawerCloseEnvV3,
    SawyerHammerEnvV3
)

# Environments, adapters and wrappers
from crl.envs.adapters import MultiTaskGridEnv, DMControlEnv, SimpleDistAsTaskEnv
from crl.envs.tasks import (
    PARKING_DISTS_SPECS,
    PARKING_TASKS_SPECS,
    SYNTHETIC_DISTS_SPECS,
    HUMANOID_TASK_BY_NAME,
)
from crl.envs.wrappers import (
    ReduceActions,
    ContinualEpisodicWrapper,
    FloatObsWrapper,
    FlattenObsWrapper,
    ImageWrapper,
    DictToVectorObs,
    MetaworldTaskSwitcher,
    HighwayParkingDistWrapper,
    VectorizeObsWrapper,
    HumanoidTaskWrapper,
)

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

@_register("switchingdist")
def _driftw_env(**kwargs) -> gym.Env:
    # Use SYNTHETIC_DISTS_SPECS as default if dist_specs not provided
    if "dist_specs" not in kwargs:
        kwargs["dist_specs"] = SYNTHETIC_DISTS_SPECS
    return SimpleDistAsTaskEnv(**kwargs)


@_register("minigrid")
def _multitaskgrid_env(**kwargs) -> gym.Env:
    """Single task goal environment with configurable task."""
    # Extract task parameter
    task = kwargs.pop('task', None)

    if task is None:
        task = "reachgreen"

    env = MultiTaskGridEnv(task_name=task, **kwargs)

    allowed = [Actions.left, Actions.right, Actions.forward, Actions.pickup]
    env = ReduceActions(env=env, allowed=allowed)

    env = ImageWrapper(env=env)
    env = FlattenObsWrapper(env=env)
    env = FloatObsWrapper(env=env)

    return env


@_register("dmcontrol")
def _dmcontrol_env(**kwargs) -> gym.Env:
    """DeepMind Control Suite environment with configurable domain and task."""
    task = kwargs.pop('task', None)
    domain = kwargs.pop('domain', None)

    if domain is None:
        domain = 'cartpole'
    if task is None:
        task = 'swingup'

    kwargs.pop("max_episode_steps", None)
    kwargs.pop("_continual_task_list_len", None)
    kwargs.pop("seed", None)  # DMControlEnv has no seed in __init__; make_env seeds via reset()
    env = DMControlEnv(domain=domain, task=task, **kwargs)
    return env


@_register("minihack")
def _minihack_env(**kwargs) -> gym.Env:
    task = kwargs.pop('task', None)
    max_episode_steps = kwargs.pop("max_episode_steps", None)
    coordinates = task.split("_")[1:]
    coordinates = tuple(int(c) for c in coordinates)

    rm = RewardManager()
    rm.add_event(DenseCoordEvent(coordinates=coordinates))

    make_kwargs = dict(
        des_file=lvl_gen.get_des(),
        observation_keys=("glyphs_crop", "blstats"),
        reward_manager=rm,
    )
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make("MiniHack-Skill-Custom-v0", **make_kwargs)

    env = DictToVectorObs(env, keys=["blstats"])

    return env


@_register("metaworld")
def _metaworld_env(**kwargs) -> gym.Env:
    # env_id is "metaworld/<env_name>" e.g. metaworld/reach-v3
    env_id = kwargs.pop("env_id", "metaworld/reach-v3")
    env_name = env_id.split("/", 1)[1]
    assert env_name in ["reach-v3", "drawer-close-v3", "hammer-v3"], f"Unknown environment name: {env_name}"

    # _continual_task_list_len: in continual mode we take that many tasks (task ids "0".."n-1").
    # task: initial task index (e.g. "0") when creating a single env for eval; do NOT slice tasks by it.
    n_tasks = kwargs.get("_continual_task_list_len")
    task_param = kwargs.pop("task", None)
    seed = kwargs.pop("seed", None)
    if seed is None:
        seed = 0  # MT1 expects int; use 0 so benchmark is at least deterministic

    envs = {
        "reach-v3": SawyerReachEnvV3(),
        "drawer-close-v3": SawyerDrawerCloseEnvV3(),
        "hammer-v3": SawyerHammerEnvV3(),
    }

    assert env_name in envs.keys(), f"Unknown environment name: {env_name}"

    benchmark = metaworld.MT1(env_name, seed=seed)
    tasks = [t for t in benchmark.train_tasks if t.env_name == env_name]
    if n_tasks is not None:
        tasks = tasks[:n_tasks]
    # else: use all tasks (e.g. 50); task_param is only the initial task index for set_task

    if not tasks:
        raise ValueError(f"MetaWorld benchmark produced no tasks for env_name={env_name!r}.")

    base_env = envs[env_name]
    env = MetaworldTaskSwitcher(base_env, tasks)
    if task_param is not None:
        env.set_task(int(task_param) if isinstance(task_param, str) else task_param)
    return env


@_register("highway_parking")
def _highway_parking_env(**kwargs) -> gym.Env:
    try:
        import highway_env  # noqa: F401
    except ImportError:
        raise ImportError("highway_env is required for highway_parking. Install with: pip install highway-env")
    task = kwargs.pop("task", "park:0")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else "park:0"
    task = str(task)
    dists_spec = kwargs.pop("dists_spec", None) or PARKING_DISTS_SPECS
    tasks_spec = {tid: spec["config"] for tid, spec in PARKING_TASKS_SPECS.items()}
    max_episode_steps = kwargs.pop("max_episode_steps", None)
    kwargs.pop("_continual_task_list_len", None)  # internal; parking-v0 doesn't accept it
    kwargs.pop("seed", None)  # ParkingEnv doesn't take seed in __init__; make_env seeds via reset()
    kwargs.pop("render_mode", None)
    env = gym.make("parking-v0", **kwargs)
    env = HighwayParkingDistWrapper(
        env,
        dists_spec=dists_spec,
        tasks_spec=tasks_spec,
        initial_task=task,
        max_episode_steps=max_episode_steps,
    )
    # parking-v0 uses KinematicsGoal (dict obs); vectorize for algo compatibility
    env = VectorizeObsWrapper(env)
    return env


# Gymnasium MuJoCo Walker2d: same dynamics, task = reward config (forward_reward_weight, ctrl_cost_weight, healthy_reward)
WALKER2D_REWARD_TASKS = {
    "default": {"forward_reward_weight": 1.0, "ctrl_cost_weight": 1e-3, "healthy_reward": 1.0},
    "forward_heavy": {"forward_reward_weight": 2.0, "ctrl_cost_weight": 1e-3, "healthy_reward": 0.5},
    "low_ctrl": {"forward_reward_weight": 1.0, "ctrl_cost_weight": 1e-4, "healthy_reward": 1.0},
    "survive": {"forward_reward_weight": 0.5, "ctrl_cost_weight": 1e-3, "healthy_reward": 2.0},
}


@_register("mujoco")
def _mujoco_walker2d_env(**kwargs) -> gym.Env:
    """Gymnasium MuJoCo Walker2d. env_id='mujoco/walker2d', task=reward config (default|forward_heavy|low_ctrl|survive)."""
    env_id = kwargs.pop("env_id", "mujoco/walker2d")
    task = kwargs.pop("task", "default")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else "default"
    task = str(task).lower()
    if task not in WALKER2D_REWARD_TASKS:
        raise ValueError(f"Unknown task {task!r}. Known: {list(WALKER2D_REWARD_TASKS.keys())}")
    reward_kwargs = dict(WALKER2D_REWARD_TASKS[task])
    kwargs.pop("max_episode_steps", None)
    kwargs.pop("_continual_task_list_len", None)
    kwargs.pop("seed", None)
    env = gym.make("Walker2d-v4", **reward_kwargs, **kwargs)
    return env


# Gymnasium MuJoCo Humanoid: task = spec from HUMANOID_TASKS_SPECS (stand, walk_forward, run_forward, etc.)
@_register("mujoco_humanoid")
def _mujoco_humanoid_env(**kwargs) -> gym.Env:
    """Gymnasium MuJoCo Humanoid with HumanoidTaskWrapper. env_id='mujoco/humanoid', task=name from crl.envs.tasks (e.g. stand, walk_forward, run_forward)."""
    env_id = kwargs.pop("env_id", "mujoco/humanoid")
    task = kwargs.pop("task", "stand")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else "stand"
    task = str(task)
    if task not in HUMANOID_TASK_BY_NAME:
        known = ", ".join(sorted(HUMANOID_TASK_BY_NAME.keys()))
        raise ValueError(f"Unknown humanoid task {task!r}. Known: {known}")
    task_spec = HUMANOID_TASK_BY_NAME[task]
    env_kwargs = dict(task_spec.get("env_kwargs_override", {}))
    kwargs.pop("max_episode_steps", None)
    kwargs.pop("_continual_task_list_len", None)
    kwargs.pop("seed", None)
    env = gym.make("Humanoid-v5", **env_kwargs, **kwargs)
    env = HumanoidTaskWrapper(env, task_spec)
    return env


# ============================================
# Task sequence registrations
# ============================================
#
# _register_task_sequence(ENV_NAME, LIST_NAME, TASKS)
#

# SimpleDistAsTaskEnv sequences
_register_task_sequence("switchingdist", "basic", ["dist:0", "dist:1", "dist:2"])

# Highway parking (parameterized reward distributions)
_register_task_sequence("highway_parking", "dists_basic", ["park:0", "park:1", "park:2"])
_register_task_sequence("highway_parking", "dists_full", ["park:0", "park:1", "park:2", "park:3", "park:4", "park:5", "park:6"])
_register_task_sequence("highway_parking", "tasks_basic", ["task_1", "task_2", "task_3", "task_4", "task_5"])
_register_task_sequence("highway_parking", "tasks_full", ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8", "task_9", "task_10"])


# GoalEnv sequences
_register_task_sequence("minigrid", "basic", ["reachgreen", "reachblue", "pickupkey"])

# DMControl sequences (new format: dmc/<domain> with simple task names)
_register_task_sequence("dmc/walker", "basic", ["stand", "walk"])
_register_task_sequence("dmc/walker", "full", ["stand", "walk", "run"])
_register_task_sequence("dmc/cartpole", "basic", ["swingup", "balance"])
_register_task_sequence("dmc/cartpole", "full", ["swingup", "balance", "balance_sparse"])

# MiniHack sequences
_register_task_sequence("minihack", "goals", [f"g_{i}_{j}" for i, j in itertools.product(range(10), range(10))])

# metaworld sequences
for env_name in ["reach-v3", "drawer-close-v3", "hammer-v3"]:
    for task_num in [10, 20, 30, 40, 50]:
        _register_task_sequence(f"metaworld/{env_name}", f"{task_num}", [str(i) for i in range(task_num)])

# MuJoCo Walker2d: task = reward config (same dynamics, different reward weights)
_register_task_sequence("mujoco/walker2d", "basic", ["default", "forward_heavy", "low_ctrl"])
_register_task_sequence("mujoco/walker2d", "full", ["default", "forward_heavy", "low_ctrl", "survive"])

# MuJoCo Humanoid: task = spec name from crl.envs.tasks (HUMANOID_TASKS_SPECS)
_HUMANOID_TASK_NAMES = list(HUMANOID_TASK_BY_NAME.keys())
_register_task_sequence("mujoco/humanoid", "basic", ["stand", "walk_forward", "run_forward", "crouch"])
_register_task_sequence("mujoco/humanoid", "full", _HUMANOID_TASK_NAMES)

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

    - env_id: registry key (case-insensitive), e.g. "minigrid", "switchingdist",
      "dmc/walker", "metaworld/reach-v3". For dmc use "dmc/<domain>" and pass task=...
    - seed, render_mode, record_stats, wrappers: applied consistently.
    - env_kwargs: passed through to the backend constructor (task, max_episode_steps,
      etc.). No global injection of domain_name; each backend receives only what it needs.
    """
    key = env_id.lower()
    # Only set kwargs that are common or backend-specific; do not inject domain_name for all.
    env_kwargs.setdefault("seed", seed)
    env_kwargs.setdefault("render_mode", render_mode)

    if key.startswith("metaworld/"):
        env_kwargs["env_id"] = env_id  # so _metaworld_env can parse "metaworld/reach-v3"
        key = "metaworld"

    if key.startswith("dmc/"):
        domain = key.split("/", 1)[1]
        if "dmcontrol" not in _REGISTRY:
            known = ", ".join(sorted(_REGISTRY.keys()))
            raise ValueError(f"DMControl not available. Known envs: {known}")
        env_kwargs["domain"] = domain
        key = "dmcontrol"

    if key == "mujoco/humanoid":
        env_kwargs["env_id"] = env_id
        key = "mujoco_humanoid"
    elif key.startswith("mujoco/"):
        env_kwargs["env_id"] = env_id
        key = "mujoco"

    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown env_id '{env_id}'. Known: {known}")

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
    # if record_stats:
    #     env = RecordEpisodeStatistics(env)  # Commented out - not needed

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
    task_list: str | list[str],
    max_episode_steps: int = 100,
    task_switch_prob: float = 1.0,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    record_stats: bool = True,
    wrappers: Optional[list[Callable[[gym.Env], gym.Env]]] = None,
    **env_kwargs: Any,
) -> gym.Env:
    """
    Create a continual episodic environment: one env from make_env, wrapped with
    ContinualEpisodicWrapper. The underlying env should expose set_task(task) or
    .task so the wrapper can switch tasks on reset; if not, a per-task env_factory
    is used (e.g. for MiniHack).

    - Each episode lasts for exactly N timesteps (max_episode_steps).
    - A task is selected at the start of each episode with task_switch_prob.
    - task_list: list of task ids, or a string sequence name to look up.

    env_kwargs are forwarded to make_env (e.g. max_episode_steps, _continual_task_list_len
    for metaworld). No task_list is passed into make_env.
    """

    # Resolve task_list if it's a string (task sequence name)
    if isinstance(task_list, str):
        task_list = get_task_sequence(env_id, task_list)

    # Kwargs for the single env: only what make_env needs (no task_list).
    # _continual_task_list_len is used by metaworld to build the right number of tasks.
    base_kwargs = {
        **env_kwargs,
        "max_episode_steps": max_episode_steps,
        "_continual_task_list_len": len(task_list),
    }

    # Single env with initial task.
    env = make_env(
        env_id,
        task=task_list[0],
        seed=seed,
        render_mode=render_mode,
        record_stats=False,
        wrappers=wrappers,
        **base_kwargs,
    )

    # Use env_factory only when env cannot switch task in-place (no set_task / .task).
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    has_set_task = callable(getattr(env, "set_task", None)) or callable(getattr(base, "set_task", None))
    has_task_attr = hasattr(env, "task") or hasattr(base, "task")
    if has_set_task or has_task_attr:
        env_factory = None
    else:
        def env_factory(task: str) -> gym.Env:
            return make_env(
                env_id, task=task, seed=seed, render_mode=render_mode,
                record_stats=False, wrappers=wrappers, **base_kwargs,
            )

    # Wrap with ContinualEpisodicWrapper
    env = ContinualEpisodicWrapper(
        env=env,
        task_list=task_list,
        max_episode_steps=max_episode_steps,
        env_factory=env_factory,
        task_switch_prob=task_switch_prob,
        seed=seed,
    )

    # if record_stats:
    #     env = RecordEpisodeStatistics(env)

    return env


if __name__ == "__main__":
    print("=" * 60)
    print("Environments")
    print("=" * 60)
    envs = list_envs()
    for env_id in envs:
        print(f"  {env_id}")

    print("\n" + "=" * 60)
    print("Task sequences (env -> sequence_name -> task list)")
    print("=" * 60)
    all_sequences = list_task_sequences()
    for env_name in sorted(all_sequences.keys()):
        print(f"\n  {env_name}:")
        for seq_name, tasks in sorted(all_sequences[env_name].items()):
            n = len(tasks)
            preview = tasks[:5] if n > 5 else tasks
            suffix = f" ... ({n} total)" if n > 5 else f" ({n} tasks)"
            print(f"    {seq_name}: {preview}{suffix}")
