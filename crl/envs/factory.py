import gymnasium as gym
try:
    from gymnasium.wrappers.vector import RecordEpisodeStatistics as GymRecordEpisodeStatistics
except ImportError:
    from gymnasium.wrappers.vector.common import RecordEpisodeStatistics as GymRecordEpisodeStatistics

# Envs dependencies
from minigrid.core.actions import Actions

# Environments, adapters and wrappers
from crl.envs.adapters import MultiTaskGridEnv
from crl.envs.continualworld import (
    cw10_v3,
    cw20_v3,
    make_continualworld_env,
)
from crl.envs.tasks import (
    MJX_TASKS_SPECS,
    MJX_SCENARIO_SEQUENCES,
)
from crl.envs.wrappers import (
    ReduceActions,
    FloatObsWrapper,
    FlattenObsWrapper,
    ImageWrapper,
    HighwayParkingDistWrapper,
    VectorizeObsWrapper,
    RecordEpisodeStatistics as CustomRecordEpisodeStatistics,
)
from crl.envs.mjx_playground_wrapper import MjxPlaygroundGymWrapper, MjxPlaygroundStdWrapper

from typing import Callable, Dict, Optional

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


@_register("minigrid")
def _multitaskgrid_env(task: str = "reachgreen", seed: int = 0, max_episode_steps: int = 1000) -> gym.Env:
    env = MultiTaskGridEnv(task_name=task)
    env = ReduceActions(env=env, allowed=[Actions.left, Actions.right, Actions.forward, Actions.pickup])
    env = ImageWrapper(env=env)
    env = FlattenObsWrapper(env=env)
    env = FloatObsWrapper(env=env)
    return env


@_register("continualworld")
def _continualworld_env(task: str = "hammer-v3", seed: int = 0, max_episode_steps: int = 1000) -> gym.Env:
    return make_continualworld_env(task=task, seed=seed, max_episode_steps=max_episode_steps)


@_register("highway_parking")
def _highway_parking_env(task: str = "park:0", seed: int = 0, max_episode_steps: int = 1000) -> gym.Env:
    try:
        import highway_env  # noqa: F401
    except ImportError:
        raise ImportError("highway_env is required. Install with: pip install highway-env")
    tasks_spec = {tid: spec["config"] for tid, spec in PARKING_TASKS_SPECS.items()}
    env = gym.make("parking-v0")
    env = HighwayParkingDistWrapper(
        env,
        dists_spec=PARKING_DISTS_SPECS,
        tasks_spec=tasks_spec,
        initial_task=task,
        max_episode_steps=max_episode_steps,
    )
    env = VectorizeObsWrapper(env)
    return env



def _resolve_mjx_task(env_id: str, task: str) -> dict:
    """Resolve MJX domain + task to the task_spec dict.

    Raises ValueError for unknown domains or tasks.
    """
    domain = env_id.split("/", 1)[1].lower()
    if domain not in MJX_TASKS_SPECS:
        raise ValueError(f"Unknown MJX domain {domain!r}. Known: {', '.join(sorted(MJX_TASKS_SPECS))}")
    task_specs = MJX_TASKS_SPECS[domain]
    if task not in task_specs:
        raise ValueError(f"Unknown MJX task {task!r} for {domain!r}. Known: {', '.join(sorted(task_specs))}")
    return dict(task_specs[task])


def _apply_physics_overrides(mjx_env, task_spec: dict) -> None:
    """Mutate *mjx_env* model parameters in-place according to *task_spec*.

    Must be called **before** the first ``jax.jit`` call so the JIT
    compilation bakes in the modified model (``jax.jit`` is lazy).
    """
    import numpy as np
    try:
        from mujoco import mjx as mujoco_mjx
    except ImportError as e:
        raise ImportError("MJX support requires `mujoco`. Try: pip install mujoco") from e

    modified = False
    if "gravity" in task_spec:
        mjx_env._mj_model.opt.gravity[:] = np.array([0.0, 0.0, -9.81 * task_spec["gravity"]])
        modified = True
    if "friction" in task_spec:
        mjx_env._mj_model.geom_friction[:, 0] = task_spec["friction"]
        modified = True
    if modified:
        mjx_env._mjx_model = mujoco_mjx.put_model(mjx_env._mj_model)


@_register("mjx")
def _mjx_env(env_id: str, task: str = "run", seed: int = 0, max_episode_steps: int = 1000) -> gym.Env:
    task_spec = _resolve_mjx_task(env_id, task)
    env = MjxPlaygroundGymWrapper(
        env_name=task_spec["env_name"],
        num_envs=1,
        seed=seed,
        device="cpu",
        max_episode_steps=max_episode_steps,
    )
    _apply_physics_overrides(env.env, task_spec)
    return MjxPlaygroundStdWrapper(
        env,
        task_name=task,
        action_coefficient=float(task_spec.get("action_coefficient", 1.0)),
        action_mask=task_spec.get("action_mask"),
    )


# ============================================
# Task sequence registrations
# ============================================
#
# _register_task_sequence(ENV_NAME, LIST_NAME, TASKS)
#

# Highway parking (parameterized reward distributions)
_register_task_sequence("highway_parking", "dists_basic", ["park:0", "park:1", "park:2"])
_register_task_sequence("highway_parking", "dists_full", ["park:0", "park:1", "park:2", "park:3", "park:4", "park:5", "park:6"])
_register_task_sequence("highway_parking", "tasks_basic", ["task_1", "task_2", "task_3", "task_4", "task_5"])
_register_task_sequence("highway_parking", "tasks_full", ["task_1", "task_2", "task_3", "task_4", "task_5", "task_6", "task_7", "task_8", "task_9", "task_10"])


# GoalEnv sequences
_register_task_sequence("minigrid", "basic", ["reachgreen", "reachblue", "pickupkey"])

# MJX scenarios
for _env_name, _seqs in MJX_SCENARIO_SEQUENCES.items():
    for _seq_name, _tasks in _seqs.items():
        _register_task_sequence(_env_name, _seq_name, _tasks)
    _all_tasks = sorted(list({t for ts in _seqs.values() for t in ts}))
    _register_task_sequence(_env_name, "full", _all_tasks)

# Continual World sequences (Meta-World task names)
_CW10_V3 = cw10_v3()
_CW20_V3 = cw20_v3()
_register_task_sequence("continualworld", "cw10", _CW10_V3)
_register_task_sequence("continualworld", "cw20", _CW20_V3)

# ============================================
# Exposed public methods
# ============================================


def make_vec_env(
    env_id: str,
    task: str = "default",
    seed: int = 0,
    max_episode_steps: int = 1000,
    num_envs: int = 4,
    mode: str = "sync",
    torch_device: Optional[str] = None,
):
    """Create multiple parallel environments.

    For **MJX** environments this returns a :class:`MjxPlaygroundStdWrapper`
    wrapping :class:`MjxPlaygroundGymWrapper` with ``jax.vmap`` for native
    GPU batching -- a single JIT compilation runs all *N* envs in one
    kernel (the ``mode`` argument is ignored).  The wrapper handles
    auto-reset, ``final_observation`` storage, and per-task action transforms.

    Pass ``torch_device`` (e.g. ``"cuda"``) to keep data on GPU and use
    DLPack zero-copy transfers between JAX and PyTorch.  In this mode
    ``step()`` accepts/returns **torch tensors** instead of numpy arrays.

    For all other environments it falls back to Gymnasium's
    ``SyncVectorEnv`` / ``AsyncVectorEnv`` (``torch_device`` is ignored).

    Args:
        env_id: Registry key, e.g. ``"mjx/cheetah"``, ``"continualworld"``.
        task: Task name within the environment.
        seed: Base random seed.
        max_episode_steps: Episode length / horizon.
        num_envs: Number of parallel environments.
        mode: ``"sync"`` or ``"async"`` (non-MJX envs only).
        torch_device: If set, MJX envs return torch tensors on this device.
    """
    if env_id.lower().startswith("mjx/"):
        task_spec = _resolve_mjx_task(env_id, task)
        device = torch_device or "cuda"
        env = MjxPlaygroundGymWrapper(
            env_name=task_spec["env_name"],
            num_envs=num_envs,
            seed=seed,
            device=device,
            max_episode_steps=max_episode_steps,
        )
        _apply_physics_overrides(env.env, task_spec)
        wrapped = MjxPlaygroundStdWrapper(
            env,
            task_name=task,
            action_coefficient=float(task_spec.get("action_coefficient", 1.0)),
            action_mask=task_spec.get("action_mask"),
        )
        return CustomRecordEpisodeStatistics(wrapped)

    def _make_fn(idx: int):
        def _init() -> gym.Env:
            return make_env(env_id, task=task, seed=seed + idx, max_episode_steps=max_episode_steps)
        return _init

    env_fns = [_make_fn(i) for i in range(num_envs)]
    if mode == "async":
        return GymRecordEpisodeStatistics(gym.vector.AsyncVectorEnv(env_fns))
    return GymRecordEpisodeStatistics(gym.vector.SyncVectorEnv(env_fns))


def make_env(
    env_id: str,
    task: str = "default",
    seed: int = 0,
    max_episode_steps: int = 1000,
) -> gym.Env:
    """Create a single-task environment.

    Args:
        env_id: Registry key, e.g. "brax/halfcheetah", "continualworld", "minigrid".
        task: Task name within the environment.
        seed: Random seed.
        max_episode_steps: Episode length / horizon.
    """
    key = env_id.lower()

    if key.startswith("mjx/"):
        return _mjx_env(env_id=env_id, task=task, seed=seed, max_episode_steps=max_episode_steps)

    if key in _REGISTRY:
        return _REGISTRY[key](task=task, seed=seed, max_episode_steps=max_episode_steps)

    # Fallback: plain Gymnasium env (e.g. "CartPole-v1", "Ant-v5")
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def get_task_sequence(env_name: str, sequence_name: str) -> list[str]:
    """Retrieve a task sequence by environment and sequence name.

    If sequence_name equals env_name (e.g. both 'mjx/humanoid'), treats it as a request
    for the default sequence and returns 'default', then 'full', then the first available.
    """
    env_key = env_name.lower()
    sequence_key = sequence_name.lower()

    if env_key not in _TASK_SEQUENCES:
        raise ValueError(f"No task sequences registered for environment '{env_name}'")

    seqs = _TASK_SEQUENCES[env_key]
    if sequence_key in seqs:
        return seqs[sequence_key]

    if sequence_key != env_key:
        available = ", ".join(sorted(seqs.keys()))
        raise ValueError(f"Unknown sequence '{sequence_name}' for env '{env_name}'. Available: {available}")

    # sequence_name was the same as env_name (e.g. --env.task_list=mjx/humanoid): use default
    for fallback in ("default", "full"):
        if fallback in seqs:
            return seqs[fallback]
    return seqs[next(iter(seqs))]


def list_task_sequences(env_name: Optional[str] = None) -> Dict[str, list[str]] | Dict[str, Dict[str, list[str]]]:
    """List task sequences for an env, or for all envs.

    Args:
        env_name: If set, return sequences for this environment only; otherwise all envs.

    Returns:
        If env_name is set: dict mapping sequence_name -> list of task ids for that env.
        If env_name is None: dict mapping env_name -> { sequence_name -> list of task ids }.
    """
    if env_name is None:
        return {k: dict(v) for k, v in _TASK_SEQUENCES.items()}
    env_key = env_name.lower()
    if env_key not in _TASK_SEQUENCES:
        return {}
    return dict(_TASK_SEQUENCES[env_key])


def list_all_sequences() -> list[tuple[str, str]]:
    """Return all (env_name, sequence_name) pairs for iteration or display."""
    out = []
    for env_key in sorted(_TASK_SEQUENCES.keys()):
        for seq_key in sorted(_TASK_SEQUENCES[env_key].keys()):
            out.append((env_key, seq_key))
    return out


if __name__ == "__main__":
    import numpy as np

    ENV_ID  = "mjx/cheetah"
    TASK    = "normal"
    SEED    = 0
    N_STEPS = 5

    print(f"Creating {ENV_ID!r} task={TASK!r} ...")
    env = make_env(env_id=ENV_ID, task=TASK, seed=SEED, max_episode_steps=1000)

    obs, info = env.reset(seed=SEED)
    print(f"  obs shape  : {obs.shape}  dtype={obs.dtype}")
    print(f"  action dim : {env.action_space.shape}")
    print(f"  task       : {info['task']}")

    for i in range(N_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step {i+1:02d} | reward={reward:.4f}  done={terminated or truncated}")
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Done.")
