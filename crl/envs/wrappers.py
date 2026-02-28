import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

import metaworld
from metaworld.sawyer_xyz_env import SawyerXYZEnv
from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from metaworld.types import Task

# ==================================================
# Minigrid wrappers
# ==================================================

class ImageWrapper(gym.ObservationWrapper):
    """Wrapper that extracts the 'image' observation from environments."""

    def __init__(self, env):
        super().__init__(env)
        if hasattr(env.observation_space, 'spaces') and 'image' in env.observation_space.spaces:
            self.observation_space = env.observation_space.spaces['image']
        else:
            # Fallback: get shape from first observation
            obs, _ = env.reset()
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs.shape, dtype=np.uint8
            )

    def observation(self, obs):
        """Extract image observation from the full observation dict."""
        if isinstance(obs, dict) and 'image' in obs:
            return obs['image']
        return obs


class ReduceActions(gym.ActionWrapper):
    def __init__(self, env, allowed):
        super().__init__(env)
        self.allowed = list(allowed)  # list of ints in the underlying env's action enum
        self.action_space = gym.spaces.Discrete(len(self.allowed))

    def action(self, a):
        return self.allowed[int(a)]


class FlattenObsWrapper(gym.ObservationWrapper):
    """Flatten multi-dimensional observations to 1D."""
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        flat_dim = int(np.prod(obs_space.shape))
        self.observation_space = gym.spaces.Box(
            low=obs_space.low.flatten()[0] if hasattr(obs_space.low, 'flatten') else obs_space.low,
            high=obs_space.high.flatten()[0] if hasattr(obs_space.high, 'flatten') else obs_space.high,
            shape=(flat_dim,),
            dtype=obs_space.dtype,
        )

    def observation(self, obs):
        return obs.flatten()


class FloatObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = env.observation_space

        assert isinstance(obs_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.astype(np.float32)


class ChannelStandardizationWrapper(gym.ObservationWrapper):
    """Wrapper that standardizes the number of channels across different environments."""

    def __init__(self, env, target_channels=4):
        super().__init__(env)
        self.target_channels = target_channels

        orig_space = env.observation_space
        self.orig_channels = orig_space.shape[2]

        height, width = orig_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(height, width, target_channels),
            dtype=np.float32
        )

    def observation(self, obs):
        """Standardize the channel dimension."""
        height, width, orig_channels = obs.shape

        if orig_channels == self.target_channels:
            # No change needed
            return obs.astype(np.float32)
        elif orig_channels < self.target_channels:
            # Pad with zeros (shouldn't happen if target_channels=4 is minimum)

            padded_obs = np.zeros((height, width, self.target_channels), dtype=np.float32)
            padded_obs[:, :, :orig_channels] = obs.astype(np.float32)
            return padded_obs
        else:
            # Reduce channels using simple strategies
            if orig_channels == 6:  # space_invaders: 6 -> 4
                # Take first 4 channels
                return obs[:, :, :4].astype(np.float32)
            elif orig_channels == 7:  # freeway: 7 -> 4
                # Take first 4 channels
                return obs[:, :, :4].astype(np.float32)
            elif orig_channels == 10:  # seaquest: 10 -> 4
                # Take first 4 channels
                return obs[:, :, :4].astype(np.float32)
            else:
                # Generic case: take first target_channels
                return obs[:, :, :self.target_channels].astype(np.float32)


class BraxTaskTransformWrapper(gym.Env):
    """
    Applies per-task action transforms (mask/inversion) and annotates task info.
    This wrapper is task-static: one wrapped env instance corresponds to one task.

    Inherits from gymnasium.Env (not gymnasium.Wrapper) so it can wrap the old-gym
    brax TorchWrapper without triggering gymnasium's isinstance assertion.
    Always exposes a gymnasium-compatible interface on the outside.
    """

    def __init__(self, env, task_name: str, task_spec: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.task_name = str(task_name)
        self.task_spec = dict(task_spec or {})
        self._action_coefficient = float(self.task_spec.get("action_coefficient", 1.0))
        self._action_mask = self.task_spec.get("action_mask")
        if self._action_mask is not None:
            self._action_mask = np.asarray(self._action_mask, dtype=np.float32)

    def _transform_action(self, action):
        try:
            import torch
            is_torch = isinstance(action, torch.Tensor)
        except ImportError:
            is_torch = False

        a = np.asarray(action.detach().cpu() if is_torch else action, dtype=np.float32)
        if self._action_mask is not None:
            a = a * self._action_mask
        a = self._action_coefficient * a
        if isinstance(self.action_space, spaces.Box):
            a = np.clip(a, self.action_space.low, self.action_space.high)

        if is_torch:
            return torch.tensor(a, dtype=torch.float32)
        return a

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # brax GymWrapper exposes seed() + no-arg reset; gymnasium uses reset(seed=...)
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)
            result = self.env.reset()
        else:
            try:
                result = self.env.reset(seed=seed, options=options)
            except TypeError:
                result = self.env.reset()

        obs, info = (result if isinstance(result, tuple) else (result, {}))
        info = dict(info or {})
        info["task"] = self.task_name
        return obs, info

    def step(self, action):
        result = self.env.step(self._transform_action(action))
        if len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, info = result
        info = dict(info or {})
        info["task"] = self.task_name
        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


class HighwayParkingDistWrapper(gym.Wrapper):
    """
    Wrapper for highway-env parking-v0. Supports two task modes via one interface:
    - dists_spec: task id -> spec with (low, high) ranges; on reset we sample reward params.
    - tasks_spec: task id -> fixed task config; on reset we use that config as-is (no sampling).
    At least one of dists_spec or tasks_spec must be provided. max_episode_steps passed as config duration.
    """

    def __init__(
        self,
        env: gym.Env,
        dists_spec: Optional[Dict[str, Dict[str, Any]]] = None,
        tasks_spec: Optional[Dict[str, Dict[str, Any]]] = None,
        initial_task: Optional[str] = None,
        seed: Optional[int] = None,
        max_episode_steps: Optional[int] = None,
    ):
        super().__init__(env)
        self.dists_spec = dists_spec or {}
        self.tasks_spec = tasks_spec or {}
        if not self.dists_spec and not self.tasks_spec:
            raise ValueError("At least one of dists_spec or tasks_spec must be non-empty")
        all_tasks = list(self.dists_spec) + list(self.tasks_spec)
        self._task_id = initial_task or all_tasks[0]
        if self._task_id not in self.dists_spec and self._task_id not in self.tasks_spec:
            raise ValueError(f"initial_task '{self._task_id}' not in dists_spec or tasks_spec")
        self._rng = np.random.default_rng(seed)
        self.max_episode_steps = max_episode_steps

    def set_task(self, task: str) -> None:
        if task not in self.dists_spec and task not in self.tasks_spec:
            raise ValueError(
                f"Unknown task '{task}'. Known dists: {list(self.dists_spec)}; "
                f"tasks: {list(self.tasks_spec)}"
            )
        self._task_id = task

    def _sample_config(self) -> Dict[str, Any]:
        if self._task_id in self.tasks_spec:
            config = dict(self.tasks_spec[self._task_id])
            if self.max_episode_steps is not None:
                config["duration"] = int(self.max_episode_steps)
            return config

        spec = self.dists_spec[self._task_id]
        rng = self._rng
        config: Dict[str, Any] = {}

        if "reward_weights" in spec:
            rw_spec = spec["reward_weights"]
            if isinstance(rw_spec, (list, tuple)) and len(rw_spec) > 0:
                if isinstance(rw_spec[0], (list, tuple)) and len(rw_spec[0]) == 2:
                    config["reward_weights"] = [float(rng.uniform(a, b)) for (a, b) in rw_spec]
                else:
                    config["reward_weights"] = [float(x) for x in rw_spec]
            else:
                config["reward_weights"] = list(rw_spec)

        for key in ("collision_reward", "success_goal_reward"):
            if key not in spec:
                continue
            v = spec[key]
            if isinstance(v, (list, tuple)) and len(v) == 2:
                config[key] = float(rng.uniform(v[0], v[1]))
            else:
                config[key] = float(v)

        if self.max_episode_steps is not None:
            config["duration"] = int(self.max_episode_steps)
        return config

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        config = self._sample_config()
        opts = dict(options) if options else {}
        opts["config"] = config
        obs, info = self.env.reset(seed=seed, options=opts)
        if info is None:
            info = {}
        info["task"] = self._task_id
        return obs, info


class VectorizeObsWrapper(gym.ObservationWrapper):
    """
    Converts non-vector observations to a single flat vector.
    - If observation_space is Dict: concatenate all subspaces (in sorted key order) into one Box.
    - If observation_space is Box: flatten to 1D.
    Used e.g. for highway-env parking-v0 (KinematicsGoal returns dict[str, ndarray]).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = env.observation_space
        if isinstance(obs_space, spaces.Dict):
            self._keys = sorted(obs_space.spaces.keys())
            lows, highs, shapes = [], [], []
            for k in self._keys:
                sp = obs_space.spaces[k]
                if isinstance(sp, spaces.Box):
                    n = int(np.prod(sp.shape))
                    shapes.append(n)
                    lows.append(np.reshape(sp.low, -1))
                    highs.append(np.reshape(sp.high, -1))
                else:
                    # Discrete or other: sample to get shape, use -inf/inf bounds
                    sample = sp.sample()
                    arr = np.asarray(sample).reshape(-1)
                    n = arr.size
                    shapes.append(n)
                    lows.append(np.full(n, -np.inf, dtype=np.float32))
                    highs.append(np.full(n, np.inf, dtype=np.float32))
            low = np.concatenate(lows).astype(np.float32)
            high = np.concatenate(highs).astype(np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self._is_dict = True
        else:
            assert isinstance(obs_space, spaces.Box), (
                f"VectorizeObsWrapper expects Dict or Box, got {type(obs_space)}"
            )
            self._keys = None
            flat_dim = int(np.prod(obs_space.shape))
            low = np.reshape(obs_space.low, -1).astype(np.float32)
            high = np.reshape(obs_space.high, -1).astype(np.float32)
            self.observation_space = spaces.Box(
                low=low, high=high, shape=(flat_dim,), dtype=np.float32
            )
            self._is_dict = False

    def observation(self, obs):
        if self._is_dict:
            parts = [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in self._keys]
            return np.concatenate(parts, axis=0)
        return np.asarray(obs, dtype=np.float32).reshape(-1)


class DictToVectorObs(gym.ObservationWrapper):
    def __init__(self, env, keys):
        super().__init__(env)
        self.keys = keys

        # Build a flat Box space by summing flattened sizes.
        # This assumes each selected subspace is a Box with fixed shape.
        flat_size = 0
        lows = []
        highs = []
        for k in self.keys:
            sp = env.observation_space[k]
            assert isinstance(sp, spaces.Box), f"{k} must be Box, got {type(sp)}"
            n = int(np.prod(sp.shape))
            flat_size += n
            lows.append(np.reshape(sp.low, -1))
            highs.append(np.reshape(sp.high, -1))

        low = np.concatenate(lows).astype(np.float32)
        high = np.concatenate(highs).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        parts = []
        for k in self.keys:
            x = obs[k]
            # Convert bytes/ints to float32 for NN-friendly vector
            parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
        return np.concatenate(parts, axis=0)


class MetaworldTaskSwitcher(gym.Wrapper):
    def __init__(self, env: SawyerXYZEnv, tasks: list[Task]):
        super().__init__(env)
        if not tasks:
            raise ValueError("The 'tasks' list cannot be empty.")
        self.tasks = tasks
        self.num_tasks = len(tasks)

        assert isinstance(env.unwrapped, SawyerXYZEnv)
        self._unwrapped_env: SawyerXYZEnv = env.unwrapped

        if self._unwrapped_env._random_reset_space is not None:
            self._unwrapped_env._last_rand_vec = self._unwrapped_env._random_reset_space.sample()
        self._unwrapped_env._freeze_rand_vec = True

        self._current_task_idx: int = 0
        self._current_task_obj: Task = self.tasks[0]
        self.set_task(0)

        self._unwrapped_env.set_task(self._current_task_obj)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info is None:
            info = {}
        info["task_idx"] = self._current_task_idx
        info["task_obj"] = self._current_task_obj
        return obs, reward, terminated, truncated, info

    def set_task(self, task_idx: int | str):
        """Manually sets the active task by its index (int or string, e.g. \"0\")."""
        task_idx = int(task_idx) if isinstance(task_idx, str) else task_idx
        if not (0 <= task_idx < self.num_tasks):
            raise IndexError(f"Task index {task_idx} is out of bounds for {self.num_tasks} tasks.")

        self._current_task_idx = task_idx
        self._current_task_obj = self.tasks[task_idx]

        self._unwrapped_env.set_task(self._current_task_obj)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed, options=options)

        obs, info = self._unwrapped_env.reset(seed=seed, options=options)
        if info is None:
            info = {}
        info['task_idx'] = self._current_task_idx
        info['task_obj'] = self._current_task_obj

        return obs, info


# ==================================================
# MJX (mujoco_playground) gymnasium adapter
# ==================================================

class MjxGymnasiumWrapper(gym.Env):
    """Wraps a mujoco_playground MjxEnv with a standard gymnasium interface.

    MjxEnv API (pure JAX, stateful):
        state = env.reset(rng: jax.Array)
        state = env.step(state, action: jax.Array)

    This wrapper stores the JAX state internally, JIT-compiles reset/step on
    first call, and converts observations/actions between numpy and JAX.
    """

    def __init__(self, mjx_env, seed: int = 0, max_episode_steps: int = 1000):
        import jax
        super().__init__()
        self._env = mjx_env
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        self._state = None
        self._key = jax.random.PRNGKey(seed)

        # JIT-compile for speed (first call triggers compilation)
        self._jit_reset = jax.jit(mjx_env.reset)
        self._jit_step  = jax.jit(mjx_env.step)

        action_size = mjx_env.action_size
        obs_size    = mjx_env.observation_size
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(action_size,),  dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        import jax
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, rng = jax.random.split(self._key)
        self._state = self._jit_reset(rng)
        self._step_count = 0
        obs = np.asarray(self._state.obs, dtype=np.float32)
        return obs, {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        import jax.numpy as jnp
        action_jax = jnp.asarray(action, dtype=jnp.float32)
        self._state = self._jit_step(self._state, action_jax)
        obs        = np.asarray(self._state.obs,    dtype=np.float32)
        reward     = float(self._state.reward)
        terminated = bool(self._state.done)
        self._step_count += 1
        truncated  = self._step_count >= self._max_episode_steps
        info       = {k: float(v) for k, v in self._state.metrics.items()}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

