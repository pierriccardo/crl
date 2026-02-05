import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Callable, Dict, Any

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


class ContinualEpisodicWrapper(gym.Wrapper):
    """
    Continual episodic setting: fixed episode length, random task at each reset
    with given switch probability. Manages task switching and truncation.
    """

    def __init__(
        self,
        env: gym.Env,
        task_list: List[str],
        max_episode_steps: int,
        env_factory: Optional[Callable[[str], gym.Env]] = None,
        task_switch_prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(env)
        if not task_list:
            raise ValueError("task_list must contain at least one task")
        if not 0.0 <= task_switch_prob <= 1.0:
            raise ValueError(f"task_switch_prob must be in [0.0, 1.0], got {task_switch_prob}")

        self.task_list = list(task_list)
        self.max_episode_steps = max_episode_steps
        self.env_factory = env_factory
        self.task_switch_prob = task_switch_prob
        self.rng = np.random.RandomState(seed)
        self.episode_timesteps = 0
        self.current_task = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if self.current_task is None or self.rng.random() < self.task_switch_prob:
            self.current_task = self.rng.choice(self.task_list)

        if self.env_factory is not None:
            old_env = self.env
            self.env = self.env_factory(self.current_task)
            if hasattr(old_env, "close"):
                old_env.close()
        else:
            if hasattr(self.env, "set_task"):
                self.env.set_task(self.current_task)
            elif hasattr(self.env, "task"):
                self.env.task = self.current_task

        obs, info = self.env.reset(seed=seed, options=options)
        self.episode_timesteps = 0
        info = info or {}
        info["task"] = self.current_task
        info["max_episode_steps"] = self.max_episode_steps
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_timesteps += 1
        if self.episode_timesteps >= self.max_episode_steps:
            truncated = True
        info = info or {}
        info["task"] = self.current_task
        info["episode_timestep"] = self.episode_timesteps
        return obs, reward, terminated, truncated, info

    def get_task_id(self) -> int:
        """Index of current task in task_list, or -1 if not set."""
        if self.current_task is None:
            return -1
        try:
            return self.task_list.index(self.current_task)
        except ValueError:
            return -1

    def get_task_name(self) -> str:
        """Current task id string, or 'unknown' if not set."""
        return self.current_task if self.current_task is not None else "unknown"


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
            raise ValueError(f"Unknown task '{task}'. Known dists: {list(self.dists_spec)}; tasks: {list(self.tasks_spec)}")
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

# ==========================================================================================
# Humanoid task wrapper
# ==========================================================================================

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from crl.envs.tasks import HUMANOID_TASKS_SPECS
import gymnasium as gym
import numpy as np


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def compute_com_xy(env) -> np.ndarray:
    # MuJoCo: center of mass of subtree rooted at torso usually works.
    # This is a simple, robust proxy: use torso body COM position.
    # If torso name differs, adjust once after printing body names.
    m = env.unwrapped.model
    d = env.unwrapped.data
    torso_id = m.body("torso").id if hasattr(m, "body") else 1
    return np.array(d.xipos[torso_id, :2], dtype=np.float64)


@dataclass
class TaskState:
    prev_com_xy: Optional[np.ndarray] = None
    min_pose_dist: float = float("inf")


class HumanoidTaskWrapper(gym.Wrapper):
    """
    Adds an auxiliary task reward r_task to the environment reward, and logs metrics in info.
    Set add_to_reward=False to log only (evaluation mode).
    """
    def __init__(self, env: gym.Env, task_spec: Dict[str, Any], add_to_reward: bool = True):
        super().__init__(env)
        self.task_spec = task_spec
        self.add_to_reward = add_to_reward
        self.state = TaskState()

        self.dt = float(getattr(env.unwrapped, "dt", 0.01))  # fallback

        obj = self.task_spec["objective"]
        self.obj_type = obj["type"]

        # Pose objective optional
        self.pose_target = None
        if self.obj_type == "qpos_pose_distance":
            self.pose_target = np.load(obj["pose_path"]).astype(np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.state = TaskState()
        self.state.prev_com_xy = compute_com_xy(self.env)
        info = dict(info)
        info.update({"task_name": self.task_spec["name"]})
        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        info = dict(info)

        # signals
        com_xy = compute_com_xy(self.env)
        v_xy = (com_xy - self.state.prev_com_xy) / self.dt
        self.state.prev_com_xy = com_xy

        qpos = np.asarray(self.env.unwrapped.data.qpos, dtype=np.float64)
        qvel = np.asarray(self.env.unwrapped.data.qvel, dtype=np.float64)
        height = float(qpos[2])

        r_task, task_metrics = self._task_reward_and_metrics(v_xy, height, qpos, qvel, action)
        info.update(task_metrics)

        r_total = float(r_env + r_task) if self.add_to_reward else float(r_env)
        return obs, r_total, terminated, truncated, info

    def _task_reward_and_metrics(
        self,
        v_xy: np.ndarray,
        height: float,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        obj = self.task_spec["objective"]
        w = obj.get("weights", {})

        if obj["type"] == "velocity_tracking":
            tx, ty = float(obj["target_x_velocity"]), float(obj["target_y_velocity"])
            sigma = float(obj["sigma_v"])
            err2 = (v_xy[0] - tx) ** 2 + (v_xy[1] - ty) ** 2
            r = float(w.get("w_vel", 1.0) * np.exp(-err2 / (2.0 * sigma * sigma)))
            return r, {"task_vx": float(v_xy[0]), "task_vy": float(v_xy[1]), "task_err2": float(err2)}

        if obj["type"] == "height_tracking":
            th = float(obj["target_height"])
            sigma = float(obj["sigma_h"])
            err2 = (height - th) ** 2
            r = float(w.get("w_height", 1.0) * np.exp(-err2 / (2.0 * sigma * sigma)))
            return r, {"task_h": float(height), "task_err2": float(err2)}

        if obj["type"] == "velocity_height_tracking":
            tx, ty = float(obj["target_x_velocity"]), float(obj["target_y_velocity"])
            sv = float(obj["sigma_v"])
            th = float(obj["target_height"])
            sh = float(obj["sigma_h"])

            err_v2 = (v_xy[0] - tx) ** 2 + (v_xy[1] - ty) ** 2
            err_h2 = (height - th) ** 2
            r = float(
                w.get("w_vel", 1.0) * np.exp(-err_v2 / (2.0 * sv * sv))
                + w.get("w_height", 1.0) * np.exp(-err_h2 / (2.0 * sh * sh))
            )
            return r, {
                "task_vx": float(v_xy[0]),
                "task_vy": float(v_xy[1]),
                "task_h": float(height),
                "task_err_v2": float(err_v2),
                "task_err_h2": float(err_h2),
            }

        if obj["type"] == "qpos_pose_distance":
            # mask: exclude root x/y
            nq = qpos.shape[0]
            mask = np.ones(nq, dtype=bool)
            if obj.get("mask", {}).get("exclude_root_xy", True):
                mask[0] = False
                mask[1] = False

            weights_mode = obj.get("weights", {}).get("mode", "uniform")
            weights = np.ones(nq, dtype=np.float64)
            if weights_mode == "uniform":
                weights[:] = 1.0
            # If you want joint-group weights, add them here after introspecting joint names.

            diff = (qpos - self.pose_target)
            dist = float(np.sqrt(np.sum(weights[mask] * diff[mask] * diff[mask])))
            self.state.min_pose_dist = min(self.state.min_pose_dist, dist)

            thr = float(obj.get("distance", {}).get("success_threshold", 0.7))
            r = float(np.exp(-(dist * dist) / (2.0 * (thr * thr))))
            return r, {"task_pose_dist": dist, "task_pose_min_dist": self.state.min_pose_dist, "task_success": self.state.min_pose_dist < thr}

        raise ValueError(f"Unknown objective type: {obj['type']}")