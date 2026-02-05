import gymnasium as gym
from gymnasium import spaces
import metaworld
import numpy as np

import metaworld

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import gymnasium as gym
import numpy as np


TASKS = ("reachgreen", "reachblue", "pickupkey")


class MultiTaskGridEnv(MiniGridEnv):
    """
    Fixed grid + fixed dynamics. Only the reward function changes based on task_name.
    """

    def __init__(
        self,
        task_name: str,
        size: int = 9,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        seed: int = 0,
        **kwargs,
    ):
        assert task_name in TASKS, f"Unknown task_name={task_name}. Valid: {TASKS}"
        self.task_name = task_name
        self._np_rng = np.random.default_rng(seed)

        mission_space = MissionSpace(mission_func=lambda: str(self.task_name))

        if max_steps is None:
            # MiniGrid convention: O(size^2)
            max_steps = 4 * size * size

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            agent_view_size=size,  # Full observability: see entire grid
            see_through_walls=True,  # Can see through obstacles
            **kwargs,
        )

        # Track events for reward computation
        self._picked_up_key = False

        # Fixed placements (set in _gen_grid)
        self._green_goal_pos: Optional[Tuple[int, int]] = None
        self._blue_goal_pos: Optional[Tuple[int, int]] = None
        self._key_pos: Optional[Tuple[int, int]] = None

    def _make_mission(self) -> str:
        # Mission string is not used for reward; it is just for documentation/debug.
        return f"task={self.task_name}"

    def _gen_grid(self, width: int, height: int) -> None:
        # Fixed grid, no randomness unless you add it.
        self.grid = Grid(width, height)

        # Surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Interior layout (example: a couple of walls + lava strip)
        for x in range(2, width - 2):
            self.grid.set(x, 3, Wall())

        for x in range(2, width - 2):
            self.grid.set(x, height - 3, Lava())

        # Place two goals and a key at fixed coordinates
        self._green_goal_pos = (width - 2, 1)
        self._blue_goal_pos = (1, height - 2)
        self._key_pos = (width // 2, height // 2)

        g = Goal()
        g.color = "green"
        self.put_obj(g, *self._green_goal_pos)

        b = Goal()
        b.color = "blue"
        self.put_obj(b, *self._blue_goal_pos)

        k = Key("yellow")
        self.put_obj(k, *self._key_pos)

        # Fixed agent start
        self.agent_pos = (1, 1)
        self.agent_dir = 0  # 0:right, 1:down, 2:left, 3:up

        self._picked_up_key = False

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info = dict(info)
        info["task_name"] = self.task_name
        return obs, info

    def step(self, action):
        # Cache s_t info you need
        prev_pos = tuple(self.agent_pos)
        prev_carrying_key = (self.carrying is not None and isinstance(self.carrying, Key))

        obs, base_reward, terminated, truncated, info = super().step(action)

        # Cache s_{t+1} info
        new_pos = tuple(self.agent_pos)
        new_carrying_key = (self.carrying is not None and isinstance(self.carrying, Key))

        # Dense task reward
        reward = self._dense_reward(prev_pos, new_pos, prev_carrying_key, new_carrying_key)

        info = dict(info)
        info["task_name"] = self.task_name
        return obs, reward, terminated, truncated, info


    def _dense_reward(self, prev_pos, new_pos, prev_has_key, new_has_key) -> float:
        # Hyperparameters (tune as needed)
        alpha = 1.0      # shaping strength
        step_pen = 0.01  # living cost
        lava_pen = 0.2   # penalty if on lava
        pickup_bonus = 0.2  # bonus exactly when key is picked up

        # Potential
        phi_prev = self._phi(prev_pos, prev_has_key)
        phi_new  = self._phi(new_pos,  new_has_key)

        r = alpha * (phi_new - phi_prev) - step_pen

        # Event bonus: just picked up key
        if (not prev_has_key) and new_has_key:
            r += pickup_bonus

        # Lava penalty based on current cell
        cell = self.grid.get(*new_pos)
        if isinstance(cell, Lava):
            r -= lava_pen

        # Safety check: ensure reward is not NaN or Inf
        if not np.isfinite(r):
            r = 0.0

        return float(r)


    def _phi(self, pos, has_key: bool) -> float:
        # Normalizer for distance -> [0,1]
        D = (self.width - 1) + (self.height - 1)

        # Guard against division by zero
        if D <= 0:
            return 0.0

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        if self.task_name == "reachgreen":
            target = self._green_goal_pos
            if target is None:
                return 0.0
            d = manhattan(pos, target)
            return 1.0 - (d / D)

        if self.task_name == "reachblue":
            target = self._blue_goal_pos
            if target is None:
                return 0.0
            d = manhattan(pos, target)
            return 1.0 - (d / D)

        if self.task_name == "pickupkey":
            # Dense progress: go to key, then (optionally) go to green goal after pickup
            if not has_key:
                target = self._key_pos
            else:
                target = self._green_goal_pos
            if target is None:
                return 0.0
            d = manhattan(pos, target)
            return 1.0 - (d / D)

        # Fallback
        return 0.0

    def set_task(self, task_name: str) -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task_name={task_name}. Valid: {TASKS}")
        self.task_name = task_name
        # No need to regenerate grid because dynamics/layout are fixed and only reward changes.

    @property
    def task(self) -> str:
        return self.task_name

    @task.setter
    def task(self, value: str) -> None:
        self.set_task(value)


class DMControlEnv(gym.Env):
    """Gymnasium wrapper for DeepMind Control Suite."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, domain="cartpole", task="swingup", render_mode=None,
                 from_pixels=False, height=84, width=84, camera_id=0, frame_skip=1):
        super().__init__()
        from dm_control import suite

        self._env = suite.load(domain_name=domain, task_name=task)
        self.render_mode = render_mode
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.frame_skip = frame_skip

        # Action space
        action_spec = self._env.action_spec()
        if action_spec.shape:
            self.action_space = spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                shape=action_spec.shape,
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                shape=(1,),
                dtype=np.float32
            )

        # Observation space
        if from_pixels:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(height, width, 3),
                dtype=np.uint8
            )
        else:
            # Flatten state observations
            obs_spec = self._env.observation_spec()
            obs_dim = int(sum(np.prod(spec.shape) for spec in obs_spec.values()))
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )

    def _get_obs(self, time_step):
        """dm_control TimeStep uses OrderedDict (or pixels if image) to
        store observation this function convert them in an array format.
        """
        if self.from_pixels:
            obs = self._env.physics.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            return obs
        else:
            # Flatten and concatenate all observation components
            # TODO: time_step.observation is an OrderedDict, is order preserved? I think so
            return np.concatenate([
                np.array(v).flatten() for _, v in time_step.observation.items()
            ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            try:
                self._env.task.random.seed(seed)
            except AttributeError:
                # Some tasks may not have random attribute
                pass
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs, {}

    def step(self, action):
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        reward = 0.0
        for _ in range(self.frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        obs = self._get_obs(time_step)
        terminated = time_step.last()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
        elif self.render_mode == "human":
            # For human rendering, you might want to use matplotlib or similar
            # This is a simple placeholder
            img = self._env.physics.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            return img

    def close(self):
        self._env.close()

# ----------------------------
# 1) Minimal base env:
#    - Accepts task token like "dist:0"
#    - Uses it to configure how w is sampled each episode
# ----------------------------

class SimpleDistAsTaskEnv(gym.Env):
    """
    Dynamics:
      s_{t+1} = tanh(A s_t + B a_t) + noise

    Reward:
      r_t = w^T phi(s_t, a_t, s_{t+1})

    Task token:
      task = "dist:k" selects distribution parameters used to sample w each episode.
      This makes "distribution" compatible with your existing "task sequence" logic.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 2,
        feat_dim: int = 4,
        max_episode_steps: int = 20,
        noise_std: float = 0.01,
        seed: int = 0,
        task: Optional[str] = None,
        render_mode: Optional[str] = None,
        dist_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.n = int(state_dim)
        self.m = int(action_dim)
        self.d = int(feat_dim)
        self.H = int(max_episode_steps)
        self.noise_std = float(noise_std)

        self.rng = np.random.default_rng(seed)

        # Fixed fast dynamics
        A = self.rng.normal(size=(self.n, self.n)).astype(np.float32)
        B = self.rng.normal(size=(self.n, self.m)).astype(np.float32)
        A *= 0.7 / (np.linalg.norm(A, 2) + 1e-6)
        B *= 0.7 / (np.linalg.norm(B, 2) + 1e-6)
        self.A, self.B = A, B

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.m,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n,), dtype=np.float32)

        # Distribution specs keyed by task token, e.g. "dist:0"
        self.dist_specs: Dict[str, Dict[str, Any]] = dist_specs or {}
        if not self.dist_specs:
            # default single distribution if none provided
            self.dist_specs = {
                "dist:0": {"w_std": 0.2, "mu_drift_std": 0.05},
            }

        # Current "distribution task"
        self.task_id: str = "dist:0"
        self.task_id_int: int = 0

        # Distribution parameters (configured via set_task)
        self.w_std = 0.2
        self.mu_drift_std = 0.05

        # Distribution state: mean of Gaussian over w
        self.mu = np.zeros((self.d,), dtype=np.float32)

        # Current episode task vector w
        self.w = np.zeros((self.d,), dtype=np.float32)

        # Episode state
        self.t = 0
        self.s = np.zeros((self.n,), dtype=np.float32)

        # Apply initial task token if provided
        if task is not None:
            self.set_task(task)
        else:
            self.set_task(self.task_id)

    def set_task(self, task_token: str) -> None:
        """
        Task token is interpreted as selecting a distribution spec.
        Example tokens: "dist:0", "dist:1", ...
        """
        if task_token not in self.dist_specs:
            raise ValueError(f"Unknown task_token={task_token}. Known: {list(self.dist_specs.keys())}")

        self.task_id = task_token

        # Parse integer id for plotting compatibility
        # If token not in form "dist:k", fallback to 0.
        try:
            self.task_id_int = int(task_token.split(":", 1)[1])
        except Exception:
            self.task_id_int = 0

        spec = self.dist_specs[task_token]

        self.w_std = float(spec.get("w_std", self.w_std))
        self.mu_drift_std = float(spec.get("mu_drift_std", self.mu_drift_std))

        if bool(spec.get("reset_mu", False)):
            self.mu[:] = 0.0

    def _phi(self, s: np.ndarray, a: np.ndarray, sp: np.ndarray) -> np.ndarray:
        """
        Cheap feature map.
        Must output shape (d,).
        """
        base = np.array(
            [
                1.0,
                -np.mean(a * a),
                -np.mean(sp * sp),
                np.mean(s * sp),
                np.mean(sp[:8]),
            ],
            dtype=np.float32,
        )

        if self.d <= base.shape[0]:
            return base[: self.d].copy()

        phi = np.zeros((self.d,), dtype=np.float32)
        phi[: base.shape[0]] = base
        k = min(self.d - base.shape[0], 8)
        phi[base.shape[0] : base.shape[0] + k] = np.tanh(sp[:k]).astype(np.float32)
        return phi

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Do not reseed here unless explicitly requested; for benchmarking you usually do not pass seed repeatedly.
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0

        # Drift mean each episode (how fast mu changes depends on current "distribution task")
        # This creates an episode-level drifting distribution, but the speed changes when task token changes.
        if self.mu_drift_std > 0:
            self.mu = (self.mu + self.mu_drift_std * self.rng.normal(size=(self.d,))).astype(np.float32)

        # Sample w for the episode from current distribution
        self.w = (self.mu + self.w_std * self.rng.normal(size=(self.d,))).astype(np.float32)
        self.w = (self.w / (np.linalg.norm(self.w) + 1e-6)).astype(np.float32)

        self.s = self.rng.normal(scale=0.2, size=(self.n,)).astype(np.float32)

        info = {
            "task_id": self.task_id,
            "task_id_int": self.task_id_int,
        }
        return self.s.copy(), info

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        noise = self.rng.normal(scale=self.noise_std, size=(self.n,)).astype(np.float32)
        sp = (np.tanh(self.A @ self.s + self.B @ a).astype(np.float32) + noise)

        phi = self._phi(self.s, a, sp)
        r = float(self.w @ phi)

        self.s = sp
        self.t += 1

        terminated = False
        truncated = (self.t >= self.H)

        info = {
            "task_id": self.task_id,
            "task_id_int": self.task_id_int,
            #"phi": phi,
        }
        return self.s.copy(), r, terminated, truncated, info

# ==================================================
# Minihack Environments
# ==================================================

from minihack import LevelGenerator
from minihack.reward_manager import Event


class DenseCoordEvent(Event):
    def __init__(self, coordinates: np.ndarray, gamma=0.99, scale=0.01):
        super().__init__(
            reward=0.0,
            repeatable=False,
            terminal_required=True,
            terminal_sufficient=True
        )
        self.goal = coordinates

    def check(self, env, previous_observation, action, observation) -> float:
        coordinates = np.array(observation[env._blstats_index][:2])
        distance = np.linalg.norm(coordinates - self.goal)
        if np.array_equal(coordinates, self.goal):
            return self._set_achieved()

        return -distance

lvl_gen = LevelGenerator(w=20, h=20)
lvl_gen.add_object("apple", "%", (2, 2))
lvl_gen.add_object("dagger", ")", (3, 3))
lvl_gen.add_trap(name="teleport")
lvl_gen.add_sink()
lvl_gen.add_monster("goblin")
lvl_gen.fill_terrain("rect", "|", 0, 0, 19, 19)