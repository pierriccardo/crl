import numpy as np
import gymnasium as gym
from gymnasium import spaces


from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any


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
