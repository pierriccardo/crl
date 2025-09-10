from __future__ import annotations

import numpy as np
import gymnasium as gym

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from typing import List
from dataclasses import dataclass


COLORS = [c for c in COLOR_NAMES if c != "grey"]


@dataclass
class TaskSpec:
    """Specification for a goal reaching task."""
    goal_pos: tuple  # Position of the goal
    name: str  # Name of the task
    color: str = "green"  # Color of the goal
    episodes_per_task: int = 100  # Number of episodes for this task
    max_episode_steps: int = 200  # Maximum steps per episode for this task


class GoalEnv(MiniGridEnv):
    """
    Single goal reaching environment that can be configured with a task specification.
    Similar to individual environments in COOM framework.
    """

    def __init__(
        self,
        task: str | TaskSpec,
        size=10,
        render_mode: str = None,
        **kwargs,
    ):
        # Grid size
        self.grid_size = size

        # Parse task specification
        if isinstance(task, str):
            self.task_spec = self._parse_task_string(task)
        elif isinstance(task, TaskSpec):
            self.task_spec = task
        else:
            raise ValueError("Task must be either a string or TaskSpec instance")

        # Current goal tracking
        self.current_goal_obj = None

        # Action space - 3 actions: turn_left, turn_right, move_forward
        self.action_space = gym.spaces.Discrete(3)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=self.task_spec.max_episode_steps,
            render_mode=render_mode,
            **kwargs,
        )

    def _parse_task_string(self, task_str: str) -> TaskSpec:
        """Parse task string into TaskSpec. Support predefined task names."""
        task_configs = {
            "north": TaskSpec(
                goal_pos=(self.grid_size // 2, 2),
                name="North Goal",
                color="red",
                episodes_per_task=100,
                max_episode_steps=200
            ),
            "east": TaskSpec(
                goal_pos=(self.grid_size - 3, self.grid_size // 2),
                name="East Goal",
                color="blue",
                episodes_per_task=100,
                max_episode_steps=200
            ),
            "south": TaskSpec(
                goal_pos=(self.grid_size // 2, self.grid_size - 3),
                name="South Goal",
                color="yellow",
                episodes_per_task=100,
                max_episode_steps=200
            ),
            "west": TaskSpec(
                goal_pos=(2, self.grid_size // 2),
                name="West Goal",
                color="green",
                episodes_per_task=100,
                max_episode_steps=200
            ),
        }

        if task_str.lower() in task_configs:
            return task_configs[task_str.lower()]
        else:
            raise ValueError(f"Unknown task: {task_str}. Available tasks: {list(task_configs.keys())}")

    @staticmethod
    def _gen_mission():
        """Generate mission description for current task."""
        return "reach the goal"

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        obs, info = super().reset(seed=seed)

        # Add task info
        info.update({
            'task_name': self.task_spec.name,
            'goal_pos': self.task_spec.goal_pos,
            'task_color': self.task_spec.color
        })

        return obs, info

    def _gen_grid(self, width, height):
        """Generate grid with goal and obstacles."""
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the goal with its specific color
        goal_x, goal_y = self.task_spec.goal_pos
        self.current_goal_obj = Goal(self.task_spec.color)
        self.put_obj(self.current_goal_obj, goal_x, goal_y)

        # Add obstacles in the middle area
        center_x, center_y = width // 2, height // 2
        obstacle_radius = min(width, height) // 4

        # Create a cross pattern of obstacles in the center
        obstacles = []

        # Horizontal line of obstacles
        for x in range(max(2, center_x - obstacle_radius),
                       min(width - 2, center_x + obstacle_radius + 1)):
            obstacles.append((x, center_y))

        # Vertical line of obstacles
        for y in range(max(2, center_y - obstacle_radius),
                       min(height - 2, center_y + obstacle_radius + 1)):
            obstacles.append((center_x, y))

        # Add some additional scattered obstacles
        if width >= 8 and height >= 8:
            corner_offset = max(2, width // 4)
            obstacles.extend([
                (corner_offset, corner_offset),
                (width - corner_offset - 1, corner_offset),
                (corner_offset, height - corner_offset - 1),
                (width - corner_offset - 1, height - corner_offset - 1),
            ])

        # Place obstacles, avoiding goal position and agent start
        for x, y in obstacles:
            if (1 < x < width - 1 and 1 < y < height - 1 and
                    (x, y) != self.task_spec.goal_pos and (x, y) != (1, 1)):
                self.grid.set(x, y, Wall())

        # Place the agent in top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.mission = self.task_spec.name

    def step(self, action):
        """Take a step in the environment."""
        obs, _, terminated, truncated, info = super().step(action)

        # Calculate task-specific reward
        reward = self.compute_reward()

        # Add task info
        info.update({
            'task_name': self.task_spec.name,
            'goal_pos': self.task_spec.goal_pos,
            'task_color': self.task_spec.color
        })

        return obs, reward, terminated, truncated, info

    def compute_reward(self) -> float:
        """Calculate distance-based reward for goal reaching."""
        distance = np.sum(np.abs(np.array(self.agent_pos) - np.array(self.task_spec.goal_pos)))

        # Normalize distance by maximum possible distance
        max_distance = self.grid_size * 2
        normalized_distance = distance / max_distance

        # Conservative reward scaling
        step_penalty = -0.05
        distance_reward = -normalized_distance * 0.5

        # Positive reward for reaching the goal
        if distance == 0:
            return 5.0

        return step_penalty + distance_reward


class ContinualGoalEnv:
    """
    A continual learning environment that manages a sequence of GoalEnv tasks.
    Follows COOM framework structure with an array of task environments.

    Similar to COOM's ContinualLearningEnv, this creates individual environments
    for each task and provides methods to iterate through them.
    """

    def __init__(
        self,
        task_sequence: List[str] = None,
        size=10,
        render_mode: str = None,
        wrappers: List[gym.Wrappers] = [],
        **kwargs,
    ):
        # Create default task sequence if none provided
        # TODO: add other sequences type

        task_sequence = ["north", "east", "south", "west"]


        self.task_sequence = task_sequence
        self.size = size
        self.render_mode = render_mode
        self.wrappers = wrappers
        self.kwargs = kwargs

        # Create array of task environments (like COOM)
        self.tasks = []
        for task_name in task_sequence:
            env = GoalEnv(
                task=task_name,
                size=size,
                render_mode=render_mode,
                **kwargs
            )
            for w in self.wrappers:
                env = w(env)
            self.tasks.append(env)

        # Current task tracking
        self.current_task_idx = 0

    def set_task(self, task_idx: int):
        """Set the current active task by index."""
        if 0 <= task_idx < len(self.tasks):
            self.current_task_idx = task_idx
        else:
            raise IndexError(f"Task index {task_idx} out of range [0, {len(self.tasks)-1}]")

    def get_task(self, task_idx: int = None):
        """Get a task environment by index. If no index provided, returns current task."""
        if task_idx is None:
            task_idx = self.current_task_idx

        if 0 <= task_idx < len(self.tasks):
            return self.tasks[task_idx]
        else:
            raise IndexError(f"Task index {task_idx} out of range [0, {len(self.tasks)-1}]")

    def reset_all_tasks(self):
        """Reset all task environments (useful for evaluation)."""
        for env in self.tasks:
            env.reset()

    def close_all_tasks(self):
        """Close all task environments."""
        for env in self.tasks:
            env.close()

    def __len__(self):
        """Return number of tasks."""
        return len(self.tasks)

    def __getitem__(self, idx):
        """Allow indexing into task array."""
        return self.tasks[idx]

    def __iter__(self):
        """Allow iteration over tasks (like COOM)."""
        return iter(self.tasks)


if __name__ == "__main__":
    pass
