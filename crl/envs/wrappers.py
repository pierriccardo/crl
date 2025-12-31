import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Optional, Callable


class ImageWrapper(gym.ObservationWrapper):
    """Wrapper that extracts the 'image' observation from environments."""

    def __init__(self, env):
        super().__init__(env)
        # Update observation space to match the image shape
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


class ActionWrapper(gym.ActionWrapper):
    """Wrapper that limits actions to a subset of the original action space."""

    def __init__(self, env, action_subset=None):
        super().__init__(env)

        if action_subset is None:
            # Default: only allow turn_left, turn_right, move_forward
            action_subset = [0, 1, 2]  # turn_left, turn_right, move_forward

        self.action_subset = action_subset
        self.action_space = gym.spaces.Discrete(len(action_subset))

    def action(self, action):
        """Map the limited action space back to the original action space."""
        return self.action_subset[action]


class ChannelStandardizationWrapper(gym.ObservationWrapper):
    """Wrapper that standardizes the number of channels across different environments."""

    def __init__(self, env, target_channels=4):
        super().__init__(env)
        self.target_channels = target_channels

        # Get original observation space
        orig_space = env.observation_space
        self.orig_channels = orig_space.shape[2]

        # Create new observation space with target channels
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
    Wrapper for continual episodic environments where:
    - Each episode lasts for exactly N timesteps
    - A task is selected at the start of each episode with a given probability
    - Tasks can persist across multiple episodes, allowing the agent to face
      the same task for consecutive episodes

    This wrapper manages task switching internally and truncates episodes
    at the specified length.
    """

    def __init__(
        self,
        env: gym.Env,
        task_list: List[str],
        episode_length: int,
        env_factory: Optional[Callable[[str], gym.Env]] = None,
        task_switch_prob: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            env: Initial environment instance (will be replaced on task switch)
            task_list: List of task identifiers to randomly sample from
            episode_length: Fixed number of timesteps per episode (N)
            env_factory: Optional callable that takes a task string and returns a new env.
                        If None, assumes env has a 'task' attribute that can be set.
            task_switch_prob: Probability of switching to a new task at each episode reset.
                             Default 1.0 means task changes every episode.
                             Lower values allow tasks to persist across episodes.
            seed: Random seed for task selection
        """
        super().__init__(env)

        if not task_list:
            raise ValueError("task_list must contain at least one task")
        if not 0.0 <= task_switch_prob <= 1.0:
            raise ValueError(f"task_switch_prob must be in [0.0, 1.0], got {task_switch_prob}")

        self.task_list = task_list
        self.episode_length = episode_length
        self.env_factory = env_factory
        self.task_switch_prob = task_switch_prob
        self.rng = np.random.RandomState(seed)

        # Episode tracking
        self.episode_timesteps = 0
        self.current_task = None

        # Store original env attributes
        self._observation_space = env.observation_space
        self._action_space = env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment and optionally switch to a new task."""
        # Switch task with probability task_switch_prob, or if no task is set yet
        if self.current_task is None or self.rng.random() < self.task_switch_prob:
            # Randomly select a new task
            self.current_task = self.rng.choice(self.task_list)

        # Create new environment with selected task if factory is provided
        if self.env_factory is not None:
            self.env = self.env_factory(self.current_task)
            # Update spaces if they changed
            self._observation_space = self.env.observation_space
            self._action_space = self.env.action_space
        else:
            # Try to set task on existing env (if supported)
            if hasattr(self.env, 'task'):
                self.env.task = self.current_task
            elif hasattr(self.env, 'set_task'):
                self.env.set_task(self.current_task)

        # Reset the environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset episode tracking
        self.episode_timesteps = 0

        # Add task info to info dict
        if info is None:
            info = {}
        info['task'] = self.current_task
        info['episode_length'] = self.episode_length

        return obs, info

    def step(self, action):
        """Step the environment and truncate at episode_length."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Increment episode timesteps
        self.episode_timesteps += 1

        # Truncate if we've reached the episode length
        if self.episode_timesteps >= self.episode_length:
            truncated = True

        # Add task info to info dict
        if info is None:
            info = {}
        info['task'] = self.current_task
        info['episode_timestep'] = self.episode_timesteps

        return obs, reward, terminated, truncated, info

    @property
    def observation_space(self):
        """Return observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """Return action space."""
        return self._action_space
