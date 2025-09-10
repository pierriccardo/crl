import numpy as np
import gymnasium as gym
from gymnasium import spaces


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
            padding = self.target_channels - orig_channels
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