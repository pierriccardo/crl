import numpy as np
import torch
import gymnasium as gym
from typing import Optional, Tuple


class ReplayBuffer():
    """
    replay buffer multi-dimensional observations and stores data on the specified device.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: torch.device,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination

        # Determine data types based on spaces
        if isinstance(observation_space, gym.spaces.Box):
            self.obs_dtype = np.float32 if observation_space.dtype == np.float32 else np.uint8
        else:
            self.obs_dtype = np.float32

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dtype = np.int64
        else:
            self.action_dtype = np.float32

        # Create storage arrays
        self.observations = np.zeros((buffer_size,) + observation_space.shape, dtype=self.obs_dtype)
        self.next_observations = np.zeros((buffer_size,) + observation_space.shape, dtype=self.obs_dtype)
        self.actions = np.zeros((buffer_size,) + action_space.shape, dtype=self.action_dtype)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.terminations = np.zeros((buffer_size,), dtype=np.bool_)
        self.truncations = np.zeros((buffer_size,), dtype=np.bool_)

        # Buffer state
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        termination: np.ndarray,
        truncation: np.ndarray,
        infos: Optional[dict] = None,
    ) -> None:
        """Add a new transition to the buffer."""

        # Handle final observations for truncated episodes
        if self.handle_timeout_termination and infos is not None:
            for idx, trunc in enumerate(truncation):
                if trunc:
                    next_obs[idx] = infos["final_observation"][idx]

        # Store the transition
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.terminations[self.pos] = np.array(termination).copy()
        self.truncations[self.pos] = np.array(truncation).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions."""
        batch_inds = np.random.randint(0, self.buffer_size if self.full else self.pos, size=batch_size)

        # Convert to tensors and move to device
        observations = torch.as_tensor(self.observations[batch_inds], device=self.device)
        next_observations = torch.as_tensor(self.next_observations[batch_inds], device=self.device)
        actions = torch.as_tensor(self.actions[batch_inds], device=self.device)
        rewards = torch.as_tensor(self.rewards[batch_inds], device=self.device)
        dones = torch.as_tensor(
            self.terminations[batch_inds] | self.truncations[batch_inds],
            device=self.device
        )

        return observations, next_observations, actions, rewards, dones

    def __len__(self) -> int:
        return self.buffer_size if self.full else self.pos