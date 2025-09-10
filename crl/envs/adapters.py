import gymnasium as gym
from gymnasium import spaces
import numpy as np
from minatar import Environment


class MiniAtarEnv(gym.Env):
    """Gym wrapper for MiniAtar games."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, game="breakout", render_mode=None):
        super().__init__()
        self.env = Environment(game, sticky_action_prob=0.1)

        # Observation: 10x10 grid with channels
        shape = self.env.state_shape()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=shape, dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(self.env.num_actions())

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        obs = self.env.state()
        return obs, {}

    def step(self, action):
        reward, done = self.env.act(action)
        obs = self.env.state()
        return obs, reward, done, False, {}

    def render(self):
        if self.render_mode == "human":
            self.env.display_state()

    def close(self):
        # MiniAtar Environment doesn't have a close method
        pass
