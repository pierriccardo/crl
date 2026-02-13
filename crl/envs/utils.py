import gymnasium as gym
from dataclasses import dataclass

from typing import Tuple

@dataclass
class EnvConfig:
    domain_name: str = "dmc/walker"
    task: str = "default" # only used if env has multiple tasks
    task_list: str = "default"
    seed: int = 0
    max_episode_steps: int = 1000
    task_switch_prob: float = .01  # Probability of switching task at each episode reset (1.0 = always switch)


def get_env_dims(env: gym.Env) -> Tuple[int, int, bool]:
    """Get the dimensions of the environment."""
    # actions
    if isinstance(env.action_space, gym.spaces.Discrete):
        a_dim = env.action_space.n
        discrete = True
    elif isinstance(env.action_space, gym.spaces.Box):
        a_dim = env.action_space.shape[0]
        discrete = False
    else:
        raise NotImplementedError

    # observations
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Box):
        s_dim = obs_space.shape[0]
    elif isinstance(obs_space, gym.spaces.Discrete):
        s_dim = obs_space.n
    elif isinstance(obs_space, gym.spaces.Dict):
        s_dim = {k: v.shape for k, v in obs_space.items()}
    else:
        raise NotImplementedError

    return s_dim, a_dim, discrete