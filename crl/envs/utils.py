import gymnasium as gym
from dataclasses import dataclass

from typing import Tuple


@dataclass
class EnvConfig:
    domain_name: str = "mjx/cheetah"
    task: str = "default"  # only used if env has multiple tasks
    task_list: str = "transfer"
    seed: int = 0
    max_episode_steps: int = 1000
    steps_per_task: int = 200_000  # stage-wise interaction budget per task


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