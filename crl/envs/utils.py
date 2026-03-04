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
    num_envs: int = 1  # number of parallel envs (vectorized)


def get_env_dims(env: gym.Env) -> Tuple[int, int, bool]:
    """Get the dimensions of the environment.

    Works with both single envs and Gymnasium VectorEnvs (uses
    single_observation_space / single_action_space when available).
    """
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    action_space = getattr(env, "single_action_space", env.action_space)

    if isinstance(action_space, gym.spaces.Discrete):
        a_dim = action_space.n
        discrete = True
    elif isinstance(action_space, gym.spaces.Box):
        a_dim = action_space.shape[0]
        discrete = False
    else:
        raise NotImplementedError

    if isinstance(obs_space, gym.spaces.Dict):
        # Unwrap single-key Dict spaces (e.g. MJX's {"state": Box(17,)}) to
        # a scalar dim so callers can use s_dim as an int directly.
        for key in ("state", "observation", "obs"):
            if key in obs_space.spaces:
                obs_space = obs_space.spaces[key]
                break

    if isinstance(obs_space, gym.spaces.Box):
        s_dim = obs_space.shape[0]
    elif isinstance(obs_space, gym.spaces.Discrete):
        s_dim = obs_space.n
    elif isinstance(obs_space, gym.spaces.Dict):
        s_dim = {k: v.shape for k, v in obs_space.items()}
    else:
        raise NotImplementedError

    return s_dim, a_dim, discrete