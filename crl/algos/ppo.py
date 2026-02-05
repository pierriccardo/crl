"""
PPO implementation
- GAE https://github.com/zplizzi/pytorch-ppo/blob/main/gae.py
"""


import tyro
import wandb
import random
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List, Any
from tqdm import tqdm

from crl.envs import EnvConfig, make_env


def get_checkpoint_dir(
    env_name: str,
    task_name: str,
    algo_name: str,
    seed: int,
    base_dir: str = "models",
) -> str:
    """
    Default checkpoint directory structure:
      <base_dir>/<env_name>/<task_name>/<algo_name>/seed_<seed>/
    env_name and task_name are sanitized for filesystem (e.g. slashes replaced).
    """
    safe_env = env_name.replace("/", "-")
    safe_task = str(task_name).replace("/", "-")
    return os.path.join(base_dir, safe_env, safe_task, algo_name, f"seed_{seed}")


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"

    env: EnvConfig = field(default_factory=EnvConfig)

    # Set dynamically from environment
    s_dim: int = -1
    a_dim: int = -1
    discrete_actions: bool = False

    a_lr: float = 3e-4
    c_lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.00
    vf_coef: float = 0.5

    # train
    epochs: int = 1000
    batch_size: int = 4096
    minibatch_size: int = 512

    # Wandb
    use_wandb: bool = True
    proj_name: str = "rl-algorithms"
    algo_name: str = "ppo"

    # Checkpointing
    save_dir: str = "models"
    save_every_epochs: Optional[int] = None  # None = only at end
    load_checkpoint: Optional[str] = None    # path or dir to resume from


# ==================================================
# Networks
# ==================================================

class Actor(nn.Module):

    def __init__(self, s_dim: int, a_dim: int, discrete: bool = False):
        super().__init__()
        self.discrete = discrete
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        if self.discrete:
            self.logits_head = nn.Linear(256, a_dim)
            self.logstd = None
        else:
            self.mu_head = nn.Linear(256, a_dim)
            self.logstd = nn.Parameter(torch.zeros(a_dim))

    def get_dist(self, s: torch.Tensor) -> torch.distributions.Distribution:
        h = self.net(s)
        if self.discrete:
            logits = self.logits_head(h)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mu_head(h)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            return torch.distributions.Normal(mean, std)

    def forward(self, s: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        dist = self.get_dist(s)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        if not self.discrete:
            # For continuous actions, sum over action dimensions
            logprob = logprob.sum(dim=-1)
        entropy = dist.entropy()
        if not self.discrete:
            # For continuous actions, sum over action dimensions
            entropy = entropy.sum(dim=-1)
        return action, logprob, entropy

class Critic(nn.Module):

    def __init__(self, s_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Return value shape (B,)
        return self.net(s).squeeze(-1)


# ==================================================
# PPO
# ==================================================

class PPO:

    def __init__(self, config: Config):

        self.config = config
        self.device = config.device
        self.actor = Actor(config.s_dim, config.a_dim, discrete=config.discrete_actions)
        self.critic = Critic(config.s_dim)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.c_lr)

    def act(self, s: torch.Tensor, values: bool = False) -> torch.Tensor:
        action, logprob, entropy = self.actor(s)
        if values:
            return action, logprob, entropy, self.critic(s)
        return action

    def gae(self, rewards, values, dones):
        """
        - code: https://github.com/zplizzi/pytorch-ppo/blob/main/gae.py
        - paper:
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(T - 1)):
            delta = rewards[t] + self.config.gamma * values[t+1] * (1 - dones[t+1]) - values[t]
            advantage = delta + self.config.gamma * self.config.gae_lambda * advantage
            advantages[t] = advantage
        return advantages

    def update(self, rollout: Dict[str, torch.Tensor]):
        """
        rollout: Dict[str, torch.Tensor]
        - obs:      (T, S)
        - actions:  (T, A)
        - logprobs: (T,)
        - values:   (T,)
        - rewards:  (T,)
        - dones:    (T,)
        """

        metrics = {
            "pg_loss": 0,
            "v_loss": 0,
            "entropy_loss": 0,
            "actor_loss": 0,
            "critic_loss": 0,
        }

        sgd_steps = 0
        b_idxs = np.arange(self.config.batch_size)
        advantages = self.gae(rollout["rewards"], rollout["values"], rollout["dones"])
        b_returns = advantages + rollout["values"]

        for start in range(0, self.config.batch_size, self.config.minibatch_size):
            end = start + self.config.minibatch_size
            mb_idxs = b_idxs[start:end]

            _, newlogprobs, entropy = self.actor(rollout["obs"][mb_idxs], rollout["actions"][mb_idxs])
            newvalue = self.critic(rollout["obs"][mb_idxs])
            log_ratio = newlogprobs - rollout["logprobs"][mb_idxs]
            ratio = torch.exp(log_ratio)

            mb_advantages = advantages[mb_idxs]
            if self.config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Polivy update
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            if self.config.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_idxs]) ** 2
                v_clipped = rollout["values"][mb_idxs] + torch.clamp(
                    newvalue - rollout["values"][mb_idxs],
                    -self.config.clip_coef,
                    self.config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_idxs]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_idxs]) ** 2).mean()

            # Entropy loss
            entropy_loss = entropy.mean()
            actor_loss = pg_loss - self.config.ent_coef * entropy_loss
            critic_loss = v_loss * self.config.vf_coef

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            metrics["pg_loss"] += float(pg_loss.detach())
            metrics["v_loss"] += float(v_loss.detach())
            metrics["entropy_loss"] += float(entropy_loss.detach())
            metrics["actor_loss"] += float(actor_loss.detach())
            metrics["critic_loss"] += float(critic_loss.detach())
            sgd_steps += 1

        for m, v in metrics.items():
            metrics[m] /= max(1, sgd_steps)

        # TODO: add explained variance
        # TODO: add KL divergence
        return metrics

    def save(
        self,
        path: str,
        include_optimizers: bool = True,
        epoch: Optional[int] = None,
        **extra: Any,
    ) -> str:
        """
        Save actor, critic, config, and optionally optimizers to path.
        path can be a directory (saves checkpoint.pt inside) or a file path.
        Returns the actual file path written.
        """
        path = path.rstrip("/")
        if os.path.isdir(path) or not path.endswith(".pt"):
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, "checkpoint.pt")

        payload = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "config": asdict(self.config),
        }
        if include_optimizers:
            payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
        if epoch is not None:
            payload["epoch"] = epoch
        payload.update(extra)

        torch.save(payload, path)
        return path

    def load(self, path: str, load_optimizers: bool = True) -> Dict[str, Any]:
        """
        Load checkpoint from path (file or directory containing checkpoint.pt).
        Agent must already have matching architecture (s_dim, a_dim, discrete_actions).
        Returns the loaded payload (e.g. epoch, config).
        """
        path = path.rstrip("/")
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at {path}")

        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic.load_state_dict(payload["critic_state_dict"])
        if load_optimizers and "actor_optimizer_state_dict" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        if load_optimizers and "critic_optimizer_state_dict" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])

        return payload

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> Tuple["PPO", Dict[str, Any]]:
        """
        Load a checkpoint into a new PPO agent. Rebuilds config from checkpoint.
        Returns (agent, payload). Use payload.get("epoch") for resuming step.
        """
        path = path.rstrip("/")
        if os.path.isdir(path):
            path = os.path.join(path, "checkpoint.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint at {path}")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        config_dict = payload["config"]
        valid_keys = set(getattr(Config, "__dataclass_fields__", {}))
        kwargs = {k: config_dict[k] for k in valid_keys if k in config_dict}
        if "env" in kwargs and isinstance(kwargs["env"], dict):
            kwargs["env"] = EnvConfig(**kwargs["env"])
        config = Config(**kwargs)
        if device is not None:
            config.device = device
        agent = cls(config)
        agent.actor.load_state_dict(payload["actor_state_dict"])
        agent.critic.load_state_dict(payload["critic_state_dict"])
        agent.actor.to(agent.device)
        agent.critic.to(agent.device)
        if "actor_optimizer_state_dict" in payload:
            agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in payload:
            agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        return agent, payload


if __name__ == "__main__":

    config = tyro.cli(Config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.use_wandb:
        wandb.init(
            project=config.proj_name,
            group=f"{config.env.domain_name}-{config.env.task_list}-s{config.env.seed}",
            name=f"{config.algo_name}-s{config.seed}",
            config=asdict(config)
        )

    env = make_env(
        env_id=config.env.domain_name,
        task=config.env.task,
        seed=config.env.seed, # NB: keep env seed fixed, change only algo seed
    )

    if hasattr(env.observation_space, 'shape'):
        if len(env.observation_space.shape) == 0:
            obs_dim = 1
        else:
            obs_dim = int(np.prod(env.observation_space.shape))
    else:
        # Fallback: reset and check observation
        obs, _ = env.reset()
        obs_dim = int(np.prod(obs.shape))

    if isinstance(env.action_space, gym.spaces.Discrete) or hasattr(env.action_space, 'n'):
        action_dim = int(env.action_space.n)
    elif hasattr(env.action_space, 'shape'):
        if len(env.action_space.shape) == 0:
            action_dim = 1
        else:
            action_dim = int(np.prod(env.action_space.shape))

    config.s_dim = obs_dim
    config.a_dim = action_dim

    if config.load_checkpoint:
        agent, payload = PPO.from_checkpoint(config.load_checkpoint, device=config.device)
        start_epoch = payload.get("epoch", 0)
    else:
        agent = PPO(config)
        start_epoch = 0

    checkpoint_dir = get_checkpoint_dir(
        config.env.domain_name,
        config.env.task,
        config.algo_name,
        config.seed,
        base_dir=config.save_dir,
    )

    obs_buffer = torch.zeros((config.batch_size, obs_dim)).to(config.device)
    actions_buffer = torch.zeros((config.batch_size, action_dim)).to(config.device)
    logprobs_buffer = torch.zeros((config.batch_size)).to(config.device)
    rewards_buffer = torch.zeros((config.batch_size)).to(config.device)
    dones_buffer = torch.zeros((config.batch_size)).to(config.device)
    values_buffer = torch.zeros((config.batch_size)).to(config.device)

    episode = 0
    episode_reward = 0.0
    obs, info = env.reset()

    for epoch in tqdm(range(start_epoch, config.epochs)):
        with torch.no_grad():
            for step in tqdm(range(config.batch_size)):
                obs_tensor = torch.as_tensor(obs, device=config.device, dtype=torch.float32).unsqueeze(0)
                action, logprob, entropy, value = agent.act(obs_tensor, values=True)
                next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])

                obs_buffer[step] = obs_tensor
                actions_buffer[step] = action
                logprobs_buffer[step] = logprob
                rewards_buffer[step] = reward
                dones_buffer[step] = terminated or truncated
                values_buffer[step] = value

                episode_reward += reward
                obs = next_obs

                if terminated or truncated:
                    episode += 1
                    if config.use_wandb:
                        wandb.log({"metrics/reward_per_episode": episode_reward}, step=episode)
                    episode_reward = 0.0
                    obs, _ = env.reset()

        rollout = {
            "obs": obs_buffer,
            "actions": actions_buffer,
            "logprobs": logprobs_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "values": values_buffer,
        }

        metrics = agent.update(rollout)
        if config.use_wandb:
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=episode)

        if config.save_every_epochs and (epoch + 1) % config.save_every_epochs == 0:
            agent.save(checkpoint_dir, epoch=epoch + 1)

    agent.save(checkpoint_dir, epoch=config.epochs)
