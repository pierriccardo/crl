"""
Proximal Policy Optimization (PPO)

In PPO the agent interacts with the environment for T steps (the `horizon`), the data collected in these steps is called a rollout.
One rollout is a batch of data used for the update. If we use N agents in parallel, we will have N rollouts and a batch of dim N*T.

Once the agent has collected a rollout, we can update the policy and value function. The update is performed by separating the
rollout (of size N*T) into mini-batches (of size `minibatch_size`), then the mini-batches are shuffled and used to perform a gradient
update step, updating the policy and value function. This is done until each mini-batch has been used for the update. This operation
can be repeated for a number of `epochs`. Reusing the same data for different epochs improve sample efficiency.

References:
- Paper: https://arxiv.org/pdf/1707.06347
- Code:
    - https://github.com/zplizzi/pytorch-ppo
    - https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    - https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
    - (GAE) https://github.com/zplizzi/pytorch-ppo/blob/main/gae.py

Additional references:
- [Ref 1] https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/?utm_source=chatgpt.com
- https://arxiv.org/pdf/2005.12729
- https://arxiv.org/abs/2006.05990
- https://blog.xa0.de/post/PPO%20---%20a-Note-on-Policy-Entropy-in-Continuous-Action-Spaces

"""
import os
import tyro
import wandb
import random
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

from tqdm import tqdm
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Tuple, List, Any


from crl.envs import EnvConfig, make_env, get_env_dims


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
    action_low: torch.Tensor = torch.tensor([-1.0])
    action_high: torch.Tensor = torch.tensor([1.0])
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
    n_steps: int = 100_000
    n_epochs: int = 5
    n_agents: int = 1  # Number of agents in parallel, N
    batch_size: int = 4096  # A.k.a. horizon, T
    minibatch_size: int = 512

    window_size: int = 100
    log_freq: int = 100

    # generic optimization parameters
    clip_grad_norm: float | None = None  # usually 0.5

    # Wandb
    use_wandb: bool = True
    proj_name: str = "rl-algorithms"
    algo_name: str = "ppo"

    # Checkpointing
    save_dir: str = "models"
    save_every_steps: Optional[int] = None  # None = only at end
    load_checkpoint: Optional[str] = None    # path or dir to resume from


# ==================================================
# Networks
# ==================================================


def init_mlp_weights(
    module: nn.Module,
    hidden_gain: Optional[float] = None,
    last_gain: Optional[float] = None,
    extra_layers: Optional[List[Tuple[nn.Module, float]]] = None,
) -> None:
    """
    Orthogonal weight init for Linear layers in `module` (e.g. nn.Sequential).
    - hidden_gain: gain for all but last layer (default: tanh).
    - last_gain: if set, use this for the last Linear in `module` instead of hidden_gain.
    - extra_layers: optional list of (layer, gain) for additional layers (e.g. policy head).
    """
    if hidden_gain is None:
        hidden_gain = nn.init.calculate_gain("tanh")
    linear_layers = [m for m in module if isinstance(m, nn.Linear)]
    for i, m in enumerate(linear_layers):
        gain = last_gain if (last_gain is not None and i == len(linear_layers) - 1) else hidden_gain
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.zeros_(m.bias)
    if extra_layers:
        for layer, gain in extra_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)


class Actor(nn.Module):

    def __init__(self, s_dim: int, a_dim: int, action_low: torch.Tensor, action_high: torch.Tensor, discrete: bool = False):
        super().__init__()
        self.discrete = discrete
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))
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

        init_mlp_weights(
            self.net,
            extra_layers=[(self.logits_head if self.discrete else self.mu_head, 0.01)],
        )

    def get_dist(self, s: torch.Tensor) -> torch.distributions.Distribution:
        h = self.net(s)
        if self.discrete:
            logits = self.logits_head(h)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mu_head(h)
            logstd = self.logstd.expand_as(mean)
            std = torch.exp(logstd)
            base = Normal(mean, std)
            # Squash to (-1, 1) via tanh; log_prob and rsample use stable Jacobian inside PyTorch
            return TransformedDistribution(base, TanhTransform(cache_size=1))

    def forward(self, s: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        dist = self.get_dist(s)
        if self.discrete:
            action = dist.sample() if action is None else action
            return action, dist.log_prob(action), dist.entropy()

        else:
            eps = 1e-6
            if action is None:
                a = dist.rsample()
            else:
                # action is in env scale [low, high]; map to (-1, 1)
                a = (action - self.action_low) / (self.action_high - self.action_low)
                a = 2.0 * a - 1.0

            a = a.clamp(-1 + eps, 1 - eps)  # safe for atanh and log_prob
            u = dist.transforms[0].inv(a)
            logprob = dist.log_prob(a).sum(dim=-1)
            base = dist.base_dist
            log_det = dist.transforms[0].log_abs_det_jacobian(u, a)
            entropy = base.entropy().sum(dim=-1) + log_det.sum(dim=-1)

            # rescale action to env scale [low, high]
            a_env = self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)
            return a_env, logprob, entropy

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
        init_mlp_weights(self.net, last_gain=1.0)

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
        self.actor = Actor(config.s_dim, config.a_dim, config.action_low, config.action_high, discrete=config.discrete_actions)
        self.critic = Critic(config.s_dim)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.a_lr, eps=1e-8, )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.c_lr, eps=1e-8)

    def act(self, s: torch.Tensor, values: bool = False):
        action, logprob, entropy = self.actor(s)
        if values:
            return action, logprob, entropy, self.critic(s)
        return (action, logprob, entropy)

    def gae(self, rewards, values, dones):
        """
        - code: https://github.com/zplizzi/pytorch-ppo/blob/main/gae.py
        - paper:
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(T)):
            delta = rewards[t] + self.config.gamma * values[t+1] * (1 - dones[t]) - values[t]
            advantage = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * advantage
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

        metrics = [{
            "pg_loss": 0,
            "v_loss": 0,
            "entropy_loss": 0,
            "actor_loss": 0,
            "critic_loss": 0,
        } for _ in range(self.config.n_epochs)]

        advantages = self.gae(rollout["rewards"], rollout["values"], rollout["dones"])
        b_returns = advantages + rollout["values"][:-1]

        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.config.n_epochs):
            # Create a random permutation of indices shuffle the batch
            indices = torch.randperm(self.config.batch_size)

            sgd_steps = 0
            for start_idx in range(0, self.config.batch_size, self.config.minibatch_size):
                end_idx = start_idx + self.config.minibatch_size
                mb_idxs = indices[start_idx:end_idx]


                _, newlogprobs, entropy = self.actor(rollout["obs"][mb_idxs], rollout["actions"][mb_idxs])
                newvalue = self.critic(rollout["obs"][mb_idxs])
                log_ratio = newlogprobs - rollout["logprobs"][mb_idxs]
                ratio = torch.exp(log_ratio)

                mb_advantages = advantages[mb_idxs]

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
                total_loss = actor_loss + critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        max_norm=self.config.clip_grad_norm
                    )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                metrics[epoch]["pg_loss"] += float(pg_loss.detach())
                metrics[epoch]["v_loss"] += float(v_loss.detach())
                metrics[epoch]["entropy_loss"] += float(entropy_loss.detach())
                metrics[epoch]["actor_loss"] += float(actor_loss.detach())
                metrics[epoch]["critic_loss"] += float(critic_loss.detach())
                sgd_steps += 1

            for m, v in metrics[epoch].items():
                metrics[epoch][m] /= max(1, sgd_steps)

        results = {}
        if self.config.n_epochs > 1:
            # Metrics for first and last epoch for debugging
            for e in [0, self.config.n_epochs - 1]:
                for k, v in metrics[e].items():
                    results[f"{k}_epoch_{e}"] = v
        # Compute mean over epochs for each metric
        mean_metrics = {}
        for k in metrics[0].keys():
            mean_metrics[k] = np.mean([metrics[epoch][k] for epoch in range(self.config.n_epochs)])
        results.update(mean_metrics)

        # TODO: add explained variance
        # TODO: add KL divergence
        return results

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

    # Seeding everything
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.use_wandb:
        wandb.init(
            project=config.proj_name,
            group=f"{config.env.domain_name}-{config.env.task}-s{config.env.seed}",
            name=f"{config.algo_name}-s{config.seed}",
            config=asdict(config)
        )

    env = make_env(
        env_id=config.env.domain_name,
        task=config.env.task,
        seed=config.env.seed, # NB: keep env seed fixed, change only algo seed
    )

    s_dim, a_dim, discrete = get_env_dims(env)
    config.s_dim = s_dim
    config.a_dim = a_dim

    config.discrete_actions = discrete
    if not discrete:
        config.action_low = env.action_space.low
        config.action_high = env.action_space.high
        print(f"action_low: {config.action_low}, action_high: {config.action_high}")

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

    obs_buffer = torch.zeros((config.batch_size, s_dim)).to(config.device)
    actions_buffer = (
        torch.zeros((config.batch_size,), dtype=torch.long).to(config.device)
        if config.discrete_actions
        else torch.zeros((config.batch_size, a_dim)).to(config.device)
    )
    logprobs_buffer = torch.zeros((config.batch_size)).to(config.device)
    rewards_buffer = torch.zeros((config.batch_size)).to(config.device)
    dones_buffer = torch.zeros((config.batch_size)).to(config.device)
    values_buffer = torch.zeros((config.batch_size + 1)).to(config.device)

    step = 0
    episode_idx = 0
    episode_len = 0
    episode_rew = 0.0
    rewards = deque(maxlen=config.window_size)  # last N step rewards

    obs, info = env.reset()

    for t in tqdm(range(config.n_steps)):
        with torch.no_grad():

            obs_tensor = torch.as_tensor(obs, device=config.device, dtype=torch.float32).unsqueeze(0)
            action, logprob, entropy, value = agent.act(obs_tensor, values=True)
            if config.discrete_actions:
                a_env = int(action.item())
            else:
                a_env = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = env.step(a_env)

            obs_buffer[step] = obs_tensor.squeeze(0)
            actions_buffer[step] = action.squeeze(0)
            logprobs_buffer[step] = logprob.squeeze(0)
            rewards_buffer[step] = reward
            dones_buffer[step] = terminated #or truncated # Dones computed only with termination signal
            values_buffer[step] = value.squeeze(0)

            obs = next_obs
            step += 1
            rewards.append(reward)
            episode_rew += reward
            episode_len += 1

            if terminated or truncated:
                episode_idx += 1
                if config.use_wandb:
                    wandb.log({"metrics/reward_per_episode": episode_rew}, step=t)
                    wandb.log({"metrics/episode_length": episode_len}, step=t)
                episode_rew = 0.0
                episode_len = 0
                obs, _ = env.reset()

        if step == config.batch_size:
            with torch.no_grad():
                obs_T = torch.as_tensor(obs, device=config.device, dtype=torch.float32).unsqueeze(0)
                values_buffer[step] = agent.critic(obs_T).squeeze(0)
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
                wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=t)
            step = 0

        if config.save_every_steps and (t + 1) % config.save_every_steps == 0:
            # TODO: rename epoch to step
            agent.save(checkpoint_dir, epoch=t + 1)

        if config.log_freq and (t + 1) % config.log_freq == 0 and rewards:
            mean_reward = sum(rewards) / len(rewards)
            wandb.log({f"metrics/reward_mean_last_{config.window_size}_steps": mean_reward}, step=t)

    agent.save(checkpoint_dir, epoch=config.n_steps)
