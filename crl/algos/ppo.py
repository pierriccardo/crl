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


from crl.envs import EnvConfig, make_env, make_vec_env, get_env_dims


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
    norm_obs: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01  # entropy bonus for exploration
    vf_coef: float = 0.5

    # train
    n_steps: int = 100_000
    n_epochs: int = 5
    batch_size: int = 4096  # Steps per env per rollout (horizon), T
    minibatch_size: int = 512

    window_size: int = 100
    log_freq: int = 100

    # generic optimization parameters
    clip_grad_norm: float | None = 0.5

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
            nn.Linear(s_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
        )

        if self.discrete:
            self.logits_head = nn.Linear(1024, a_dim)
            self.logstd = None
        else:
            self.mu_head = nn.Linear(1024, a_dim)
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
            # Squashed entropy: H(Y) = H(X) + E[log|dy/dx|]; TanhTransform gives log|da/du| = log(1-a²)
            log_det = dist.transforms[0].log_abs_det_jacobian(u, a)
            entropy = base.entropy().sum(dim=-1) + log_det.sum(dim=-1)

            # rescale action to env scale [low, high]
            a_env = self.action_low + (a + 1.0) * 0.5 * (self.action_high - self.action_low)
            return a_env, logprob, entropy

class Critic(nn.Module):

    def __init__(self, s_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1),
        )
        init_mlp_weights(self.net, last_gain=1.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # Return value shape (B,)
        return self.net(s).squeeze(-1)


# ==================================================
# Observation normalization
# ==================================================


class RunningMeanStd:
    """Welford's online algorithm for tracking running mean/variance.

    Works with batched updates: call ``update(batch)`` where batch has
    shape ``(B, *shape)`` and the statistics are maintained per-feature
    (i.e. over the leading ``B`` dimension).
    """

    def __init__(self, shape: tuple = (), eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps
        self.eps = eps

    def update(self, batch: np.ndarray):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[np.newaxis]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean)
                / np.sqrt(self.var + self.eps)).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, state: dict):
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


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

        self.obs_rms = RunningMeanStd(shape=(config.s_dim,)) if config.norm_obs else None

    def normalize_obs(self, obs: np.ndarray | torch.Tensor, update: bool = True) -> np.ndarray | torch.Tensor:
        """Normalize observations using running mean/std.

        Accepts both numpy arrays and torch tensors. When a torch tensor is
        passed the return value is a torch tensor on the same device.

        Args:
            obs: raw observations, shape ``(N, s_dim)`` or ``(s_dim,)``.
            update: if True, update running statistics (use False at eval time).
        """
        if self.obs_rms is None:
            if isinstance(obs, torch.Tensor):
                return obs.float()
            return np.asarray(obs, dtype=np.float32)

        is_torch = isinstance(obs, torch.Tensor)
        obs_np = obs.detach().cpu().numpy() if is_torch else np.asarray(obs, dtype=np.float32)
        if update:
            self.obs_rms.update(obs_np)
        normalized = self.obs_rms.normalize(obs_np)
        if is_torch:
            return torch.from_numpy(normalized).to(device=obs.device)
        return normalized

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
        rollout keys (all flat, first dim = B = T * N with vectorized envs):
        - obs:        (B, S)
        - actions:    (B, A)  or (B,) for discrete
        - logprobs:   (B,)
        - values:     (B,)   -- rollout values for clipped value loss
        - advantages: (B,)   -- pre-computed GAE advantages
        - returns:    (B,)   -- advantages + values
        """
        B = rollout["obs"].shape[0]
        advantages = rollout["advantages"]
        b_returns = rollout["returns"]

        if self.config.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = [{
            "pg_loss": 0,
            "v_loss": 0,
            "entropy_loss": 0,
            "actor_loss": 0,
            "critic_loss": 0,
        } for _ in range(self.config.n_epochs)]

        for epoch in range(self.config.n_epochs):
            indices = torch.randperm(B, device=rollout["obs"].device)

            sgd_steps = 0
            for start_idx in range(0, B, self.config.minibatch_size):
                end_idx = start_idx + self.config.minibatch_size
                mb_idxs = indices[start_idx:end_idx]

                _, newlogprobs, entropy = self.actor(rollout["obs"][mb_idxs], rollout["actions"][mb_idxs])
                newvalue = self.critic(rollout["obs"][mb_idxs])
                log_ratio = newlogprobs - rollout["logprobs"][mb_idxs]
                ratio = torch.exp(log_ratio)

                mb_advantages = advantages[mb_idxs]

                # Policy loss
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
                    v_loss = v_loss_max.mean()
                else:
                    v_loss = ((newvalue - b_returns[mb_idxs]) ** 2).mean()

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

        # Debugging: approx KL (old vs new policy) and explained variance (value fit)
        with torch.no_grad():
            _, new_logprobs, _ = self.actor(rollout["obs"], rollout["actions"])
            approx_kl = (rollout["logprobs"] - new_logprobs).mean().item()
            new_values = self.critic(rollout["obs"])
            var_returns = b_returns.var().item() + 1e-8
            explained_var = (1.0 - (b_returns - new_values).var().item() / var_returns)

        results["approx_kl"] = approx_kl
        results["explained_variance"] = explained_var
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
        if self.obs_rms is not None:
            payload["obs_rms"] = self.obs_rms.state_dict()
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
        if self.obs_rms is not None and "obs_rms" in payload:
            self.obs_rms.load_state_dict(payload["obs_rms"])

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
        if agent.obs_rms is not None and "obs_rms" in payload:
            agent.obs_rms.load_state_dict(payload["obs_rms"])
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

    num_envs = config.env.num_envs
    T = config.batch_size  # horizon (steps per env per rollout)

    _is_mjx = config.env.domain_name.lower().startswith("mjx/")
    envs = make_vec_env(
        env_id=config.env.domain_name,
        task=config.env.task,
        seed=config.env.seed,
        num_envs=num_envs,
        torch_device=config.device if _is_mjx else None,
    )

    s_dim, a_dim, discrete = get_env_dims(envs)
    config.s_dim = s_dim
    config.a_dim = a_dim

    config.discrete_actions = discrete
    if not discrete:
        act_space = envs.single_action_space
        config.action_low = act_space.low
        config.action_high = act_space.high
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

    # Buffers: (T, N, ...) where T = horizon, N = num_envs
    obs_buffer = torch.zeros((T, num_envs, s_dim), device=config.device)
    actions_buffer = (
        torch.zeros((T, num_envs), dtype=torch.long, device=config.device)
        if config.discrete_actions
        else torch.zeros((T, num_envs, a_dim), device=config.device)
    )
    logprobs_buffer = torch.zeros((T, num_envs), device=config.device)
    rewards_buffer = torch.zeros((T, num_envs), device=config.device)
    dones_buffer = torch.zeros((T, num_envs), device=config.device)
    values_buffer = torch.zeros((T + 1, num_envs), device=config.device)

    total_batch = T * num_envs
    num_updates = config.n_steps // total_batch

    episode_idx = 0
    episode_returns = np.zeros(num_envs, dtype=np.float64)
    episode_lengths = np.zeros(num_envs, dtype=np.int64)
    recent_ep_returns = deque(maxlen=config.window_size)
    global_step = 0

    obs, info = envs.reset()  # (N, s_dim)

    pbar = tqdm(total=config.n_steps, desc="PPO Training")

    for update_idx in range(num_updates):
        for step in range(T):
            with torch.no_grad():
                obs_norm = agent.normalize_obs(obs, update=True)
                obs_tensor = (
                    obs_norm if isinstance(obs_norm, torch.Tensor)
                    else torch.as_tensor(obs_norm, device=config.device, dtype=torch.float32)
                )
                action, logprob, entropy, value = agent.act(obs_tensor, values=True)

                if config.discrete_actions:
                    a_env = action.cpu().numpy().astype(int)
                elif _is_mjx:
                    a_env = action
                else:
                    a_env = action.cpu().numpy()

                next_obs, reward, terminated, truncated, info = envs.step(a_env)

                if not isinstance(reward, torch.Tensor):
                    reward = torch.as_tensor(np.asarray(reward, dtype=np.float32), device=config.device)
                    terminated = torch.as_tensor(np.asarray(terminated), device=config.device)
                    truncated = torch.as_tensor(np.asarray(truncated), device=config.device)
                done = (terminated | truncated)

                obs_buffer[step] = obs_tensor
                actions_buffer[step] = action
                logprobs_buffer[step] = logprob
                rewards_buffer[step] = reward.float()
                dones_buffer[step] = done.float()
                values_buffer[step] = value

                reward_np = reward.detach().cpu().numpy()
                done_np = done.detach().cpu().numpy()
                episode_returns += reward_np
                episode_lengths += 1

                # Log episode stats: from env info (CleanRL-style) when available, else from our accumulation
                if "episode" in info and "_episode" in info:
                    _ep, _r, _l = info["_episode"], info["episode"]["r"], info["episode"]["l"]
                    if torch.is_tensor(_ep):
                        ep_done = _ep.cpu().numpy().reshape(num_envs)
                        ep_r = _r.cpu().numpy().reshape(num_envs)
                        ep_l = _l.cpu().numpy().reshape(num_envs)
                    else:
                        ep_done = np.asarray(_ep).reshape(num_envs)
                        ep_r = np.asarray(_r).reshape(num_envs)
                        ep_l = np.asarray(_l).reshape(num_envs)
                    for i in np.where(ep_done)[0]:
                        episode_idx += 1
                        recent_ep_returns.append(float(ep_r[i]))
                        if config.use_wandb:
                            wandb.log(
                                {"metrics/reward_per_episode": float(ep_r[i]), "metrics/episode_length": int(ep_l[i])},
                                step=global_step,
                                commit=False,
                            )
                else:
                    for i in range(num_envs):
                        if done_np[i]:
                            episode_idx += 1
                            recent_ep_returns.append(episode_returns[i])
                            if config.use_wandb:
                                wandb.log(
                                    {
                                        "metrics/reward_per_episode": episode_returns[i],
                                        "metrics/episode_length": episode_lengths[i],
                                    },
                                    step=global_step,
                                    commit=False,
                                )
                            episode_returns[i] = 0.0
                            episode_lengths[i] = 0

                obs = next_obs
                global_step += num_envs
                pbar.update(num_envs)

        # Bootstrap value for the last observation in each env
        with torch.no_grad():
            obs_norm = agent.normalize_obs(obs, update=False)
            obs_tensor = (
                obs_norm if isinstance(obs_norm, torch.Tensor)
                else torch.as_tensor(obs_norm, device=config.device, dtype=torch.float32)
            )
            values_buffer[T] = agent.critic(obs_tensor)

        # Compute GAE on (T, N) shaped tensors
        advantages = agent.gae(
            rewards_buffer.to(config.device),
            values_buffer.to(config.device),
            dones_buffer.to(config.device),
        )
        returns = advantages + values_buffer[:-1]

        # Flatten (T, N, ...) -> (T*N, ...) for mini-batch SGD
        rollout = {
            "obs": obs_buffer.reshape(total_batch, s_dim),
            "actions": actions_buffer.reshape(total_batch) if config.discrete_actions else actions_buffer.reshape(total_batch, a_dim),
            "logprobs": logprobs_buffer.reshape(total_batch),
            "values": values_buffer[:-1].reshape(total_batch),
            "advantages": advantages.reshape(total_batch),
            "returns": returns.reshape(total_batch),
        }
        metrics = agent.update(rollout)

        if config.use_wandb:
            log_dict = {f"train/{k}": v for k, v in metrics.items()}
            if config.log_freq and recent_ep_returns:
                log_dict[f"metrics/mean_ep_return_last_{config.window_size}"] = (
                    sum(recent_ep_returns) / len(recent_ep_returns)
                )
            wandb.log(log_dict, step=global_step, commit=True)

        if config.save_every_steps and global_step >= config.save_every_steps and global_step % config.save_every_steps < total_batch:
            agent.save(checkpoint_dir, epoch=global_step)

    pbar.close()
    agent.save(checkpoint_dir, epoch=global_step)
