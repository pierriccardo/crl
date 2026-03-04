import tyro
import wandb
import random
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, asdict, field
from typing import Any, Optional
from collections import deque
from tqdm import tqdm

from crl.envs import EnvConfig, make_env, make_vec_env, get_task_sequence, get_env_dims
from crl.algos.ppo import PPO, Config as PPOConfig
from crl.algos.continual_eval import ContinualMetricTracker, extract_success_from_info, obs_to_np_vector
from crl.buffers import SimpleTrajBuffer


def _extract_success_from_infos(infos, num_envs: int):
    """Best-effort extraction of per-env success flags from vectorized infos."""
    if not isinstance(infos, dict):
        return None
    for key in ("success", "is_success"):
        if key not in infos:
            continue
        raw = infos[key]
        if torch.is_tensor(raw):
            arr = raw.detach().cpu().numpy()
        else:
            arr = np.asarray(raw)
        arr = np.asarray(arr).reshape(-1)
        if arr.size < num_envs:
            continue
        try:
            return arr[:num_envs].astype(np.float32) > 0.0
        except Exception:
            return arr[:num_envs].astype(bool)
    return None


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda"
    # to be set dynamically from environment
    discrete: Optional[bool] = None
    s_dim: int = -1
    a_dim: int = -1

    h_dim: int = 256
    z_dim: int = 8
    beta_kl: float = 0.0
    vae_lr: float = 3e-4

    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    # train
    batch_size: int = 4096
    minibatch_size: int = 512

    num_steps: int = 0  # total train steps (0 => len(task_list)*env.steps_per_task)
    buffer_size: int = 100_000  # size of the replay buffer

    warmup_steps: int = 2_000  # steps before first PPO/VAE update (avoid long random-only phase)
    #train_freq_steps: int = 1000  # unused; varibad updates when PPO rollout buffer is full (for API parity with fb_cpr)
    log_freq: int = 100

    # VAE training
    vae_batch_size: int = 64  # Number of episodes per VAE batch
    vae_seq_length: int = -1  # Sequence length for VAE training (-1 = auto: max_episode_steps // 3)
    vae_burnin: int = 0  # Burn-in steps for VAE training

    # Wandb
    use_wandb: bool = True
    proj_name: str = "continual-rl"
    algo_name: str = "variBAD"

    # eval
    do_eval: bool = False
    num_eval_episodes: int = 5


# ==================================================
# Networks
# ==================================================

# This is used to encode trajectory \tau -> h_T
# Encoder input is the step t data, that is:
# x = [s_t, a_t, r_t, s_{t+1}, done_t] shape [B, x_dim]
# with x_dim = s_dim + a_dim + 1 + s_dim + 1
class RecurrentEncoder(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=x_dim, hidden_size=h_dim, batch_first=True)

    def forward(self, x, h0=None):
        out, hT = self.gru(x, h0)

        # Here we have 2 outputs:
        # out.shape == [B, T, h_dim]
        # hT.shape == [num_layers, B, h_dim]
        # For a single layer, hT.shape == [1, B, h_dim]
        # So:
        # out[:, -1, :] == h_last[0, :, :]
        return out, hT

# This encodes p(s_{t+1} | s_t, a_t, z_t)
class TransitionDecoder(nn.Module):
    def __init__(self, s_dim, a_dim, z_dim, h_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, s_dim),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))

# This encodes p(r_{t+1} | s_t, a_t, z_t)
class RewardDecoder(nn.Module):
    def __init__(self, s_dim, a_dim, z_dim, h_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, s, a, z):
        return self.net(torch.cat([s, a, z], dim=-1))


class VAE(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Encoder compress trajectory \tau -> h_T
        # x_t = [s_t, a_t, r_t, s_{t+1}, done_t] where a_t is one-hot for discrete
        x_dim = config.s_dim + config.a_dim + 1 + config.s_dim + 1
        self.encoder = RecurrentEncoder(x_dim, config.h_dim)

        # Posterior parameters \mu, \log\sigma^2,
        # these represents q_\phi(z | h_T) = N(\mu, \sigma^2)
        self.posterior_mu = nn.Linear(config.h_dim, config.z_dim)
        self.posterior_logvar = nn.Linear(config.h_dim, config.z_dim)

        # Decoder reconstruct trajectory \tau from z
        self.decoder_T = TransitionDecoder(config.s_dim, config.a_dim, config.z_dim, config.h_dim)
        self.decoder_R = RewardDecoder(config.s_dim, config.a_dim, config.z_dim, config.h_dim)

    def _action_to_vae(self, a):
        # a: [B,T,1] or [B,T] for discrete; [B,T,a_dim] for continuous
        if not self.config.discrete:
            return a.float()
        a_idx = a.long()
        if a_idx.dim() == 3 and a_idx.shape[-1] == 1:
            a_idx = a_idx.squeeze(-1)
        a_oh = torch.nn.functional.one_hot(a_idx, num_classes=self.config.a_dim).float()
        return a_oh

    def infer_step(self, x_t, h_prev):
        # Used to act online
        # inputs:
        #   1) x_t = (s_t,a_t,r_t,s_{t+1},d_{t+1}) shape: [B, x_dim]
        #   2) h: previous encoder hidden state shape: [1, B, h_dim]
        #
        # Internally, h_prev: the previous encoder hidden state
        # Shape: [1, B, h_dim] or None at reset
        # What it does internally
        # Feeds the single step into the GRU
        # Updates the hidden state
        # Computes posterior parameters from the new hidden state
        # Samples z_t
        out, h_next = self.encoder(x_t.unsqueeze(1), h_prev)  # out: [B,1,h_dim]
        h_t = out[:, -1]                                 # [B,h_dim]
        mu_t = self.posterior_mu(h_t)
        logvar_t = self.posterior_logvar(h_t)
        z_t = self.sample_z(mu_t, logvar_t)
        return mu_t, logvar_t, z_t, h_next

    def infer_seq(self, x_seq, h0=None):
        # Used to train VAE does the same as infer_step but
        # for an entire trajectory of transitions
        # x_seq: [B,T,x_dim]
        h_seq, _ = self.encoder(x_seq, h0)               # [B,T,h_dim]
        mu = self.posterior_mu(h_seq)                    # [B,T,z_dim]
        logvar = self.posterior_logvar(h_seq)            # [B,T,z_dim]
        z = self.sample_z(mu, logvar)                    # [B,T,z_dim]
        return mu, logvar, z

    @staticmethod
    def sample_z(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_diag(mu_q, logvar_q, mu_p, logvar_p):
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        kl = 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p).pow(2)) / (var_p + 1e-8) - 1.0)
        return kl.sum(dim=-1)

    def loss(self, batch, beta=1.0, burnin: int = 0):
        # batch fields: s,a,r,sp,d with shapes [B,T,*]
        s  = batch["s"]
        a  = self._action_to_vae(batch["a"])  # Handles discrete and continuous actions
        r  = batch["r"]
        sp = batch["sp"]
        d  = batch["d"]

        x_seq = torch.cat([s, a, r, sp, d], dim=-1)  # [B,T,x_dim]
        mu, logvar, z = self.infer_seq(x_seq)

        B, T, _ = s.shape

        # optional burn-in: score only t >= burnin
        t0 = burnin
        if t0 >= T:
            raise ValueError("burnin must be < T")

        s2  = s[:, t0:]
        a2  = a[:, t0:]
        r2  = r[:, t0:]
        sp2 = sp[:, t0:]
        d2  = d[:, t0:]
        z2  = z[:, t0:]
        mu2 = mu[:, t0:]
        lv2 = logvar[:, t0:]

        # recon losses (flatten time)
        B2, T2, _ = s2.shape
        s_f  = s2.reshape(B2*T2, -1)
        a_f  = a2.reshape(B2*T2, -1)
        r_f  = r2.reshape(B2*T2, -1)
        sp_f = sp2.reshape(B2*T2, -1)
        z_f  = z2.reshape(B2*T2, -1)

        r_hat = self.decoder_R(s_f, a_f, z_f).reshape(B2, T2, -1)
        sp_hat = self.decoder_T(s_f, a_f, z_f).reshape(B2, T2, -1)

        recon_r = (r_hat - r2).pow(2).mean(dim=-1)       # [B2,T2]
        recon_s = (sp_hat - sp2).pow(2).mean(dim=-1)     # [B2,T2]

        # sequential KL: KL(q_t || q_{t-1}) with q_0 = N(0,I), fully vectorized
        mu0 = torch.zeros_like(mu[:, :1])           # [B,1,z_dim]
        lv0 = torch.zeros_like(logvar[:, :1])       # [B,1,z_dim]
        mu_prev = torch.cat([mu0, mu[:, :-1]], dim=1)       # [B,T,z_dim]
        lv_prev = torch.cat([lv0, logvar[:, :-1]], dim=1)   # [B,T,z_dim]
        kl = self.kl_diag(mu, logvar, mu_prev, lv_prev)     # [B,T]
        kl = kl[:, t0:]               # [B,T2]

        mask = 1.0 - d2.squeeze(-1)   # [B2,T2]
        loss = ((recon_r + recon_s) * mask + beta * kl * mask).mean()

        metrics = {
            "loss": loss.item(),
            "recon_r": (recon_r * mask).mean().item(),
            "recon_s": (recon_s * mask).mean().item(),
            "kl": (kl * mask).mean().item(),
            "std_mean": (0.5 * lv2).exp().mean().item(),
        }
        return loss, metrics


# ==================================================
# VariBAD
# ==================================================

class VariBAD:
    def __init__(self, config: Config):
        # PPO needs to see [obs, z] concatenated
        ppo_config = config.ppo
        ppo_config.s_dim = config.s_dim + config.z_dim
        ppo_config.a_dim = config.a_dim
        ppo_config.discrete_actions = config.discrete  # Use correct field name
        ppo_config.device = config.device

        self.config = config
        self.ppo = PPO(config=ppo_config)
        self.vae = VAE(config=config).to(config.device)
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=config.vae_lr)
        self.h = None  # encoder hidden state for online inference

    def reset(self):
        self.h = None

    def act(self, obs, x_t_for_encoder=None, values=False):
        # If x_t_for_encoder is provided, update VAE state
        if x_t_for_encoder is not None:
            mu, logvar, z, self.h = self.vae.infer_step(x_t_for_encoder, self.h)
            z_pol = z.detach()
        else:
            # Use zero z if no encoder input (first step of episode)
            z_pol = torch.zeros((obs.shape[0], self.config.z_dim), device=obs.device)

        obs_aug = torch.cat([obs, z_pol], dim=-1)
        info = {"z_pol": z_pol}

        if values:
            a, logp, entropy, v = self.ppo.act(obs_aug, values=True)
            return a, logp, entropy, v, info
        out = self.ppo.act(obs_aug, values=False)
        if isinstance(out, tuple) and len(out) == 3:
            a, logp, entropy = out
        else:
            a = out if not isinstance(out, tuple) else out[0]
            logp = entropy = None
        return a, logp, info

    def update(self, rollout: dict, traj_buffer: Any, beta: float = 1.0):
        # Update PPO
        ppo_metrics = self.ppo.update(rollout)

        # Update VAE if trajectory buffer has enough data
        vae_B = self.config.vae_batch_size
        vae_L = self.config.vae_seq_length
        burnin = self.config.vae_burnin

        if len(traj_buffer) >= vae_B * vae_L:
            vae_batch = traj_buffer.sample(B=vae_B, L=vae_L, burnin=burnin)
            vae_loss, vae_metrics = self.vae.loss(vae_batch, beta=beta, burnin=burnin)

            self.opt_vae.zero_grad()
            vae_loss.backward()
            self.opt_vae.step()
        else:
            vae_metrics = {"loss": 0.0, "recon_r": 0.0, "recon_s": 0.0, "kl": 0.0, "std_mean": 0.0}

        out = {}
        out.update({f"train/ppo/{k}": v for k, v in ppo_metrics.items()})
        out.update({f"train/vae/{k}": v for k, v in vae_metrics.items()})
        return out


def evaluate_task_set(agent: VariBAD, config: Config, task_list: list[str]):
    """Evaluate current policy on all tasks; return vectors of mean reward/success."""
    reward_means: list[float] = []
    success_means: list[float] = []

    for task in task_list:
        eval_env = make_env(
            env_id=config.env.domain_name,
            task=task,
            max_episode_steps=config.env.max_episode_steps,
            seed=config.env.seed,
        )

        ep_returns = np.zeros((config.num_eval_episodes,), dtype=np.float32)
        ep_successes: list[float] = []
        saw_success_signal = False

        for ep in range(config.num_eval_episodes):
            obs, _ = eval_env.reset(seed=config.env.seed + ep)
            obs = obs_to_np_vector(obs)
            terminated = truncated = False

            h = None
            has_prev = False
            prev_obs = np.zeros((config.s_dim,), dtype=np.float32)
            prev_reward = 0.0
            if config.discrete:
                prev_action = 0
            else:
                prev_action = np.zeros((config.a_dim,), dtype=np.float32)
            ep_success = False

            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, device=config.device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if has_prev:
                        if config.discrete:
                            prev_action_enc = np.zeros((config.a_dim,), dtype=np.float32)
                            prev_action_enc[int(prev_action)] = 1.0
                        else:
                            prev_action_enc = prev_action.astype(np.float32)

                        x_t_np = np.concatenate(
                            [
                                prev_obs,
                                prev_action_enc,
                                np.array([prev_reward], dtype=np.float32),
                                obs,
                                np.array([0.0], dtype=np.float32),
                            ],
                            axis=0,
                        )
                        x_t = torch.as_tensor(x_t_np, device=config.device, dtype=torch.float32).unsqueeze(0)
                        _, _, z, h = agent.vae.infer_step(x_t, h)
                        z_current = z.detach()
                    else:
                        z_current = torch.zeros((1, config.z_dim), device=config.device)

                    obs_aug = torch.cat([obs_tensor, z_current], dim=-1)
                    action, _, _ = agent.ppo.act(obs_aug, values=False)

                if config.discrete:
                    action_env = int(action.squeeze(0).detach().cpu().item())
                    prev_action = action_env
                else:
                    action_env = action.squeeze(0).detach().cpu().numpy()
                    prev_action = action_env

                next_obs, reward, terminated, truncated, info = eval_env.step(action_env)
                success = extract_success_from_info(info)
                if success is not None:
                    saw_success_signal = True
                    ep_success = ep_success or (success > 0)

                ep_returns[ep] += float(reward)
                prev_obs = obs.copy()
                prev_reward = float(reward)
                has_prev = True
                obs = obs_to_np_vector(next_obs)

            if saw_success_signal:
                ep_successes.append(float(ep_success))

        eval_env.close()
        reward_means.append(float(np.mean(ep_returns)))
        if saw_success_signal and ep_successes:
            success_means.append(float(np.mean(ep_successes)))
        else:
            success_means.append(float("nan"))

    return np.asarray(reward_means, dtype=np.float32), np.asarray(success_means, dtype=np.float32)




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

    task_list = get_task_sequence(config.env.domain_name, config.env.task_list)
    print(f"task list on main: {task_list}")

    num_envs = config.env.num_envs

    _tmp_env = make_env(
        env_id=config.env.domain_name,
        task=task_list[0],
        max_episode_steps=config.env.max_episode_steps,
        seed=config.env.seed,
    )
    s_dim, a_dim, discrete = get_env_dims(_tmp_env)
    if not discrete:
        action_low = _tmp_env.action_space.low
        action_high = _tmp_env.action_space.high
    _tmp_env.close()

    config.s_dim = s_dim
    config.a_dim = a_dim
    config.discrete_actions = discrete
    config.discrete = discrete

    if not discrete:
        config.ppo.action_low = action_low
        config.ppo.action_high = action_high
        print(f"action_low: {config.ppo.action_low}, action_high: {config.ppo.action_high}")

    if config.vae_seq_length == -1:
        config.vae_seq_length = max(8, config.env.max_episode_steps // 3)
        print(f"Auto-set vae_seq_length to {config.vae_seq_length} (max_episode_steps={config.env.max_episode_steps})")
    if config.vae_seq_length > config.env.max_episode_steps:
        print(f"Warning: vae_seq_length ({config.vae_seq_length}) > max_episode_steps ({config.env.max_episode_steps}). Capping.")
        config.vae_seq_length = config.env.max_episode_steps

    agent = VariBAD(config)

    # VAE trajectory buffer
    replay_buffer = SimpleTrajBuffer(
        device=config.device,
        capacity_episodes=config.buffer_size
    )

    T = config.ppo.batch_size  # horizon (steps per env per rollout)
    obs_aug_dim = s_dim + config.z_dim
    total_batch = T * num_envs

    # PPO buffers: (T, N, ...)
    obs_buffer = torch.zeros((T, num_envs, obs_aug_dim), device=config.device)
    if discrete:
        actions_buffer = torch.zeros((T, num_envs), dtype=torch.long, device=config.device)
    else:
        actions_buffer = torch.zeros((T, num_envs, a_dim), device=config.device)
    logprobs_buffer = torch.zeros((T, num_envs), device=config.device)
    rewards_buffer = torch.zeros((T, num_envs), device=config.device)
    dones_buffer = torch.zeros((T, num_envs), device=config.device)
    values_buffer = torch.zeros((T + 1, num_envs), device=config.device)

    steps_per_task = max(1, int(config.env.steps_per_task))
    total_steps = int(config.num_steps) if config.num_steps > 0 else len(task_list) * steps_per_task

    global_step = 0
    episode_idx = 0

    pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

    _use_torch_env = config.env.domain_name.lower().startswith("mjx/") and not discrete
    reward_tracker: ContinualMetricTracker | None = None
    success_tracker: ContinualMetricTracker | None = None
    has_success_eval = False

    if config.do_eval:
        reward_tracker = ContinualMetricTracker(num_tasks=len(task_list), name="reward")
        success_tracker = ContinualMetricTracker(num_tasks=len(task_list), name="success")
        base_rewards, base_success = evaluate_task_set(agent, config, task_list)
        reward_tracker.set_baseline(base_rewards)
        if not np.all(np.isnan(base_success)):
            success_tracker.set_baseline(base_success)
            has_success_eval = True

    for task_idx, current_task in enumerate(task_list):

        envs = make_vec_env(
            env_id=config.env.domain_name,
            task=current_task,
            max_episode_steps=config.env.max_episode_steps,
            seed=config.env.seed,
            num_envs=num_envs,
            torch_device=config.device if _use_torch_env else None,
        )

        obs, _ = envs.reset()  # torch tensor (MJX GPU) or numpy
        last_n_rewards = deque(maxlen=100)
        episode_success = np.zeros(num_envs, dtype=bool)
        has_success_signal = False

        # Per-env VAE encoder state
        h = None  # GRU hidden: [1, N, h_dim] or None
        has_prev = np.zeros(num_envs, dtype=bool)
        prev_obs = np.zeros((num_envs, s_dim), dtype=np.float32)
        if discrete:
            prev_actions = np.zeros(num_envs, dtype=np.int64)
        else:
            prev_actions = np.zeros((num_envs, a_dim), dtype=np.float32)
        prev_rewards = np.zeros(num_envs, dtype=np.float32)
        z_current = torch.zeros((num_envs, config.z_dim), device=config.device)

        # Per-env VAE episode tracking
        vae_episodes = [
            {k: [] for k in ["s", "a", "r", "sp", "d"]}
            for _ in range(num_envs)
        ]
        ppo_step = 0
        task_steps_done = 0

        while task_steps_done < steps_per_task:
            with torch.no_grad():
                obs_tensor = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, device=config.device, dtype=torch.float32)
                obs_np = obs_tensor.detach().cpu().numpy()

                # Build encoder input x_t for all envs
                if has_prev.any():
                    if discrete:
                        prev_action_enc = np.zeros((num_envs, a_dim), dtype=np.float32)
                        for i in range(num_envs):
                            if has_prev[i]:
                                prev_action_enc[i, int(prev_actions[i])] = 1.0
                    else:
                        prev_action_enc = prev_actions.copy()

                    x_t_np = np.concatenate([
                        prev_obs,
                        prev_action_enc,
                        prev_rewards.reshape(num_envs, 1),
                        obs_np,
                        np.zeros((num_envs, 1), dtype=np.float32),
                    ], axis=-1)

                    # Zero-out rows for envs without a previous transition
                    x_t_np[~has_prev] = 0.0

                    x_t = torch.as_tensor(x_t_np, device=config.device, dtype=torch.float32)
                    _, _, z, h = agent.vae.infer_step(x_t, h)
                    z_current = z.detach()

                    mask = torch.as_tensor(
                        has_prev, device=config.device, dtype=torch.float32
                    ).unsqueeze(-1)
                    z_current = z_current * mask
                else:
                    z_current = torch.zeros(
                        (num_envs, config.z_dim), device=config.device
                    )

                # Augmented obs for PPO
                obs_aug = torch.cat([obs_tensor, z_current], dim=-1)
                action, logprob, entropy, value = agent.ppo.act(obs_aug, values=True)

                if discrete:
                    a_env = action.cpu().numpy().astype(int)
                elif _use_torch_env:
                    a_env = action
                else:
                    a_env = action.detach().cpu().numpy()

            next_obs, rewards_raw, terminated_raw, truncated_raw, infos = envs.step(a_env)

            # Batch-convert to numpy for storage (single GPU sync point)
            if isinstance(next_obs, torch.Tensor):
                next_obs_np = next_obs.detach().cpu().numpy()
                rewards_np = rewards_raw.cpu().numpy()
                terminated_np = terminated_raw.cpu().numpy().astype(bool)
                truncated_np = truncated_raw.cpu().numpy().astype(bool)
            else:
                next_obs_np = np.asarray(next_obs, dtype=np.float32)
                rewards_np = np.asarray(rewards_raw, dtype=np.float32)
                terminated_np = np.asarray(terminated_raw, dtype=bool)
                truncated_np = np.asarray(truncated_raw, dtype=bool)
            done = terminated_np | truncated_np

            a_env_np = a_env.detach().cpu().numpy() if isinstance(a_env, torch.Tensor) else np.asarray(a_env)

            last_n_rewards.extend(rewards_np.tolist())
            task_steps_done += num_envs
            global_step += num_envs
            pbar.update(num_envs)

            # Terminal obs for done envs (VectorEnv auto-resets)
            real_next_obs_np = next_obs_np.copy()
            if "final_observation" in infos:
                for i in range(num_envs):
                    fo = infos["final_observation"][i]
                    if done[i] and fo is not None:
                        real_next_obs_np[i] = fo.detach().cpu().numpy() if isinstance(fo, torch.Tensor) else fo
            info_ep_mask = np.zeros(num_envs, dtype=bool)
            info_ep_returns = np.zeros(num_envs, dtype=np.float32)
            info_ep_lengths = np.zeros(num_envs, dtype=np.float32)
            if "episode" in infos and "_episode" in infos:
                raw_ep_mask = infos["_episode"]
                raw_ep_returns = infos["episode"]["r"]
                raw_ep_lengths = infos["episode"]["l"]
                info_ep_mask = (
                    raw_ep_mask.detach().cpu().numpy().astype(bool)
                    if torch.is_tensor(raw_ep_mask)
                    else np.asarray(raw_ep_mask, dtype=bool)
                ).reshape(-1)
                info_ep_returns = (
                    raw_ep_returns.detach().cpu().numpy().astype(np.float32)
                    if torch.is_tensor(raw_ep_returns)
                    else np.asarray(raw_ep_returns, dtype=np.float32)
                ).reshape(-1)
                info_ep_lengths = (
                    raw_ep_lengths.detach().cpu().numpy().astype(np.float32)
                    if torch.is_tensor(raw_ep_lengths)
                    else np.asarray(raw_ep_lengths, dtype=np.float32)
                ).reshape(-1)
            success_flags = _extract_success_from_infos(infos, num_envs)
            if success_flags is not None:
                has_success_signal = True
                episode_success |= success_flags

            # Store in PPO buffer
            obs_buffer[ppo_step] = obs_aug
            actions_buffer[ppo_step] = action
            logprobs_buffer[ppo_step] = logprob
            rewards_buffer[ppo_step] = torch.as_tensor(rewards_np, dtype=torch.float32, device=config.device)
            dones_buffer[ppo_step] = torch.as_tensor(done, dtype=torch.float32, device=config.device)
            values_buffer[ppo_step] = value
            ppo_step += 1

            # Store per-env VAE episode data
            for i in range(num_envs):
                vae_episodes[i]["s"].append(obs_np[i])
                vae_episodes[i]["a"].append(int(a_env_np[i]) if discrete else a_env_np[i])
                vae_episodes[i]["r"].append([float(rewards_np[i])])
                vae_episodes[i]["sp"].append(real_next_obs_np[i])
                vae_episodes[i]["d"].append([float(done[i])])

            # Update per-env previous-transition state
            prev_obs[:] = obs_np
            prev_actions[:] = a_env_np
            prev_rewards[:] = rewards_np
            has_prev[:] = True

            # Handle episode terminations
            step_ep_returns = []
            step_ep_lengths = []
            step_ep_success = []
            for i in range(num_envs):
                if info_ep_mask[i]:
                    ep_return = float(info_ep_returns[i])
                    ep_length = float(info_ep_lengths[i])
                    step_ep_returns.append(ep_return)
                    step_ep_lengths.append(ep_length)
                    if has_success_signal:
                        step_ep_success.append(float(episode_success[i]))
                    episode_success[i] = False

                    if len(vae_episodes[i]["s"]) > 0 and global_step >= config.warmup_steps:
                        ep_data = {}
                        for key in vae_episodes[i]:
                            ep_data[key] = torch.tensor(
                                np.array(vae_episodes[i][key]), dtype=torch.float32
                            )
                        replay_buffer.add_episode(ep_data)

                    pbar.set_postfix(
                        task=f"{task_idx+1}/{len(task_list)} {current_task}",
                        ep_return=f"{ep_return:.2f}",
                        avg_r=f"{np.mean(last_n_rewards):.3f}" if last_n_rewards else "n/a",
                    )

                    # Reset per-env state
                    vae_episodes[i] = {k: [] for k in ["s", "a", "r", "sp", "d"]}
                    has_prev[i] = False
                    if h is not None:
                        h[:, i, :] = 0.0
                    episode_idx += 1

            obs = next_obs

            # PPO + VAE update when rollout buffer is full
            if ppo_step == T:
                if global_step >= config.warmup_steps:
                    with torch.no_grad():
                        obs_last = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, device=config.device, dtype=torch.float32)
                        obs_aug_last = torch.cat([obs_last, z_current], dim=-1)
                        values_buffer[T] = agent.ppo.critic(obs_aug_last)

                    advantages = agent.ppo.gae(
                        rewards_buffer.to(config.device),
                        values_buffer.to(config.device),
                        dones_buffer.to(config.device),
                    )
                    returns = advantages + values_buffer[:-1]

                    rollout = {
                        "obs": obs_buffer.reshape(total_batch, obs_aug_dim),
                        "actions": actions_buffer.reshape(total_batch) if discrete else actions_buffer.reshape(total_batch, a_dim),
                        "logprobs": logprobs_buffer.reshape(total_batch),
                        "values": values_buffer[:-1].reshape(total_batch),
                        "advantages": advantages.reshape(total_batch),
                        "returns": returns.reshape(total_batch),
                    }
                    metrics = agent.update(rollout, replay_buffer)
                    if config.use_wandb:
                        wandb.log(metrics, step=global_step)
                ppo_step = 0

            if config.use_wandb and step_ep_returns:
                log_payload = {
                    "metrics/episode_return": float(np.mean(step_ep_returns)),
                    "metrics/episode_length": float(np.mean(step_ep_lengths)),
                }
                if step_ep_success:
                    log_payload["metrics/episode_success"] = float(np.mean(step_ep_success))
                wandb.log(log_payload, step=global_step)

        envs.close()

        if config.do_eval and reward_tracker is not None:
            eval_rewards, eval_success = evaluate_task_set(agent, config, task_list)
            eval_payload = {}
            for j, tname in enumerate(task_list):
                eval_payload[f"eval/reward/{tname}"] = float(eval_rewards[j])
            eval_payload.update(reward_tracker.update(task_idx, eval_rewards))

            if success_tracker is not None and (has_success_eval or not np.all(np.isnan(eval_success))):
                if not has_success_eval and not np.all(np.isnan(eval_success)):
                    success_tracker.set_baseline(eval_success)
                    has_success_eval = True
                if has_success_eval:
                    for j, tname in enumerate(task_list):
                        if not np.isnan(eval_success[j]):
                            eval_payload[f"eval/success/{tname}"] = float(eval_success[j])
                    eval_payload.update(success_tracker.update(task_idx, eval_success))

            if config.use_wandb:
                wandb.log(eval_payload, step=global_step)

    pbar.close()
