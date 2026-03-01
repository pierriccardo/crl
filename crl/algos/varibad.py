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

from crl.envs import EnvConfig, make_env, get_task_sequence, get_env_dims
from crl.algos.ppo import PPO, Config as PPOConfig
from crl.buffers import SimpleTrajBuffer

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
    vae_batch_size: int = 32  # Number of episodes per VAE batch (reduced from 64)
    vae_seq_length: int = 32  # Sequence length for VAE training (-1 = auto: max_episode_steps // 3)
    vae_burnin: int = 0  # Burn-in steps for VAE training

    # Wandb
    use_wandb: bool = True
    proj_name: str = "continual-rl"
    algo_name: str = "variBAD"


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

        # sequential KL computed on full sequence then sliced (simplest)
        mu0 = torch.zeros_like(mu[:, 0])
        lv0 = torch.zeros_like(logvar[:, 0])
        kl0 = self.kl_diag(mu[:, 0], logvar[:, 0], mu0, lv0)  # [B]
        kls = [kl0]
        for t in range(1, T):
            kls.append(self.kl_diag(mu[:, t], logvar[:, t], mu[:, t-1], logvar[:, t-1]))
        kl = torch.stack(kls, dim=1)  # [B,T]
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

    env = make_env(
        env_id=config.env.domain_name,
        task=task_list[0],
        max_episode_steps=config.env.max_episode_steps,
        seed=config.env.seed,
    )

    s_dim, a_dim, discrete = get_env_dims(env)
    config.s_dim = s_dim
    config.a_dim = a_dim
    config.discrete_actions = discrete
    config.discrete = discrete  # for backward compat in loop

    if not config.discrete_actions:
        config.ppo.action_low = env.action_space.low
        config.ppo.action_high = env.action_space.high
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

    obs_buffer = torch.zeros((config.ppo.batch_size + 1, config.s_dim + config.z_dim)).to(config.device)
    if config.discrete_actions:
        actions_buffer = torch.zeros((config.ppo.batch_size + 1,), dtype=torch.long).to(config.device)
    else:
        actions_buffer = torch.zeros((config.ppo.batch_size + 1, config.a_dim)).to(config.device)
    logprobs_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)
    rewards_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)
    dones_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)
    values_buffer = torch.zeros((config.ppo.batch_size + 1)).to(config.device)

    steps_per_task = max(1, int(config.env.steps_per_task))
    total_steps = int(config.num_steps) if config.num_steps > 0 else len(task_list) * steps_per_task

    global_step = 0
    episode_idx = 0
    ppo_buffer_idx = 0

    pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

    for task_idx, current_task in enumerate(task_list):

        env = make_env(
            env_id=config.env.domain_name,
            task=current_task,
            max_episode_steps=config.env.max_episode_steps,
            seed=config.env.seed,
        )

        obs, _ = env.reset()
        terminated = truncated = False
        episode_return = 0.0
        last_n_rewards = deque(maxlen=100)
        episode_steps = 0
        prev_obs = None
        prev_action = None
        prev_reward = 0.0
        vae_episode = {k: [] for k in ["s", "a", "r", "sp", "d"]}
        agent.reset()

        for task_step in range(steps_per_task):

            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, device=config.device, dtype=torch.float32).unsqueeze(0)

                if prev_obs is not None:
                    if config.discrete_actions:
                        action_onehot = np.zeros(config.a_dim, dtype=np.float32)
                        action_onehot[int(prev_action)] = 1.0
                        prev_action_arr = action_onehot
                    else:
                        prev_action_arr = np.asarray(prev_action, dtype=np.float32).flatten()
                    x_t_np = np.concatenate([
                        prev_obs.flatten(),
                        prev_action_arr,
                        [prev_reward],
                        obs.flatten(),
                        [0.0]
                    ], dtype=np.float32)
                    x_t = torch.from_numpy(x_t_np).unsqueeze(0).to(config.device)
                else:
                    x_t = None

                action, logprob, entropy, value, info_dict = agent.act(obs_tensor, x_t, values=True)
                z_pol = info_dict["z_pol"]
                action_np = action.cpu().item() if config.discrete_actions else action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, info = env.step(action_np)
            episode_steps += 1
            if episode_steps >= config.env.max_episode_steps:
                truncated = True
            done = terminated or truncated

            episode_return += reward
            last_n_rewards.append(reward)
            global_step += 1
            if global_step % 10 == 0:
                pbar.update(10)

            obs_aug = torch.cat([obs_tensor.squeeze(0), z_pol.squeeze(0)], dim=-1)
            obs_buffer[ppo_buffer_idx] = obs_aug
            actions_buffer[ppo_buffer_idx] = action.squeeze(0)
            logprobs_buffer[ppo_buffer_idx] = logprob.item()
            rewards_buffer[ppo_buffer_idx] = reward
            dones_buffer[ppo_buffer_idx] = float(done)
            values_buffer[ppo_buffer_idx] = value.item()
            ppo_buffer_idx += 1

            vae_episode["s"].append(obs)
            vae_episode["a"].append(action_np)
            vae_episode["r"].append([reward])
            vae_episode["sp"].append(next_obs)
            vae_episode["d"].append([float(done)])

            prev_obs = obs
            prev_action = action_np
            prev_reward = reward
            obs = next_obs

            # PPO update when rollout buffer full and past warmup
            if ppo_buffer_idx == config.ppo.batch_size + 1:
                if global_step >= config.warmup_steps:
                    rollout = {
                        "obs": obs_buffer[:config.ppo.batch_size],
                        "actions": actions_buffer[:config.ppo.batch_size],
                        "logprobs": logprobs_buffer[:config.ppo.batch_size],
                        "rewards": rewards_buffer[:config.ppo.batch_size],
                        "dones": dones_buffer[:config.ppo.batch_size],
                        "values": values_buffer[:config.ppo.batch_size + 1],
                    }
                    metrics = agent.update(rollout, replay_buffer)
                    if config.use_wandb:
                        wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                ppo_buffer_idx = 0

            # Episode termination: log, reset episode state, reset env
            if terminated or truncated:
                # Only add to VAE buffer after warmup so it isn't filled with random trajectories
                if len(vae_episode["s"]) > 0 and global_step >= config.warmup_steps:
                    for key in vae_episode:
                        vae_episode[key] = torch.tensor(np.array(vae_episode[key]), dtype=torch.float32)
                    replay_buffer.add_episode(vae_episode)

                pbar.set_postfix(
                    task=f"{task_idx+1}/{len(task_list)} {current_task}",
                    ep_return=f"{episode_return:.2f}",
                    avg_r=f"{np.mean(last_n_rewards):.3f}",
                )

                if config.use_wandb:
                    wandb.log({"train/episode_return": episode_return}, step=global_step)
                if config.use_wandb and config.log_freq > 0 and global_step % config.log_freq == 0:
                    wandb.log(
                        {
                            "train/last_n_rewards": np.mean(last_n_rewards),
                            "train/task_id": task_idx,
                        },
                        step=global_step,
                    )

                episode_return = 0.0
                episode_idx += 1
                prev_obs = None
                prev_action = None
                prev_reward = 0.0
                vae_episode = {k: [] for k in ["s", "a", "r", "sp", "d"]}
                agent.reset()

                obs, _ = env.reset()
                terminated = truncated = False
                episode_steps = 0

    pbar.close()

