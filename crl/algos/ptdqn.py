"""
Implementation of Permanent Transient DQN (PT-DQN) for continual reinforcement learning.

- Paper: "Prediction and Control in Continual Reinforcement Learning" (NeurIPS 2023)
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/c94bbbef466ab1b2cfa100e41413b3a8-Abstract-Conference.html

This implementation decomposes the value function into permanent and transient components
for continual learning without explicit task boundaries.
"""

import os
import json
import tyro
import torch
import wandb
import random
import dataclasses
import numpy as np
import gymnasium as gym
from collections import deque

from torch import nn
from torch.optim import Adam
from dataclasses import dataclass, asdict
from tqdm import tqdm

from crl.buffers import ReplayBuffer
from crl.envs import EnvConfig, make_env, get_task_sequence


# ==============================
# Parameters
# ==============================

@dataclass
class Config:

    s_dim: int = -1
    a_dim: int = -1

    seed: int = 0
    env: EnvConfig = dataclasses.field(default_factory=EnvConfig)

    # Training parameters
    num_steps: int = 0  # total train steps (0 => len(task_list)*env.steps_per_task)
    buffer_size: int = 100_000

    # Dual-timescale specific parameters
    adaptation_threshold: float = 0.1     # Change detection (lowered for sensitivity)
    prediction_window: int = 50   # Window for computing prediction error variance (smaller for faster detection)
    min_transient_weight: float = 0.05
    max_transient_weight: float = 0.95
    epsilon_restore_threshold: float = 0.2  # Lower threshold for epsilon restoration

    device: str = "cuda"

    # Dual-timescale learning rates
    lr_permanent: float = 2e-4  # Slower learning for permanent component
    lr_transient: float = 5e-3  # Faster learning for transient component
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.2
    epsilon_decay: float = 0.999

    target_update_freq: int = 1000
    tau: float = 1.0  # Hard update coefficient for target network
    batch_size: int = 128
    train_freq_steps: int = 1  # Train every N environment steps
    warmup_steps: int = 1000  # Steps before training starts

    # Evaluation
    eval_freq_steps: int = 10000
    do_eval: bool = True
    num_eval_episodes: int = 5

    # wandb
    use_wandb: bool = True
    algo_name: str = "PTDQN"
    proj_name: str = "continual-rl"




# ==============================
# Networks
# ==============================

class DualTimescaleQNetwork(nn.Module):
    """Q-Network with separate permanent and transient components (MLP for flat vector states)"""

    def __init__(self, s_dim, a_dim):
        super().__init__()
        # Architecture for flat vector inputs
        # s_dim can be a tuple like (147,) or an int
        if isinstance(s_dim, tuple):
            obs_dim = s_dim[0] if len(s_dim) == 1 else int(np.prod(s_dim))
        else:
            obs_dim = s_dim

        # Shared fully connected layers
        self.shared_fc = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Permanent component head (learns slowly, general knowledge)
        self.permanent_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim)
        )

        # Transient component head (learns quickly, adapts to changes)
        self.transient_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim)
        )

        self._init_weights()

    def _init_weights(self):
        # Hidden Linear: Kaiming for ReLU, bias=0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Output layers: small uniform near 0
        for head in [self.permanent_head, self.transient_head]:
            output_layer = head[-1]
            nn.init.uniform_(output_layer.weight, -1e-3, 1e-3)
            nn.init.constant_(output_layer.bias, 0.0)

    def forward(self, x, return_components=False):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        features = self.shared_fc(x)

        q_permanent = self.permanent_head(features)
        q_transient = self.transient_head(features)

        if return_components:
            return q_permanent, q_transient
        else:
            # Return combined Q-values (weighting will be handled by agent)
            return q_permanent + q_transient


# ==============================
# Dual-Timescale DQN Agent
# ==============================
class PTDQN():
    """Dual-Timescale Deep Q-Network agent for continual learning without task boundaries"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" or config.device == "mps" else "cpu")

        # Networks
        self.q_network = DualTimescaleQNetwork(config.s_dim, config.a_dim).to(self.device)
        self.target_network = DualTimescaleQNetwork(config.s_dim, config.a_dim).to(self.device)

        # Separate optimizers for different learning rates
        permanent_params = (list(self.q_network.shared_fc.parameters()) +
                            list(self.q_network.permanent_head.parameters()))
        transient_params = list(self.q_network.transient_head.parameters())

        self.permanent_optimizer = Adam(permanent_params, lr=config.lr_permanent)
        self.transient_optimizer = Adam(transient_params, lr=config.lr_transient)

        self.target_network.load_state_dict(self.q_network.state_dict())

        # Training state
        self.epsilon = config.epsilon
        self.global_step = 0

        # Change detection and adaptation
        self.prediction_errors = deque(maxlen=config.prediction_window)
        self.transient_weight = 0.5  # Initial weight for transient component
        self.change_signal = 0.0

    def _update_change_detection(self, td_error: float):
        """Update change detection based on prediction error variance"""
        self.prediction_errors.append(abs(td_error))

        if len(self.prediction_errors) < self.config.prediction_window:
            return

        # Compute coefficient of variation of recent prediction errors
        errors = np.array(self.prediction_errors)
        error_variance = np.var(errors)
        error_mean = np.mean(errors)

        if error_mean > 1e-6:
            cv = error_variance / (error_mean ** 2)
        else:
            cv = 0.0

        raw_signal = min(cv / self.config.adaptation_threshold, 1.0)

        prev_change_signal = self.change_signal
        self.change_signal = 0.9 * self.change_signal + 0.1 * raw_signal

        if len(self.prediction_errors) % 20 == 0:
            print(f"[Debug] Change detection: signal={self.change_signal:.3f}, epsilon={self.epsilon:.3f}, td_error={td_error:.3f}")

        # Restore epsilon when significant change is detected
        if self.change_signal > self.config.epsilon_restore_threshold and prev_change_signal <= self.config.epsilon_restore_threshold:
            self.epsilon = self.config.epsilon
            print(f"[Change Detection] Epsilon restored to {self.epsilon:.3f} (signal: {self.change_signal:.3f})")

        # Update transient weight based on change signal
        target_weight = (self.config.min_transient_weight +
                         (self.config.max_transient_weight - self.config.min_transient_weight) *
                         self.change_signal)
        self.transient_weight = 0.9 * self.transient_weight + 0.1 * target_weight

    def act(self, state, training: bool = False):
        """Select action using epsilon-greedy policy on combined Q-values"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.config.a_dim)

        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)

            q_permanent, q_transient = self.q_network(state_tensor, return_components=True)

            q_combined = ((1 - self.transient_weight) * q_permanent +
                          self.transient_weight * q_transient)

            return q_combined.argmax().item()

    def update(self, replay_buffer):
        """Perform one training step with dual timescales"""
        if len(replay_buffer) < self.config.batch_size:
            return

        states, next_states, actions, rewards, dones = replay_buffer.sample(self.config.batch_size)

        q_permanent, q_transient = self.q_network(states, return_components=True)

        q_permanent_selected = q_permanent.gather(1, actions.unsqueeze(1))
        q_transient_selected = q_transient.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_permanent, next_q_transient = self.target_network(next_states, return_components=True)

            next_q_combined = ((1 - self.transient_weight) * next_q_permanent +
                               self.transient_weight * next_q_transient)

            next_q_max = next_q_combined.max(1)[0].unsqueeze(1)
            target_q_values = (rewards.unsqueeze(1) +
                               (self.config.gamma * next_q_max * ~dones.unsqueeze(1)))

        criterion = nn.SmoothL1Loss()
        permanent_loss = criterion(q_permanent_selected, target_q_values)
        transient_loss = criterion(q_transient_selected, target_q_values)

        combined_q = ((1 - self.transient_weight) * q_permanent_selected +
                      self.transient_weight * q_transient_selected)
        td_error = (target_q_values - combined_q).mean().item()
        self._update_change_detection(td_error)


        self.permanent_optimizer.zero_grad()
        self.transient_optimizer.zero_grad()

        total_loss = permanent_loss + transient_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_value_(
            list(self.q_network.shared_fc.parameters()) +
            list(self.q_network.permanent_head.parameters()), 100)
        torch.nn.utils.clip_grad_value_(
            list(self.q_network.transient_head.parameters()), 100)

        self.permanent_optimizer.step()
        self.transient_optimizer.step()

        if self.global_step % self.config.target_update_freq == 0:
            for t_param, q_param in zip(self.target_network.parameters(),
                                        self.q_network.parameters()):
                t_param.data.copy_(
                    self.config.tau * q_param.data +
                    (1.0 - self.config.tau) * t_param.data
                )

        # Decay epsilon
        if self.epsilon > self.config.epsilon_min:
            self.epsilon *= self.config.epsilon_decay

        # Increment global step
        self.global_step += 1

        # Return diagnostic information
        metrics = {
            'permanent_loss': permanent_loss.item(),
            'transient_loss': transient_loss.item(),
            'combined_loss': criterion(combined_q, target_q_values).item(),
            'td_error': abs(td_error),
            'change_signal': self.change_signal,
            'transient_weight': self.transient_weight,
            'q_permanent_mean': q_permanent_selected.mean().item(),
            'q_transient_mean': q_transient_selected.mean().item(),
            'q_combined_mean': combined_q.mean().item(),
            'target_mean': target_q_values.mean().item(),
            'reward_mean': rewards.mean().item(),
            'reward_max': rewards.max().item(),
            'reward_min': rewards.min().item()
        }

        return metrics

    def save(self, path):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "permanent_optimizer": self.permanent_optimizer.state_dict(),
            "transient_optimizer": self.transient_optimizer.state_dict(),
            "epsilon": self.epsilon,
            "global_step": self.global_step,
            "transient_weight": self.transient_weight,
            "change_signal": self.change_signal,
            "config": self.config.__dict__,
            "metrics": self.metrics,
        }
        torch.save(payload, path)

    def load(self, path):
        """Load the model"""
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.permanent_optimizer.load_state_dict(ckpt["permanent_optimizer"])
        self.transient_optimizer.load_state_dict(ckpt["transient_optimizer"])
        self.epsilon = ckpt.get("epsilon", self.config.epsilon)
        self.global_step = ckpt.get("global_step", 0)
        self.transient_weight = ckpt.get("transient_weight", 0.5)
        self.change_signal = ckpt.get("change_signal", 0.0)

        # Load data collection state
        self.metrics = ckpt.get("metrics", {'training': [], 'eval': [], 'continual': [], 'config': asdict(self.config)})


def evaluate_all_tasks(agent, config, t):

    task_list = get_task_sequence(config.env.domain_name, config.env.task_list)

    if config.use_wandb:
        # One table for all tasks at this eval step (preferred)
        returns_table = wandb.Table(columns=["eval_step", "task", "episode", "return"])

    for task in task_list:
        eval_env = make_env(env_id=config.env.domain_name, task=task)

        total_reward = np.zeros((config.num_eval_episodes,), dtype=np.float32)
        for ep in range(config.num_eval_episodes):
            obs, _ = eval_env.reset()

            terminated = truncated = False
            while not (terminated or truncated):

                obs_tensor = torch.as_tensor(obs, device=agent.device, dtype=torch.float32)
                action = agent.act(obs_tensor, training=False)

                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                total_reward[ep] += reward
                obs = next_obs

            if config.use_wandb:
                returns_table.add_data(int(t), str(task), int(ep), float(total_reward[ep]))

        # Compute statistics after all episodes for this task
        mean_r = float(np.mean(total_reward))
        std_r = float(np.std(total_reward))

        if config.use_wandb:
            wandb.log({
                f"eval/{task}/reward_mean": mean_r,
                f"eval/{task}/reward_std": std_r,
            }, step=t)

        eval_env.close()

    if config.use_wandb:
        # Log the per-episode returns table once per eval step
        wandb.log({
            "eval/returns_table": returns_table,
        }, step=t)



if __name__ == "__main__":

    config = tyro.cli(Config)

    if config.use_wandb:
        wandb.init(
            project=config.proj_name,
            group=f"{config.env.domain_name}-{config.env.task_list}-s{config.env.seed}",
            name=f"{config.algo_name}-s{config.seed}",
            config=asdict(config)
        )

    if isinstance(config.env.task_list, str):
        task_list = get_task_sequence(config.env.domain_name, config.env.task_list)
    else:
        task_list = config.env.task_list

    print(f"Task list: {task_list}")

    env = make_env(
        env_id=config.env.domain_name,
        task=task_list[0],
        max_episode_steps=config.env.max_episode_steps,
        seed=config.env.seed,
    )

    assert isinstance(env.action_space, gym.spaces.Discrete) or hasattr(env.action_space, 'n'), print("Actions must be discrete")

    s_dim = env.observation_space.shape
    a_dim = env.action_space.n

    config.s_dim = s_dim
    config.a_dim = a_dim

    agent = PTDQN(config)

    replay_buffer = ReplayBuffer(
        buffer_size=config.buffer_size,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=config.device
    )

    cumulative_reward = 0.0
    steps_per_task = max(1, int(config.env.steps_per_task))
    total_steps = int(config.num_steps) if config.num_steps > 0 else len(task_list) * steps_per_task
    global_step = 0

    for task_idx, task_name in enumerate(task_list):
        if global_step >= total_steps:
            break
        env.close()
        env = make_env(
            env_id=config.env.domain_name,
            task=task_name,
            max_episode_steps=config.env.max_episode_steps,
            seed=config.env.seed,
        )
        task_steps = 0
        state, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        while task_steps < steps_per_task and global_step < total_steps:
            if global_step > 0 and global_step % config.eval_freq_steps == 0:
                print(f"\n[Step {global_step}] Running evaluation on all tasks...")
                evaluate_all_tasks(agent, config, global_step)

            if (
                global_step % config.train_freq_steps == 0
                and len(replay_buffer) >= config.batch_size
                and global_step >= config.warmup_steps
            ):
                metrics = agent.update(replay_buffer)
                if metrics is not None and config.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)

            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_steps += 1
            if episode_steps >= config.env.max_episode_steps:
                truncated = True
            replay_buffer.add(state, next_state, action, reward, terminated, truncated)

            episode_reward += reward
            cumulative_reward += reward
            state = next_state
            global_step += 1
            task_steps += 1

            if terminated or truncated:
                if config.use_wandb:
                    wandb.log(
                        {
                            "metrics/cumulative_reward": cumulative_reward,
                            "metrics/log_cumulative_reward": np.log(max(1e-8, cumulative_reward)),
                            "metrics/reward_per_episode": episode_reward,
                            "metrics/task_id": task_idx,
                        },
                        step=global_step,
                    )
                state, _ = env.reset()
                episode_reward = 0.0
                episode_steps = 0



