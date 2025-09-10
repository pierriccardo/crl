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
import numpy as np
import gymnasium as gym
from collections import deque

from torch import nn
from torch.optim import Adam
from dataclasses import dataclass, asdict

from crl.buffers import ReplayBuffer
from crl.envs import make_env, get_task_sequence


# ==============================
# Parameters
# ==============================
@dataclass
class Args:
    results_dir: str = "results"
    """Directory to save results"""
    model_dir: str = "models"
    """Directory to save models"""

    algo_name: str = os.path.basename(__file__).split('.')[0]
    """Name of the algorithm"""
    env_name: str = "goalenv"
    """Name of the environment"""
    task_sequence: str = 'cardinal'
    """Name of the task, or sequence of tasks for continual"""
    rb_size: int = 10**6
    """Replay buffer size"""
    seed: int = 0
    """Random seed"""
    load: bool = False
    """Whether to load model from path or train from scratch"""
    log_freq: int = 1000
    """Log loss info every log_freq train steps."""
    eval_freq: int = 1000
    """Evaluate on task every eval_freq train steps."""

    # Dual-timescale learning rates
    lr_permanent: float = 2e-4  # Slower learning for permanent component
    lr_transient: float = 5e-3  # Faster learning for transient component
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.2
    epsilon_decay: float = 0.999

    buffer_size: int = 100_000
    target_update_freq: int = 1000
    tau: float = 1.0  # Hard update coefficient for target network
    batch_size: int = 128
    train_steps_per_task: int = 8_000
    start_steps_per_task: int = 2000
    eval_episodes: int = 5  # Number of episodes for evaluation

    # Dual-timescale specific parameters
    adaptation_threshold: float = 0.1     # Change detection (lowered for sensitivity)
    prediction_window: int = 50   # Window for computing prediction error variance (smaller for faster detection)
    min_transient_weight: float = 0.05
    max_transient_weight: float = 0.95
    epsilon_restore_threshold: float = 0.2  # Lower threshold for epsilon restoration

    device: str = "mps"
    weight_init_noise: float = 0.1

    def __post_init__(self):
        """Set up task_list from task_sequence after initialization."""
        self.task_list = get_task_sequence(self.env_name, self.task_sequence)
        self.model_path = f"{self.model_dir}/{self.env_name}/{self.task_sequence}/{self.algo_name}/{self.seed}.pt"
        self.results_path = f"{self.results_dir}/{self.env_name}/{self.task_sequence}/{self.algo_name}/{self.seed}"


# ==============================
# Networks
# ==============================

class DualTimescaleQNetwork(nn.Module):
    """Q-Network with separate permanent and transient components"""

    def __init__(self, env):
        super().__init__()
        # Architecture for image inputs with shape (H, W, C)
        s_dim = env.observation_space.shape
        a_dim = env.action_space.n
        height, width, channels = s_dim

        # Store normalization bounds for general Box environments
        self.obs_low = torch.FloatTensor(env.observation_space.low)
        self.obs_high = torch.FloatTensor(env.observation_space.high)

        # Shared convolutional layers
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Shared fully connected layers
        conv_out_size = 32 * height * width
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
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
        # Conv & hidden Linear: Kaiming for ReLU, bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Output layers: small uniform near 0
        for head in [self.permanent_head, self.transient_head]:
            output_layer = head[-1]
            nn.init.uniform_(output_layer.weight, -1e-3, 1e-3)
            nn.init.constant_(output_layer.bias, 0.0)

    def forward(self, x, return_components=False):
        # Handle input shape: expect (batch, H, W, C) or (H, W, C)
        # Convert to PyTorch conv format: (batch, C, H, W)

        # Add batch dimension if needed
        if len(x.shape) == 3:  # (H, W, C)
            x = x.unsqueeze(0)  # (1, H, W, C)

        # Permute to PyTorch conv format: (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Simple normalization for MiniAtar (already in [0,1] range)
        if x.min() < 0 or x.max() > 1:
            # Normalize to [0,1] if not already
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)

        # Shared feature extraction
        x = self.shared_conv(x)
        features = self.shared_fc(x)

        # Compute permanent and transient Q-values
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

    def __init__(self, env: gym.Env, config: Args):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        self.s_dim = env.observation_space.shape
        self.a_dim = env.action_space.n

        # Networks
        self.q_network = DualTimescaleQNetwork(env).to(self.device)
        self.target_network = DualTimescaleQNetwork(env).to(self.device)

        # Separate optimizers for different learning rates
        permanent_params = (list(self.q_network.shared_conv.parameters()) +
                            list(self.q_network.shared_fc.parameters()) +
                            list(self.q_network.permanent_head.parameters()))
        transient_params = list(self.q_network.transient_head.parameters())

        self.permanent_optimizer = Adam(permanent_params, lr=config.lr_permanent)
        self.transient_optimizer = Adam(transient_params, lr=config.lr_transient)

        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.config.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device
        )

        # TODO: adjust epsilon every change detection, otherwise it will shrink
        # Training state
        self.epsilon = config.epsilon
        self.global_step = 0

        # Change detection and adaptation
        self.prediction_errors = deque(maxlen=config.prediction_window)
        self.transient_weight = 0.5  # Initial weight for transient component
        self.change_signal = 0.0

        # Data collection - single metrics dict
        self.metrics = {
            'training': [],   # Training losses and diagnostics
            'eval': [],       # Periodic evaluations during training (learning curves)
            'continual': [],  # Task completion evaluations (for FT, BT, forgetting)
            'config': asdict(self.config)
        }

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

        # Convert to change signal
        raw_signal = min(cv / self.config.adaptation_threshold, 1.0)

        # Smooth the signal
        prev_change_signal = self.change_signal
        self.change_signal = 0.9 * self.change_signal + 0.1 * raw_signal

        # Debug: Print change detection info periodically
        if len(self.prediction_errors) % 20 == 0:  # Every 20 updates
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
            return random.randrange(self.a_dim)

        with torch.no_grad():
            # Convert state to tensor
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)

            # Get Q-values from both components
            q_permanent, q_transient = self.q_network(state_tensor, return_components=True)

            # Combine Q-values using adaptive weighting
            q_combined = ((1 - self.transient_weight) * q_permanent +
                          self.transient_weight * q_transient)

            return q_combined.argmax().item()

    def train_step(self):
        """Perform one training step with dual timescales"""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample batch
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.config.batch_size)

        # Get current Q-values from both components
        q_permanent, q_transient = self.q_network(states, return_components=True)

        # Select Q-values for taken actions
        q_permanent_selected = q_permanent.gather(1, actions.unsqueeze(1))
        q_transient_selected = q_transient.gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            next_q_permanent, next_q_transient = self.target_network(next_states, return_components=True)

            # For target, use current transient weight
            next_q_combined = ((1 - self.transient_weight) * next_q_permanent +
                               self.transient_weight * next_q_transient)

            next_q_max = next_q_combined.max(1)[0].unsqueeze(1)
            target_q_values = (rewards.unsqueeze(1) +
                               (self.config.gamma * next_q_max * ~dones.unsqueeze(1)))

        # Compute losses for both components
        criterion = nn.SmoothL1Loss()
        permanent_loss = criterion(q_permanent_selected, target_q_values)
        transient_loss = criterion(q_transient_selected, target_q_values)

        # Update change detector
        combined_q = ((1 - self.transient_weight) * q_permanent_selected +
                      self.transient_weight * q_transient_selected)
        td_error = (target_q_values - combined_q).mean().item()
        self._update_change_detection(td_error)

        # Backpropagation with different learning rates
        # Clear gradients for both optimizers
        self.permanent_optimizer.zero_grad()
        self.transient_optimizer.zero_grad()

        # Compute total loss and backpropagate once
        total_loss = permanent_loss + transient_loss
        total_loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_value_(
            list(self.q_network.shared_conv.parameters()) +
            list(self.q_network.shared_fc.parameters()) +
            list(self.q_network.permanent_head.parameters()), 100)
        torch.nn.utils.clip_grad_value_(
            list(self.q_network.transient_head.parameters()), 100)

        # Update both optimizers
        self.permanent_optimizer.step()
        self.transient_optimizer.step()

        # Return diagnostic information
        diagnostics = {
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

        return diagnostics

    def train(self):
        """Main training loop with task-based structure"""
        step = 0  # Global step

        for task_idx, task_id in enumerate(self.config.task_list):
            task_env = make_env(self.config.env_name, task=task_id)
            state, _ = task_env.reset()

            for task_step in range(self.config.train_steps_per_task):
                # Select action
                action = self.act(state, training=True)

                next_state, reward, terminated, truncated, _ = task_env.step(action)
                self.replay_buffer.add(state, next_state, action, reward, terminated, truncated)
                state = next_state

                # Reset if episode ended
                if terminated or truncated:
                    state, _ = task_env.reset()

                # Train after initial steps
                if task_step >= self.config.start_steps_per_task:
                    diagnostics = self.train_step()

                    if diagnostics is not None:
                        wandb.log({
                            "train/permanent_loss": diagnostics['permanent_loss'],
                            "train/transient_loss": diagnostics['transient_loss'],
                            "train/combined_loss": diagnostics['combined_loss'],
                            "train/td_error": diagnostics['td_error'],
                            "train/change_signal": diagnostics['change_signal'],
                            "train/transient_weight": diagnostics['transient_weight'],
                            "train/q_permanent_mean": diagnostics['q_permanent_mean'],
                            "train/q_transient_mean": diagnostics['q_transient_mean'],
                            "train/q_combined_mean": diagnostics['q_combined_mean'],
                            "train/target_mean": diagnostics['target_mean'],
                            "train/reward_mean": diagnostics['reward_mean'],
                            "train/reward_max": diagnostics['reward_max'],
                            "train/reward_min": diagnostics['reward_min'],
                            "train/epsilon": self.epsilon,
                            "train/buffer_size": len(self.replay_buffer),
                        }, step=step)

                        # Save training metrics
                        self.metrics['training'].append({
                            'step': step,
                            'task': task_id,
                            'task_index': task_idx,
                            **diagnostics,
                            'epsilon': self.epsilon,
                            'change_signal': self.change_signal,
                            'transient_weight': self.transient_weight,
                            'buffer_size': len(self.replay_buffer)
                        })

                    # Evaluate periodically for monitoring
                    if step % self.config.eval_freq == 0:
                        avg_reward, avg_std = self.evaluate(task_id)

                        wandb.log({
                            "eval/avg_reward": avg_reward,
                            "eval/reward_std": avg_std,
                        }, step=step)

                        # Save periodic evaluation on CURRENT task (for learning curves)
                        self.metrics['eval'].append({
                            'step': step,
                            'task_learned': task_id,
                            'task_evaluated': task_id,  # Same task
                            'reward_mean': avg_reward,
                            'reward_std': avg_std
                        })

                        loss_val = diagnostics['combined_loss'] if diagnostics else 0.0
                        print("-"*50)
                        print(f"[Step] {task_step}/{self.config.train_steps_per_task} "
                              f"[Task] {task_idx}/{len(self.config.task_list)}, "
                              f"Loss {loss_val:.3f} Reward {avg_reward:.3f} ± {avg_std:.3f} "
                              f"Epsilon: {self.epsilon:.3f}")
                        if diagnostics:
                            print(f"Change Signal: {diagnostics['change_signal']:.3f}, "
                                  f"Transient Weight: {diagnostics['transient_weight']:.3f}")
                            print(f"P-Loss: {diagnostics['permanent_loss']:.3f}, "
                                  f"T-Loss: {diagnostics['transient_loss']:.3f}")

                    # Target network update
                    if step % self.config.target_update_freq == 0:
                        for t_param, q_param in zip(self.target_network.parameters(),
                                                    self.q_network.parameters()):
                            t_param.data.copy_(
                                self.config.tau * q_param.data +
                                (1.0 - self.config.tau) * t_param.data
                            )

                    # Decay epsilon
                    if self.epsilon > self.config.epsilon_min:
                        self.epsilon *= self.config.epsilon_decay

                    step += 1

            # Evaluation for continual metrics
            print(f"\nCompleted task {task_idx}/{len(self.config.task_list) - 1}: {task_id} evaluating on all tasks...")
            for eval_task_id in self.config.task_list:
                avg_reward, std_reward = self.evaluate(eval_task_id)

                self.metrics['continual'].append({
                    'step': step,
                    'task_learned': task_id,
                    'task_evaluated': eval_task_id,  # Different tasks
                    'reward_mean': avg_reward,
                    'reward_std': std_reward
                })
                wandb.summary[f"eval/{eval_task_id}_after_{task_id}"] = avg_reward

        os.makedirs(self.config.results_path, exist_ok=True)
        with open(f"{self.config.results_path}/data.json", 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)

    def evaluate(self, task_id):
        """Evaluate the agent"""
        total_rewards = []
        eval_env = make_env(self.config.env_name, task=task_id)

        try:
            for episode in range(self.config.eval_episodes):
                state, _ = eval_env.reset()
                episode_reward = 0

                while True:
                    action = self.act(state, training=False)
                    next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                    state = next_obs
                    episode_reward += reward

                    if terminated or truncated:
                        break

                total_rewards.append(episode_reward)

            avg_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            print(f"[Eval] task {task_id}: {avg_reward:.2f} ± {std_reward:.2f} on {self.config.eval_episodes}")

            return avg_reward, std_reward
        finally:
            # Always clean up the environment
            eval_env.close()

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


if __name__ == "__main__":
    args = tyro.cli(Args)

    wandb.init(
        project="crl",
        group=f"{args.algo_name}",
        name=f"{args.env_name}_{args.task_sequence}_{args.algo_name}_{args.seed}",
        config=asdict(args)
    )
    env = make_env(args.env_name)

    agent = PTDQN(env, args)
    if args.load:
        agent.load(args.model_path)
    else:
        agent.train()
        agent.save(args.model_path)
