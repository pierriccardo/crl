"""
Implementation of the DQN algorithm.

References:
- Paper: https://arxiv.org/abs/1312.5602 (Playing Atari with Deep Reinforcement Learning)
- paper: https://arxiv.org/abs/1509.06461 (Double DQN)
"""
# TODO: add double DQN?

import os
import json
import tyro
import torch
import wandb
import random
import numpy as np
import gymnasium as gym

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
    task_sequence: str = "cardinal"
    """Name of the sequence of tasks for continual"""
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

    lr: float = 5e-4  # Increased learning rate for faster adaptation
    gamma: float = 0.99  # Reduced gamma for better stability with sparse rewards
    epsilon: float = 1.0
    epsilon_min: float = 0.1  # Higher min epsilon for more exploration
    epsilon_decay: float = 0.9995  # Much slower decay to maintain exploration

    buffer_size: int = 50_000  # Reduced buffer size
    target_update_freq: int = 500   # Less frequent target updates for stability
    tau: float = 1.0  # Soft update coefficient for target network
    batch_size: int = 64       # Larger batch size for more stable updates
    train_steps_per_task: int = 4_000  # Train steps per task
    start_steps_per_task: int = 1000    # Warmup steps per task
    eval_episodes: int = 5  # Number of episodes for evaluation

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


class QNetwork(nn.Module):

    def __init__(self, env):
        super().__init__()
        # Architecture for image inputs with shape (H, W, C)
        s_dim = env.observation_space.shape
        a_dim = env.action_space.n
        height, width, channels = s_dim

        # Store normalization bounds for general Box environments
        self.obs_low = torch.FloatTensor(env.observation_space.low)
        self.obs_high = torch.FloatTensor(env.observation_space.high)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),  # (H, W) -> (H, W)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (H, W) -> (H, W)
            nn.ReLU(),
        )

        # Calculate the flattened size: 32 channels * height * width
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * height * width, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # 1) Conv & hidden Linear: Kaiming for ReLU, bias=0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # 2) Output layer: small uniform near 0 (overrides the line above)
        out = self.fc_layers[-1]
        nn.init.uniform_(out.weight, -1e-3, 1e-3)
        nn.init.constant_(out.bias, 0.0)

    def forward(self, x):
        # Handle input shape: expect (batch, H, W, C) or (H, W, C)
        # Convert to PyTorch conv format: (batch, C, H, W)

        # Add batch dimension if needed
        if len(x.shape) == 3:  # (H, W, C)
            x = x.unsqueeze(0)  # (1, H, W, C)

        # Permute to PyTorch conv format: (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # General normalization for any Box environment
        # Normalize to [0, 1] using the observation space bounds
        device = x.device
        obs_low = self.obs_low.to(device)
        obs_high = self.obs_high.to(device)

        # Reshape bounds to match conv format and input tensor shape
        # obs_low/obs_high are (H, W, C), need to match x which is (batch, C, H, W)
        batch_size = x.shape[0]

        # Permute bounds to (C, H, W) then add batch dimension
        obs_low = obs_low.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        obs_high = obs_high.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)

        # Handle case where low and high might be the same (avoid division by zero)
        range_vals = obs_high - obs_low
        range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)

        x = (x - obs_low) / range_vals
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ==============================
# DQN Agent
# ==============================
class DQN:
    """Deep Q-Network agent with experience replay and target network"""

    def __init__(self, env: gym.Env, config: Args):
        self.env = env
        self.config = config
        self.device = torch.device(config.device)

        self.s_dim = env.observation_space.shape
        self.a_dim = env.action_space.n  # is an integer in DQN

        # Networks
        self.q_network = QNetwork(env).to(self.device)
        self.target_network = QNetwork(env).to(self.device)

        self.optimizer = Adam(self.q_network.parameters(), lr=self.config.lr)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = ReplayBuffer(
            buffer_size=self.config.buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device
        )

        # Training state
        self.epsilon = config.epsilon
        self.step = 0  # Global step counter

        # Data collection - single metrics dict
        self.metrics = {
            'training': [],   # Training losses and diagnostics
            'eval': [],       # Periodic evaluations during training (learning curves)
            'continual': [],  # Task completion evaluations (for FT, BT, forgetting)
            'config': asdict(self.config)
        }

    def act(self, state, training: bool = False):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.a_dim)

        with torch.no_grad():
            # Convert state to tensor - network handles shape transformation internally
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)

            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample batch
        states, next_states, actions, rewards, dones = self.replay_buffer.sample(self.config.batch_size)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards.unsqueeze(1) + \
                (self.config.gamma * next_q_values * ~dones.unsqueeze(1))

        # Compute loss and update
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        # Return diagnostic information
        diagnostics = {
            'loss': loss.item(),
            'q_mean': current_q_values.mean().item(),
            'q_std': current_q_values.std().item(),
            'target_mean': target_q_values.mean().item(),
            'reward_mean': rewards.mean().item(),
            'reward_max': rewards.max().item(),
            'reward_min': rewards.min().item()
        }

        return diagnostics

    def train(self):

        step = 0  # Global step

        for task_idx, task_id in enumerate(self.config.task_list):
            task_env = make_env(self.config.env_name, task=task_id)

            state, _ = task_env.reset()

            for task_step in range(self.config.train_steps_per_task):

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
                            "train/loss": diagnostics['loss'],
                            "train/q_mean": diagnostics['q_mean'],
                            "train/q_std": diagnostics['q_std'],
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

                        loss_val = diagnostics['loss'] if diagnostics else 0.0
                        print("-"*50)
                        print(f"[Step] {task_step}/{self.config.train_steps_per_task} "
                              f"[Task] {task_idx}/{len(self.config.task_list)}, "
                              f"Loss {loss_val:.3f} Reward {avg_reward:.3f} ± {avg_std:.3f} "
                              f"Epsilon: {self.epsilon:.3f}")
                        if diagnostics:
                            print(f"Q-values: {diagnostics['q_mean']:.3f}±{diagnostics['q_std']:.3f}, "
                                  f"Targets: {diagnostics['target_mean']:.3f}, "
                                  f"Rewards: [{diagnostics['reward_min']:.3f}, {diagnostics['reward_max']:.3f}]")

                    # Target soft update
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
            print(f"\nCompleted task {task_idx}/{len(self.config.task_list) -1 }: {task_id} evaluating on all tasks...")
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
        # Get current task from training env
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
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step": self.step,
            "config": self.config.__dict__,
            "metrics": self.metrics,
        }
        torch.save(payload, path)

    def load(self, path):
        """Load the model"""
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_network"])
        self.target_network.load_state_dict(ckpt["target_network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.config.epsilon)
        self.step = ckpt.get("step", 0)

        # Load data collection state
        self.metrics = ckpt.get("metrics", {'training': [], 'eval': [], 'continual': []})


if __name__ == "__main__":
    args = tyro.cli(Args)

    print(args.model_path)

    wandb.init(
        project="crl",
        group=f"{args.algo_name}",
        name=f"{args.env_name}_{args.task_sequence}_{args.algo_name}_{args.seed}",
        config=asdict(args)
    )
    env = make_env(args.env_name)

    agent = DQN(env, args)
    if args.load:
        agent.load(args.model_path)
    else:
        agent.train()
        agent.save(args.model_path)
