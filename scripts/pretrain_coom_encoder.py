"""
Pre-train a CNN encoder on COOM (Continual DOOM) observations.

Collects frames from the COOM env, trains an autoencoder (encoder + decoder)
with reconstruction loss, then saves only the encoder for use as a fixed
feature extractor in front of your algorithms (FB, Varibad, PPO).

Usage:
  pip install COOM  # required
  python scripts/pretrain_coom_encoder.py --sequence co8 --steps 100000 --out coom_encoder.pt
"""


import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Config:
    """Pre-train CNN encoder on COOM observations."""

    sequence: str = "co8"
    steps: int = 100_000
    batch_size: int = 64
    epochs: int = 20
    latent_dim: int = 256
    lr: float = 1e-3
    out: str = "coom_encoder.pt"
    seed: int = 0
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Encoder: CNN for (C, H, W) or (H, W, C) image -> latent_dim
# ---------------------------------------------------------------------------

class COOMEncoder(nn.Module):
    """CNN encoder for COOM observations (84x84, 4 stacked frames)."""

    def __init__(self, obs_shape: tuple, latent_dim: int = 256):
        super().__init__()
        # obs_shape can be (H, W, C) or (C, H, W)
        if len(obs_shape) == 3:
            c, h, w = obs_shape[0], obs_shape[1], obs_shape[2]
            # Gym/NumPy often (H, W, C)
            if c in (1, 3, 4) and h >= w:
                c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
        else:
            c, h, w = 4, 84, 84

        self._latent_dim = latent_dim
        self._c, self._h, self._w = c, h, w

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        # 84 -> 42 -> 21 -> 11 -> 4
        with torch.no_grad():
            x = torch.zeros(1, c, h, w)
            x = self.conv(x)
            flat = int(x.numel())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, latent_dim))

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # (B, H, W, C) -> (B, C, H, W)
        if x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.fc(x)


class COOMDecoder(nn.Module):
    """Simple decoder for reconstruction loss (training only)."""

    def __init__(self, latent_dim: int, out_shape: tuple):
        super().__init__()
        c, h, w = out_shape[0], out_shape[1], out_shape[2]
        if c not in (1, 3, 4):
            c, h, w = out_shape[2], out_shape[0], out_shape[1]
        self._size = c * h * w
        self._c, self._h, self._w = c, h, w
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self._size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).view(-1, self._c, self._h, self._w)


def obs_to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H,W,C) or (C,H,W) float/uint8 -> (1,C,H,W) float in [-1,1] or [0,1]."""
    x = torch.from_numpy(obs).float().to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] in (1, 3, 4):
        x = x.permute(0, 3, 1, 2)
    if x.max() > 1.5:
        x = x / 255.0 * 2.0 - 1.0
    return x


def collect_observations(env, steps: int, seed: int) -> np.ndarray:
    """Collect observations from env (random actions)."""
    obs_list = []
    obs, _ = env.reset(seed=seed)
    obs_list.append(obs)
    n = 1
    while n < steps:
        a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        obs_list.append(obs)
        n += 1
        if term or trunc:
            obs, _ = env.reset()
            obs_list.append(obs)
            n += 1
    return np.stack(obs_list[:steps], axis=0).astype(np.float32)


def main():
    config = tyro.cli(Config)
    device = torch.device(config.device)
    if config.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    from crl.envs import make_continual_episodic_env

    print("Creating COOM env ...")
    env = make_continual_episodic_env(
        "coom",
        config.sequence,
        steps_per_env=min(50_000, config.steps),
        seed=config.seed,
    )
    obs_space = env.observation_space
    obs_shape = obs_space.shape
    print(f"Observation shape: {obs_shape}")

    print("Collecting observations ...")
    data = collect_observations(env, config.steps, config.seed)
    env.close()
    print(f"Collected {len(data)} frames, dtype={data.dtype}, shape={data.shape}")

    encoder = COOMEncoder(obs_shape, latent_dim=config.latent_dim).to(device)
    decoder = COOMDecoder(config.latent_dim, obs_shape).to(device)
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.lr,
    )

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    def to_batch(b):
        x = b[0].to(device)
        if x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2)
        if x.max() > 1.5:
            x = x / 255.0 * 2.0 - 1.0
        return x

    print("Training autoencoder ...")
    for epoch in range(config.epochs):
        total_loss = 0.0
        for (batch,) in loader:
            x = to_batch(batch)
            z = encoder(x)
            recon = decoder(z)
            loss = nn.functional.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        n_batches = len(loader)
        print(f"  Epoch {epoch + 1}/{config.epochs}  loss={total_loss / n_batches:.4f}")

    os.makedirs(os.path.dirname(config.out) or ".", exist_ok=True)
    state = {
        "encoder": encoder.state_dict(),
        "obs_shape": obs_shape,
        "latent_dim": config.latent_dim,
    }
    torch.save(state, config.out)
    print(f"Saved encoder to {config.out} (obs_shape={obs_shape}, latent_dim={config.latent_dim})")


if __name__ == "__main__":
    main()
