# Continual Reinforcement Learning

Experiments with continual reinforcement learning: multiple tasks in sequence, with optional task switching and step-based or episodic regimes.

## Installation

1. Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
2. Create and activate the environment:
   ```bash
   micromamba create -f environment.yml -y
   micromamba activate crl
   pip install -r requirements.txt
   pip install -e .
   ```

## Algorithms

Each algorithm is a **single-file implementation** in `crl/algos/`. Logging uses **Weights & Biases (wandb)**.

- **`ppo.py`** — Proximal Policy Optimization (single-task or multi-task via env)
- **`dqn.py`** — DQN with experience replay (supports image obs)
- **`ptdqn.py`** — Permanent–Transient DQN
- **`varibad.py`** — VariBAD (variational task inference + PPO)
- **`fb.py`** — Forward–Backward (continual RL with backward/forward maps)
- **`fb_cpr.py`** — Forward–Backward with CPR

Run with `--help` for options, e.g.:
```bash
python crl/algos/ppo.py --help
python crl/algos/fb_cpr.py --help
```

## Environments

- **Single task:** `make_env(env_id, task=...)`
- **Continual (episodic):** `make_continual_episodic_env(env_id, task_list, ...)` — fixed-length episodes with optional task switching each reset.

Helpers: `list_envs()`, `get_task_sequence(env_name, sequence_name)`, `list_task_sequences(env_name)`.

Registered envs include: `minigrid`, `switchingdist`, `dmc/*`, `highway_parking`, `metaworld/*`, `mujoco/walker2d`, `mujoco/humanoid`, `coom`. Add or extend envs and sequences in `crl/envs/factory.py`.

Example:
```python
from crl.envs import make_continual_episodic_env, get_task_sequence

env = make_continual_episodic_env("highway_parking", "tasks_basic", max_episode_steps=500)
```

## Results and plotting
Experiment configs and metrics are logged to **wandb**. Local results layout and plotting (e.g. `scripts/plot.py`) may vary; see script help and `scripts/experiments.py` for run patterns.
