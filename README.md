# Continual Reinforcement Learning
Repository to experiment with Continual Reinforcement Learning (CRL)

# Getting Started

This project uses `uv` for dependency management. To get started:

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   or check [installation guide](https://docs.astral.sh/uv/getting-started/installation/).

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run commands** using `uv run`:
   ```bash
   uv run python3 crl/algos/dqn.py --help
   uv run python3 plots.py --help
   ```

# Environments and task definition
Environments can be imported from `envs`, a continual learning environment is created as a **sequence of single task environments**, in practice is a list of envs with different reward functions.
- A **task** is a `str` identifier for a particular goal e.g., for `goalenv` we have `north` specifying that task is reaching goal in the north, in `minatar` env a task is a complete game like `breakout`
- a **sequence** is a list of tasks done in continual fashion e.g., "cardinal" = ["north", "south", "east", "west"]

there are 4 main functions available:
- `make_env` create an env for a single task
- `list_envs` return available envs
- `get_task_sequence` retrieve a list of tasks given a sequence name
- `list_task_sequences` list available sequences

A single task env can be created as follows
```python
from crl.envs import make_env

env = make_env("goalenv", task="north")
```
Continual learning setup is in general defined by a for loop over single task envs, to better track metrics:

```python

@dataclass
class Args:
    env_name: str = 'minatar'
    task_sequence: str = 'classic'  # Get classic games sequence
    train_steps_per_task: int = 1000
    ...

args = tyro.cli(Args)

for task in get_task_sequence(args.env_name, args.task_sequence):
    env = make_env(args.env_name, task=task)

    state, _ = env.reset()
    for task_step in args.train_steps_per_task:
        ...
```
To get available list of envs and sequences, do the following:
```python
from crl.envs import list_envs, list_task_sequences

print(list_envs())
print(list_task_sequences('minatar'))
print(list_task_sequences())  # return all envs
```
To add new envs or sequences modify `crl/envs/factory.py`. Currently, the available envs are:
- goalenv (custom goal reaching minigrid environment)
- minatar


# Algorithm Structure

Each baseline algorithm in `crl/algos/` follows a consistent interface pattern. Every algorithm is implemented as a class with standardized methods for training, evaluation, and model persistence. Here's the common structure:

```python
@dataclass
class Args:
    """Algorithm configuration parameters"""
    # Common parameters
    env_name: str = "goalenv"
    task_sequence: str = "cardinal"
    seed: int = 0
    lr: float = 5e-4
    # ... algorithm-specific parameters

    def __post_init__(self):
        self.task_list = get_task_sequence(self.env_name, self.task_sequence)
        self.model_path = f"{self.model_dir}/{self.env_name}/{self.task_sequence}/{self.algo_name}/{self.seed}.pt"
        self.results_path = f"{self.results_dir}/{self.env_name}/{self.task_sequence}/{self.algo_name}/{self.seed}"

class AlgorithmName:
    """Main algorithm class"""

    def __init__(self, env: gym.Env, config: Args):
        """Initialize networks, optimizers, buffers, etc."""

    def act(self, state, training: bool = False) -> int:
        """Select action given state (epsilon-greedy for DQN-based)"""

    def train_step(self) -> Dict:
        """Perform one training step, return diagnostics"""

    def train(self):
        """Main training loop over all tasks"""

    def evaluate(self, task_id: str) -> tuple[float, float]:
        """Evaluate on specific task, return (mean_reward, std_reward)"""

    def save(self, path: str):
        """Save model checkpoint"""

    def load(self, path: str):
        """Load model checkpoint"""
```
## Usage
To run or inspect baselines do:
```python
# Running an algorithm
uv run python3 crl/algos/dqn.py --env-name goalenv --task-sequence cardinal --seed 0

# Get parameters options
uv run python3 crl/algos/dqn.py --help
```
Available Algorithms:
- **`dqn.py`**: Standard Deep Q-Network with experience replay
- **`ptdqn.py`**: Permanent-Transient DQN with dual timescales and change detection
- **`csp.py`**: Continual Subspace of Policies with adaptive epsilon scheduling


# Experiments results saving
Each experiment is saved as a single file under `results/` dir, univocally identified by: `env_name/task_sequence/algo_name/seed/data.json`. Each `data.json` contains the following data (some fields might vary depending on algorithm) regarding `"training"` metrics such as losses or values used for debug, `"eval"` metrics contains the evaluations performed during training on the current policy for a certain number of eval episodes, `"continual"` contains data to compute continual learning metrics after training (e.g., forgetting, forward transfer)these metrics are collected during training: every time a task ends we evaluate the agent on all the task. Finally, `"config"` contains all the parameters value used for that specific run. Below an example of one experiment log from `ptdqn.py`:
```json
{
  "training": [ # list of entries, one for each train step
    {
      "step": 0,
      "task": "north",
      "task_index": 0,
      "loss": 0.08370006829500198,
      "q_mean": 1.2798545867553912e-07,
      "q_std": 7.567967986688018e-05,
      "target_mean": -0.08038944005966187,
      "reward_mean": -0.08046874403953552,
      "reward_max": 5.0,
      "reward_min": -0.25,
      "epsilon": 1.0,
      "buffer_size": 501
    },
    ...
  ],
  "eval": [ # one entry for each evaluation step
    {
      "step": 0,
      "task_learned": "north",
      "task_evaluated": "north",
      "reward_mean": -35.00000000000004,
      "reward_std": 0.0
    },
    ...
  ],
  "continual": [ # for each task change, one entry per env n_tasks * n_tasks
    {
      "step": 4500,
      "task_learned": "north",
      "task_evaluated": "north",
      "reward_mean": 4.475,
      "reward_std": 0.0
    },
    {
      "step": 4500,
      "task_learned": "north",
      "task_evaluated": "south",
      "reward_mean": -50.025,
      "reward_std": 0.0
    },
    ...
  ],
  "config": { # algorithm params
    "results_dir": "results",
    "model_dir": "models",
    "algo_name": "ptdqn",
    "env_name": "goalenv",
    "task_sequence": "cardinal",
    "rb_size": 1000000,
    "seed": 0,
    "load": false,
    "log_freq": 1000,
    "eval_freq": 100,
    "lr_permanent": 0.0002,
    "lr_transient": 0.005,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.2,
    "epsilon_decay": 0.999,
    "buffer_size": 100000,
    "target_update_freq": 1000,
    "tau": 1.0,
    "batch_size": 128,
    "train_steps_per_task": 5000,
    "start_steps_per_task": 500,
    "eval_episodes": 5,
    "adaptation_threshold": 0.1,
    "prediction_window": 50,
    "min_transient_weight": 0.05,
    "max_transient_weight": 0.95,
    "epsilon_restore_threshold": 0.2,
    "device": "mps",
    "weight_init_noise": 0.1
  }
}
```

# Plotting Results

The `plots.py` script provides comprehensive visualization and analysis tools for experimental results. It can generate training curves, performance matrices, and continual learning metrics.

## Config and visualization functions
```python
@dataclass
class Args:
    results_dir: str = "results"           # Where to find experiment data (do not modify)
    save_path: str = "plots"               # Where to save plots
    env_name: str = "goalenv"              # Environment name
    task_sequence: str = "cardinal"        # Task sequence name
    algorithms: List[str] = None           # Algorithms to compare (default: ["dqn", "ptdqn", "csp"])
    seeds: List[int] = None                # Seeds to aggregate (default: [0])
```

1. **`plot_training_rewards`**: Training reward curves over time
2. **`plot_performance_matrices`**: Heatmaps showing continual learning performance
3. **`create_metrics_table`**: Continual learning metrics summary

### Usage
```bash
# Specify custom parameters
uv run python3 plots.py --env-name goalenv --task-sequence cardinal --algorithms dqn ptdqn csp --seeds 0 1 2
```
The script creates the following files in `plots/`:

### Continual Learning Metrics

The metrics table includes:
- **Forward Transfer**: Performance improvement on new tasks due to prior learning
- **Backward Transfer**: Change in performance on old tasks after learning new ones
- **Learning Accuracy**: Average performance on each task after learning it
- **Average Forgetting**: Negative backward transfer
