import os
import json
import tyro
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from crl.envs import get_task_sequence

from dataclasses import dataclass
from typing import Dict, List, Optional

import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class Emoji:
    alert: str = '⚠️'
    no: str = '❌'
    ok: str = '✅'


@dataclass
class Args:

    results_dir: str = "results"
    save_path: str = "plots"
    env_name: str = "goalenv"
    task_sequence: str = "cardinal"
    algorithms: List[str] = None
    seeds: List[int] = None

    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["dqn", "ptdqn", "csp"]
        if self.seeds is None:
            self.seeds = [0]

        self.task_list = get_task_sequence(self.env_name, self.task_sequence)


def load_experiment_results(args: Args) -> Dict:
    """Load experiment results for multiple algorithms and seeds."""
    results = {}
    for algo in args.algorithms:
        results[algo] = {}
        for seed in args.seeds:
            results_path = os.path.join(
                args.results_dir,
                args.env_name,
                args.task_sequence,
                algo,
                str(seed),
                'data.json'
            )
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results[algo][seed] = json.load(f)
                print(f"{Emoji.ok} Loaded {algo} seed {seed}")
            else:
                print(f"{Emoji.no} Missing {results_path}")

    return results


def plot_training_rewards(args: Args, results, save_path: str):
    """Plot training reward curves with 95% confidence intervals."""
    # TODO: add confidence interval over different seeds as well

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("husl", len(args.algorithms))

    task_list = get_task_sequence(args.env_name, args.task_sequence)
    train_steps_single = results[args.algorithms[0]][args.seeds[0]]['config']['train_steps_per_task'] \
        - results[args.algorithms[0]][args.seeds[0]]['config']['start_steps_per_task']
    boundaries = [i * train_steps_single for i in range(len(task_list) + 1)]

    # Draw vertical lines at boundaries (skip first if you want)
    for b in boundaries[1:-1]:  # skip start and end
        plt.axvline(x=b, color='gray', linestyle='--')

    # Add labels centered under each segment
    ymin, ymax = plt.gca().get_ylim()
    for i in range(len(task_list)):
        # Midpoint of each segment
        xmid = (boundaries[i] + boundaries[i+1]) / 2
        plt.text(
            xmid, ymin - 0.1*(ymax - ymin),  # below x-axis
            task_list[i],
            ha='center', va='top'
        )

    for i, algo in enumerate(args.algorithms):

        data_per_seed = {}

        for seed in args.seeds:
            data = results[algo][seed]
            training_data = data['eval']

            n_exps = data['config']['eval_episodes']
            train_steps = len(get_task_sequence(args.env_name, args.task_sequence)) * \
                (data['config']['train_steps_per_task'] - data['config']['start_steps_per_task'])

            # Extract step and reward_mean
            data_per_seed[seed] = {
                'means': [],
                'stds': []
            }
            for entry in training_data:
                if 'reward_mean' in entry and 'step' in entry:
                    data_per_seed[seed]['means'].append(entry['reward_mean'])
                    data_per_seed[seed]['stds'].append(entry['reward_std'])

        if not data_per_seed:
            print(f" No training data found for {algo}")
            continue

        data = data_per_seed[0]
        means = np.array(data['means'])
        stds = np.array(data['stds'])

        ci = 1.96 * np.array(stds) / np.sqrt(n_exps)
        x = np.linspace(0, train_steps, len(means))
        plt.plot(x, means, label=algo)
        plt.fill_between(x, (means-ci), (means+ci), alpha=0.2, color=colors[i])

    plt.xlabel('Training Steps')
    plt.ylabel('Mean Reward')
    plt.title('Training Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Training rewards plot saved to {save_path}")
    plt.show()


def compute_performance_matrix(args: Args, continual_data: List[Dict]) -> np.ndarray:
    """Compute performance matrix from continual evaluation data."""
    n_tasks = len(args.task_list)
    matrix = np.full((n_tasks, n_tasks), np.nan)

    # Group evaluations by step (task completion events)
    evaluations_by_step = {}
    for entry in continual_data:
        step = entry['step']
        if step not in evaluations_by_step:
            evaluations_by_step[step] = []
        evaluations_by_step[step].append(entry)

    # Process each task completion in order
    for step in sorted(evaluations_by_step.keys()):
        step_entries = evaluations_by_step[step]

        # All entries at same step should have same task_learned
        task_learned = step_entries[0]['task_learned']

        if task_learned in args.task_list:
            learned_idx = args.task_list.index(task_learned)

            # Fill the row for this learned task
            for entry in step_entries:
                task_evaluated = entry['task_evaluated']
                if task_evaluated in args.task_list:
                    eval_idx = args.task_list.index(task_evaluated)
                    matrix[learned_idx, eval_idx] = entry['reward_mean']

    return matrix


def plot_performance_matrices(args: Args, results: dict, save_path: Optional[str] = None):
    """Plot performance matrices for each algorithm."""

    n_algos = len(args.algorithms)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 5))
    if n_algos == 1:
        axes = [axes]

    for i, algo in enumerate(args.algorithms):
        # Average performance matrix across seeds
        matrices = []
        for seed in args.seeds:
            if algo in results and seed in results[algo]:
                continual_data = results[algo][seed]['continual']
                if continual_data:  # Check if data exists
                    matrix = compute_performance_matrix(args, continual_data)
                    matrices.append(matrix)

        if matrices:
            avg_matrix = np.nanmean(matrices, axis=0)

            # Plot heatmap
            im = axes[i].imshow(avg_matrix, cmap='viridis', aspect='equal')
            axes[i].set_title(f'{algo.upper()} Performance Matrix')
            axes[i].set_xlabel('Task Evaluated')
            axes[i].set_ylabel('Task Learned')

            # Set tick labels
            axes[i].set_xticks(range(len(args.task_list)))
            axes[i].set_yticks(range(len(args.task_list)))
            axes[i].set_xticklabels([t.replace('_', ' ').title() for t in args.task_list], rotation=45)
            axes[i].set_yticklabels([t.replace('_', ' ').title() for t in args.task_list])

            # Disable grid lines to prevent them from overriding text annotations
            axes[i].grid(False)

            # Add text annotations
            for x in range(len(args.task_list)):
                for y in range(len(args.task_list)):
                    if not np.isnan(avg_matrix[y, x]):
                        text = f'{avg_matrix[y, x]:.2f}'
                        color = 'white' if avg_matrix[y, x] < np.nanmax(avg_matrix)/2 else 'black'
                        axes[i].text(x, y, text, ha='center', va='center', color=color)

            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        else:
            axes[i].text(0.5, 0.5, f'No data for {algo}', ha='center', va='center',
                        transform=axes[i].transAxes)
            axes[i].set_title(f'{algo.upper()} Performance Matrix')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"- Performance matrices saved to {save_path}")
    plt.show()


def compute_continual_metrics(args: Args, performance_matrix: np.ndarray) -> Dict:
    """Compute continual learning metrics from performance matrix."""
    n_tasks = len(args.task_list)

    # Learning Accuracy - average diagonal performance
    diagonal_values = []
    for i in range(n_tasks):
        if not np.isnan(performance_matrix[i, i]):
            diagonal_values.append(performance_matrix[i, i])
    learning_accuracy = np.mean(diagonal_values) if diagonal_values else 0

    # Forward Transfer - improvement on new tasks
    forward_transfer = 0
    ft_count = 0
    for i in range(1, n_tasks):  # Skip first task
        if not np.isnan(performance_matrix[i, i]):
            # Compare with baseline (assume 0 or use first task performance)
            baseline = 0
            forward_transfer += performance_matrix[i, i] - baseline
            ft_count += 1
    forward_transfer = forward_transfer / ft_count if ft_count > 0 else 0

    # Backward Transfer - final vs initial performance
    backward_transfer = 0
    bt_count = 0
    for i in range(n_tasks):
        final_perf = performance_matrix[-1, i]  # Last row
        initial_perf = performance_matrix[i, i]  # Diagonal
        if not np.isnan(final_perf) and not np.isnan(initial_perf):
            backward_transfer += final_perf - initial_perf
            bt_count += 1
    backward_transfer = backward_transfer / bt_count if bt_count > 0 else 0

    # Average Forgetting
    average_forgetting = -backward_transfer

    return {
        'forward_transfer': forward_transfer,
        'backward_transfer': backward_transfer,
        'learning_accuracy': learning_accuracy,
        'average_forgetting': average_forgetting
    }


def create_metrics_table(args: Args, results: Dict, save_path: Optional[str] = None) -> pd.DataFrame:
    """Create a summary table with continual learning metrics."""

    table_data = []

    for algo in args.algorithms:
        # Collect metrics across seeds
        all_metrics = []

        for seed in args.seeds:
            if algo in results and seed in results[algo]:
                continual_data = results[algo][seed]['continual']
                if continual_data:
                    matrix = compute_performance_matrix(args, continual_data)
                    metrics = compute_continual_metrics(args, matrix)
                    all_metrics.append(metrics)

        if all_metrics:
            # Average across seeds
            avg_metrics = {}
            for metric in ['forward_transfer', 'backward_transfer', 'learning_accuracy', 'average_forgetting']:
                values = [m[metric] for m in all_metrics if not np.isnan(m[metric])]
                avg_metrics[metric] = np.mean(values) if values else 0
                avg_metrics[f'{metric}_std'] = np.std(values) if len(values) > 1 else 0

            table_data.append({
                'Algorithm': algo.upper(),
                'Forward Transfer': f"{avg_metrics['forward_transfer']:.3f} ± {avg_metrics['forward_transfer_std']:.3f}",
                'Backward Transfer': f"{avg_metrics['backward_transfer']:.3f} ± {avg_metrics['backward_transfer_std']:.3f}",
                'Learning Accuracy': f"{avg_metrics['learning_accuracy']:.3f} ± {avg_metrics['learning_accuracy_std']:.3f}",
                'Average Forgetting': f"{avg_metrics['average_forgetting']:.3f} ± {avg_metrics['average_forgetting_std']:.3f}"
            })

    df = pd.DataFrame(table_data)

    # Display table
    print("\n" + "="*80)
    print("CONTINUAL LEARNING METRICS SUMMARY")
    print("="*80)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No data to display")
    print("="*80)

    if save_path and not df.empty:
        df.to_csv(save_path, index=False)
        print(f"- Metrics table saved to {save_path}")

    return df


def plot_all_results(args, output_dir: str = 'plots'):
    """Generate all plots and tables for the experiment results."""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    print(f"- Loading results for {args.env_name}/{args.task_sequence}...")
    results = load_experiment_results(args)

    if not results:
        print(f"{Emoji.no} No results found!")
        return

    print(f"- Generating plots...")

    # 1. Training rewards
    plot_training_rewards(
        args, results,
        save_path=f"{output_dir}/training_rewards_{args.env_name}_{args.task_sequence}.png"
    )

    # 2. Performance matrices
    plot_performance_matrices(
        args, results,
        save_path=f"{output_dir}/performance_matrices_{args.env_name}_{args.task_sequence}.png"
    )

    # 3. Metrics table
    create_metrics_table(
        args, results,
        save_path=f"{output_dir}/metrics_table_{args.env_name}_{args.task_sequence}.csv"
    )

    print(f"{Emoji.ok} All plots generated and saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    args = tyro.cli(Args)

    # Generate all plots using CLI arguments
    plot_all_results(args, output_dir="plots")
