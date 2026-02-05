#!/usr/bin/env python3
"""
Simplified plotting script for wandb metrics.

Usage:
  # Plot a single metric for a group
  python plot.py --group GROUP_NAME --metric train/loss

  # Plot 3 metrics in a composite plot
  python plot.py --group GROUP_NAME --composite --metrics train/loss train/acc test/acc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tyro
from scipy import stats


def _get_nested(d: Any, path: List[str]) -> Any:
    """Get nested value from dict."""
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _get_any(cfg: Dict[str, Any], candidates: List[List[str]]) -> Any:
    """Try multiple candidate paths to get config value."""
    for path in candidates:
        if len(path) == 1 and "." in path[0]:
            if path[0] in cfg:
                return cfg[path[0]]
            continue
        v = _get_nested(cfg, path)
        if v is not None:
            return v
    return None


def extract_meta(run: wandb.apis.public.Run) -> Dict[str, Any]:
    """Extract metadata from wandb run."""
    cfg = dict(run.config or {})
    cfg = {k: v for k, v in cfg.items() if not str(k).startswith("_")}

    algo_name = _get_any(cfg, [["algo_name"]])
    algo_seed = _get_any(cfg, [["seed"]])

    return {
        "run_id": run.id,
        "group": run.group,
        "algo_name": algo_name or "unknown",
        "algo_seed": algo_seed,
    }


def load_metric_data(
    run: wandb.apis.public.Run,
    metric: str,
    max_samples: int = 200_000,
) -> pd.DataFrame:
    """Load a single metric from a wandb run."""
    try:
        df = run.history(keys=[metric], pandas=True, samples=max_samples)
        if df.empty:
            return pd.DataFrame(columns=["step", "value"])

        step_col = "_step" if "_step" in df.columns else "step"
        if step_col not in df.columns:
            return pd.DataFrame(columns=["step", "value"])

        result = pd.DataFrame({
            "step": df[step_col],
            "value": df[metric] if metric in df.columns else np.nan
        })
        return result.dropna(subset=["value"])
    except Exception:
        return pd.DataFrame(columns=["step", "value"])


def fetch_group_data(
    api: wandb.Api,
    entity: str,
    project: str,
    group: str,
    metrics: List[str],
    max_samples: int = 200_000,
) -> pd.DataFrame:
    """Fetch metrics for all runs in a group."""
    runs = api.runs(f"{entity}/{project}", filters={"group": group})

    all_data = []
    for run in runs:
        meta = extract_meta(run)

        for metric in metrics:
            metric_data = load_metric_data(run, metric, max_samples)
            if metric_data.empty:
                continue

            metric_data["metric"] = metric
            metric_data["run_id"] = meta["run_id"]
            metric_data["algo_name"] = meta["algo_name"]
            metric_data["algo_seed"] = meta["algo_seed"]
            all_data.append(metric_data)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    confidence_level: float = 0.95,
    save_pdf: bool = False,
):
    """Plot a single metric with confidence intervals per algorithm."""
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        print(f"  No data for metric: {metric}")
        return

    # Aggregate by algorithm and step
    agg = (
        sub.groupby(["algo_name", "step"])["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )

    # Check if this is a task_id metric (needs special handling)
    is_task_id = "task_id" in metric.lower()

    # Compute confidence intervals (skip for task_id)
    alpha = 1 - confidence_level
    if not is_task_id:
        agg["margin"] = agg.apply(
            lambda row: stats.t.ppf(1 - alpha/2, row["n"] - 1) * row["std"] / np.sqrt(row["n"])
            if row["n"] > 1 else 0,
            axis=1
        )

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))

    colors = _get_style_colors()
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    algos = sorted(agg["algo_name"].unique())

    for i, algo in enumerate(algos):
        d = agg[agg["algo_name"] == algo].sort_values("step")
        color = colors[i % len(colors)]

        if is_task_id:
            # Task ID: step plot, no markers, no confidence intervals
            ax.plot(
                d["step"], d["mean"],
                label=f"{algo} (n={d['n'].iloc[0]})",
                linewidth=4.0,
                color=color,
                drawstyle='steps-post',
                marker='',
            )
        else:
            marker = markers[i % len(markers)]
            markevery = max(1, len(d) // 8)

            ax.plot(
                d["step"], d["mean"],
                label=f"{algo} (n={d['n'].iloc[0]})",
                linewidth=4.0,
                color=color,
                marker=marker,
                markevery=markevery,
                markersize=8,
                markeredgewidth=1.5,
                markerfacecolor=color,
                markeredgecolor='white',
            )
            ax.fill_between(
                d["step"],
                d["mean"] - d["margin"],
                d["mean"] + d["margin"],
                alpha=0.2,
                color=color,
            )

    ax.set_xlabel("Step")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} - {int(confidence_level*100)}% CI")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Format axes
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    if save_pdf:
        pdf_path = out_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved PDF: {pdf_path}")

    plt.close(fig)


def _parse_metric_name(metric: str) -> str:
    """Extract the last part of a metric name for shorter labels."""
    # Split by '/' and take the last part
    parts = metric.split('/')
    short_name = parts[-1]
    # Replace underscores with spaces
    short_name = short_name.replace('_', ' ')
    return short_name


def _sanitize_filename(name: str) -> str:
    """Sanitize a name for use in filenames by replacing problematic characters."""
    return name.replace('/', '_')


def _get_style_colors() -> List[str]:
    """Get colors from matplotlib's current prop_cycle (set by style file)."""
    # Colors from paper.mplstyle: ['E69F00', '56B4E9', '009E73', 'CC79A7', 'EE3377', '33BBEE', 'BBBBBB']
    # Try to extract from prop_cycle first
    try:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = []
        for props in prop_cycle:
            if 'color' in props:
                color = props['color']
                # Convert to hex format
                import matplotlib.colors as mcolors
                color_hex = mcolors.to_hex(color)
                colors.append(color_hex)
            if len(colors) >= 7:
                break
        if colors:
            return colors
    except:
        pass

    # Fallback: use colors from paper.mplstyle directly
    return ['#E69F00', '#56B4E9', '#009E73', '#CC79A7', '#EE3377', '#33BBEE', '#BBBBBB']


def plot_composite(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    confidence_level: float = 0.95,
    save_pdf: bool = False,
):
    """Plot 3 metrics vertically aligned."""
    if len(metrics) != 3:
        raise ValueError("Composite plot requires exactly 3 metrics")

    available_metrics = [m for m in metrics if m in df["metric"].unique()]
    if not available_metrics:
        print(f"  No data for requested metrics: {metrics}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True, gridspec_kw={'hspace': 0.05})

    colors = _get_style_colors()
    alpha = 1 - confidence_level

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sub = df[df["metric"] == metric].copy()

        if sub.empty:
            short_metric = _parse_metric_name(metric)
            ax.text(0.5, 0.5, f"No data for {short_metric}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(short_metric, labelpad=2, fontsize=8)
            continue

        # Aggregate by algorithm and step
        agg = (
            sub.groupby(["algo_name", "step"])["value"]
            .agg(mean="mean", std="std", n="count")
            .reset_index()
        )

        # Compute confidence intervals
        agg["margin"] = agg.apply(
            lambda row: stats.t.ppf(1 - alpha/2, row["n"] - 1) * row["std"] / np.sqrt(row["n"])
            if row["n"] > 1 else 0,
            axis=1
        )

        # Check if this is a task_id metric (needs special handling)
        is_task_id = "task_id" in metric.lower()
        is_cumulative_reward = "cumulative_reward" in metric.lower()

        # Plot each algorithm
        algos = sorted(agg["algo_name"].unique())
        markers = ['o', 's', '^', 'D', 'v', 'p', '*']
        for i, algo in enumerate(algos):
            d = agg[agg["algo_name"] == algo].sort_values("step")

            # Downsample if too many points (keep max 1000 points)
            # But skip downsampling for task_id as it represents time periods
            if not is_task_id and len(d) > 1000:
                step_size = len(d) // 1000
                d = d.iloc[::step_size].copy()

            color = colors[i % len(colors)]

            if is_task_id:
                # Task ID: step plot, no markers, no confidence intervals
                ax.plot(
                    d["step"], d["mean"],
                    label=f"{algo} (n={d['n'].iloc[0]})" if idx == 0 else "",
                    linewidth=1.0,
                    color=color,
                    drawstyle='steps-post',
                    marker='',
                )
            elif is_cumulative_reward:
                # Cumulative reward: thick line with markers
                marker = markers[i % len(markers)]
                markevery = max(1, len(d) // 8)  # Show ~8 markers
                ax.plot(
                    d["step"], d["mean"],
                    label=f"{algo} (n={d['n'].iloc[0]})" if idx == 0 else "",
                    linewidth=3.0,
                    color=color,
                    marker=marker,
                    markevery=markevery,
                    markersize=6,
                    markeredgewidth=1.0,
                    markerfacecolor=color,
                    markeredgecolor='white',
                )
                ax.fill_between(
                    d["step"],
                    d["mean"] - d["margin"],
                    d["mean"] + d["margin"],
                    alpha=0.2,
                    color=color,
                )
            else:
                # Regular metrics: line plot with confidence intervals, no markers
                ax.plot(
                    d["step"], d["mean"],
                    label=f"{algo} (n={d['n'].iloc[0]})" if idx == 0 else "",
                    linewidth=1.0,
                    color=color,
                    marker='',
                )
                ax.fill_between(
                    d["step"],
                    d["mean"] - d["margin"],
                    d["mean"] + d["margin"],
                    alpha=0.2,
                    color=color,
                )

        short_metric = _parse_metric_name(metric)
        ax.set_ylabel(short_metric, labelpad=2, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        if idx == 0:
            ax.legend(loc='best', fontsize=9)

    axes[-1].set_xlabel("Step", labelpad=2)
    axes[-1].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    axes[-1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out_path}")

    if save_pdf:
        pdf_path = out_path.with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  Saved PDF: {pdf_path}")

    plt.close(fig)


@dataclass
class Args:
    group: str  # Required: group name to filter runs
    entity: str = "pierriccardo"
    project: str = "continual-rl"
    metric: Optional[List[str]] = None  # Metric(s) to plot (creates separate plot for each)
    composite: bool = False  # Create composite plot
    metrics: Optional[List[str]] = None  # Metrics for composite plot (exactly 3)
    out_dir: str = "./imgs"
    confidence_level: float = 0.95
    max_samples: int = 200_000
    pdf: bool = False  # Also save plots as PDF


def main(args: Args):
    """Main function."""
    # Load paper style
    style_path = Path(__file__).parent / "paper.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
        print(f"Using paper style from: {style_path}")
    else:
        print(f"Warning: paper.mplstyle not found at {style_path}, using default style")

    if not args.composite and not args.metric:
        raise ValueError("Must specify either --metric or --composite")

    if args.composite:
        if not args.metrics or len(args.metrics) != 3:
            raise ValueError("--composite requires exactly 3 metrics via --metrics")
        metrics_to_fetch = args.metrics
    else:
        # Convert single metric to list if needed
        if isinstance(args.metric, str):
            metrics_to_fetch = [args.metric]
        else:
            metrics_to_fetch = args.metric

    # Fetch data
    print(f"Fetching data for group: {args.group}")
    print(f"Metrics: {metrics_to_fetch}")

    api = wandb.Api()
    df = fetch_group_data(
        api, args.entity, args.project, args.group,
        metrics_to_fetch, args.max_samples
    )

    if df.empty:
        raise RuntimeError("No data found for the specified group and metrics")

    print(f"Loaded {df['run_id'].nunique()} runs")
    print(f"Algorithms: {sorted(df['algo_name'].unique())}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    safe_group = _sanitize_filename(args.group)

    if args.composite:
        out_path = out_dir / f"plot_{safe_group}_composite.png"
        plot_composite(df, args.metrics, out_path, args.confidence_level, save_pdf=args.pdf)
    else:
        # Create a separate plot for each metric
        for metric in metrics_to_fetch:
            safe_metric = _sanitize_filename(metric)
            out_path = out_dir / f"plot_{safe_group}_{safe_metric}.png"
            plot_metric(df, metric, out_path, args.confidence_level, save_pdf=args.pdf)

    print(f"\nDone! Output saved to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
