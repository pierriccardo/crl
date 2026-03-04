#!/usr/bin/env python3
"""Run commands in parallel. Import run_parallel() or use as script."""

import os
import queue
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def free_gpus(min_free_mib: int = 10_000) -> list[int]:
    """Return GPU indices that have at least *min_free_mib* MiB free."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    result = []
    for line in out.strip().splitlines():
        idx, free = line.split(",")
        if int(free.strip()) >= min_free_mib:
            result.append(int(idx.strip()))
    return result


def run_parallel(
    commands,
    cwd=None,
    max_workers=None,
    shell=False,
    quiet=False,
    gpus: list[int] | None = None,
    jobs_per_gpu: int = 1,
):
    """
    Run commands in parallel.

    - commands: list of str (one shell command each) or list of list (argv per command).
      If shell is True, str is passed to shell; otherwise list is passed to Popen(args).
    - cwd: working directory (default: repo root).
    - max_workers: max concurrent jobs (default: len(commands), or len(gpus)*jobs_per_gpu
      when gpus is set).
    - shell: if True, each str command is run with shell=True.
    - quiet: if True, suppress stdout/stderr of children.
    - gpus: list of GPU indices to use.  Each job is assigned an exclusive GPU via
      CUDA_VISIBLE_DEVICES.  Pass gpus="auto" to detect free GPUs automatically.
    - jobs_per_gpu: how many concurrent jobs to allow per GPU slot (default 1).

    Returns list of returncodes (same order as commands).
    """
    cwd = cwd or ROOT

    if gpus == "auto":
        gpus = free_gpus()
        if not gpus:
            raise RuntimeError("No free GPUs found (nvidia-smi returned none above threshold).")

    gpu_queue: queue.Queue | None = None
    if gpus is not None:
        gpu_queue = queue.Queue()
        for gpu_id in gpus:
            for _ in range(jobs_per_gpu):
                gpu_queue.put(gpu_id)
        if max_workers is None:
            max_workers = len(gpus) * jobs_per_gpu

    max_workers = max_workers or len(commands)

    base_kwargs: dict = {"cwd": cwd}
    if quiet:
        base_kwargs["stdout"] = subprocess.DEVNULL
        base_kwargs["stderr"] = subprocess.DEVNULL

    def run_one(cmd):
        kwargs = dict(base_kwargs)
        env = os.environ.copy()
        # Prevent JAX from pre-allocating 75% of GPU memory so that
        # JAX and PyTorch can coexist without starving each other.
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        if gpu_queue is not None:
            gpu_id = gpu_queue.get()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        kwargs["env"] = env
        try:
            if isinstance(cmd, str):
                return subprocess.run(cmd, shell=True, **kwargs).returncode
            return subprocess.run(cmd, shell=False, **kwargs).returncode
        finally:
            if gpu_queue is not None:
                gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, c) for c in commands]
        return [futures[i].result() for i in range(len(futures))]


def main():
    # Example: python scripts/run_parallel.py -- "python crl/algos/ppo.py --seed 0" "python crl/algos/ppo.py --seed 1"
    # Or generate commands in Python and call run_parallel(commands) from your script.
    import argparse
    p = argparse.ArgumentParser(description="Run commands in parallel")
    p.add_argument("commands", nargs="+", help="Commands to run (each as one argument)")
    p.add_argument("-j", "--jobs", type=int, default=None, help="Max parallel jobs")
    p.add_argument("--cwd", type=Path, default=ROOT, help="Working directory")
    p.add_argument("-q", "--quiet", action="store_true", help="Suppress child stdout/stderr")
    args = p.parse_args()
    codes = run_parallel(args.commands, cwd=args.cwd, max_workers=args.jobs, shell=True, quiet=args.quiet)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
