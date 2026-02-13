#!/usr/bin/env python3
"""Run commands in parallel. Import run_parallel() or use as script."""

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]


def run_parallel(
    commands,
    cwd=None,
    max_workers=None,
    shell=False,
    quiet=False,
):
    """
    Run commands in parallel.

    - commands: list of str (one shell command each) or list of list (argv per command).
      If shell is True, str is passed to shell; otherwise list is passed to Popen(args).
    - cwd: working directory (default: repo root).
    - max_workers: max concurrent jobs (default: len(commands)).
    - shell: if True, each str command is run with shell=True.
    - quiet: if True, suppress stdout/stderr of children.

    Returns list of returncodes (same order as commands).
    """
    cwd = cwd or ROOT
    max_workers = max_workers or len(commands)
    kwargs = {"cwd": cwd}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    def run_one(cmd):
        if isinstance(cmd, str):
            return subprocess.run(cmd, shell=True, **kwargs).returncode
        return subprocess.run(cmd, shell=False, **kwargs).returncode

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
