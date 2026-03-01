import sys
import tyro
import dataclasses
from run_parallel import run_parallel
from crl.envs import get_task_sequence, list_task_sequences



ENVS = [
    ("mjx/walker", "forgetting"),
]

@dataclasses.dataclass
class Args:
    env_name: str = "highway_parking"
    task_list: str = "tasks_full"
    n_seeds: int = 3


if __name__ == "__main__":


    args = tyro.cli(Args)
    commands = []
    for env, task_list in ENVS:
            #print(f"Running {env} {task_list}")
            common_args = [
                f"--env.domain_name={env}",
                f"--env.task_list={task_list}",
                "--env.steps_per_task=1_000_000",
                "--env.max_episode_steps=1000",
                "--env.seed=0",
                #"--no-use-wandb",
            ]

            # FB-CPR
            for s in range(args.n_seeds):
                command = "uv run python3"
                command += " crl/algos/fb_cpr.py"
                command += f" --seed {s}"
                command += f" --model.device cuda"
                command += f" --no-do-eval "
                command += " ".join(common_args)
                commands.append(command)


            # Varibad
            for s in range(args.n_seeds):
                command = "python3"
                command += " crl/algos/varibad.py"
                command += f" --seed {s} "
                command += " ".join(common_args)
                commands.append(command)

    codes = run_parallel(commands, max_workers=40, shell=True, quiet=False)
    print(codes)  # [0, 0, 0, 0]
