import sys
import tyro
import dataclasses
from run_parallel import run_parallel
from crl.envs import list_envs, list_task_sequences

@dataclasses.dataclass
class Args:
    env_name: str = "highway_parking"
    task_list: str = "tasks_full"
    n_seeds: int = 5


if __name__ == "__main__":


    args = tyro.cli(Args)
    commands = []
    for task_list in list_task_sequences()[args.env_name].keys():
        print(f"Running {task_list}")
        common_args = [
            "--env.domain_name=highway_parking",
            f"--env.task_list={task_list}",
            "--env.task_switch_prob=0.01",
            "--env.max_episode_steps=1000",
            "--env.seed=0",
            "--num_episodes=10000",
            #"--no-use-wandb",
        ]

        # FB-CPR
        for s in range(args.n_seeds):
            command = "python3"
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
    #for command in commands:
    #    print(command)