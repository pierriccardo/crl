from run_parallel import run_parallel
from crl.envs import list_task_sequences, get_task_sequence


commands = []
for s in range(2):
    for task in get_task_sequence("highway_parking", "tasks_full"):
        command = "python3"
        command += " crl/algos/ppo.py"
        command += f" --env.domain_name highway_parking"
        command += f" --env.task {task}"
        command += f" --seed {s}"
        command += f" --batch_size 2048"
        command += f" --minibatch_size 64"
        command += f" --n_steps 1_000_000"
        command += f" --n_epochs 5"
        command += f" --ent_coef 0.01"
        command += f" --gamma 0.99"
        command += f" --gae_lambda 0.95"
        commands.append(command)

codes = run_parallel(commands, max_workers=10, shell=True, quiet=False)
print(codes)  # [0, 0, 0, 0]