from run_parallel import run_parallel


NUM_ENVS = 128  # parallel envs per task for faster rollout collection
TASKS = [
    "run",
    "walk",
    "stand",
    "hugegravity",
    "inverted_actions",
    "moon",
    "rainfall",
    "noleg_right",
    "noleg_left",
    "noknees",
    "noankles",
]
commands = []
for idx, (s, env, task) in enumerate(
    [(s, e, t) for s in range(1) for e in ["mjx/walker"] for t in TASKS]
):
    command = "python3"
    command += " crl/algos/ppo.py"
    command += f" --env.domain_name {env}"
    command += f" --env.task {task}"
    command += f" --env.num_envs {NUM_ENVS}"
    command += f" --seed {s}"
    command += f" --device cuda"
    command += f" --batch_size 1024"
    command += f" --minibatch_size 128"
    command += f" --n_steps 10_000_000"
    command += f" --n_epochs 5"
    command += f" --ent_coef 0.01"
    command += f" --gamma 0.99"
    command += f" --gae_lambda 0.95"
    commands.append(command)

codes = run_parallel(commands, shell=True, quiet=False, gpus="auto", max_workers=10)