from run_parallel import run_parallel


commands = []
for s in range(2):
    for env in ["Walker2d-v3", "HalfCheetah-v3", "Hopper-v3"]:
        command = "python3"
        command += " crl/algos/ppo.py"
        command += f" --env.domain_name {env}"
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