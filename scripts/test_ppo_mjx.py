from run_parallel import run_parallel

try:
    import torch
    N_GPUS = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
except Exception:
    N_GPUS = 0

commands = []
for idx, (s, env, task) in enumerate(
    [(s, e, t) for s in range(2) for e in ["mjx/walker"] for t in ["run", "walk", "stand", "noleg_left", "hugegravity"]]
):
    gpu_id = idx % N_GPUS if N_GPUS else None
    # JAX_PLATFORMS=cpu avoids "no supported devices found for platform CUDA" when jaxlib
    # CUDA is incompatible with the driver; MJX runs on CPU, PyTorch (PPO) still uses GPU.
    env_prefix = "JAX_PLATFORMS=cpu "
    if gpu_id is not None:
        env_prefix += f"CUDA_VISIBLE_DEVICES={gpu_id} "
    command = env_prefix + "python3"
    command += " crl/algos/ppo.py"
    command += f" --env.domain_name {env}"
    command += f" --env.task {task}"
    command += f" --seed {s}"
    command += f" --device cuda" if N_GPUS else " --device cpu"
    command += f" --batch_size 2048"
    command += f" --minibatch_size 64"
    command += f" --n_steps 1_000_000"
    command += f" --n_epochs 5"
    command += f" --ent_coef 0.01"
    command += f" --gamma 0.99"
    command += f" --gae_lambda 0.95"
    commands.append(command)

max_workers = min(len(commands), N_GPUS) if N_GPUS else len(commands)
codes = run_parallel(commands, max_workers=max_workers, shell=True, quiet=False)