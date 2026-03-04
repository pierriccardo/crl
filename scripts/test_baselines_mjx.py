from run_parallel import run_parallel


ENVS = [
    ("mjx/cheetah", "transfer"),
    ("mjx/walker", "transfer"),
    ("mjx/humanoid", "default"),
]

N_SEEDS = 1
NUM_ENVS = 64
STEPS_PER_TASK = 20_000_000
MAX_EPISODE_STEPS = 1000

# Fair-comparison preset:
# - Match update cadence in env-steps across algorithms.
# - Keep similar batch scale per optimization step.
VARIBAD_ROLLOUT_STEPS = 16                 # varibad --batch_size (T)
VARIBAD_MINIBATCH_SIZE = 256
FB_TRAIN_FREQ_STEPS = NUM_ENVS * VARIBAD_ROLLOUT_STEPS
FB_TRAIN_BATCH_SIZE = NUM_ENVS * VARIBAD_ROLLOUT_STEPS
DO_EVAL = True
NUM_EVAL_EPISODES = 5


if __name__ == "__main__":
    commands = []
    for env, task_list in ENVS:
        common_args = [
            f"--env.domain_name={env}",
            f"--env.task_list={task_list}",
            f"--env.num_envs={NUM_ENVS}",
            f"--env.steps_per_task={STEPS_PER_TASK}",
            f"--env.max_episode_steps={MAX_EPISODE_STEPS}",
            "--env.seed=0",
            f"--num-eval-episodes={NUM_EVAL_EPISODES}",
            # "--no-use-wandb",
        ]
        eval_flag = "--do-eval" if DO_EVAL else "--no-do-eval"

        # FB-CPR
        for s in range(N_SEEDS):
            command = "uv run python3"
            command += " crl/algos/fb_cpr.py"
            command += f" --seed {s}"
            command += " --model.device cuda"
            command += " --buffer_size 1_000_000"
            command += f" --train.batch_size {FB_TRAIN_BATCH_SIZE}"
            command += f" --train_freq_steps {FB_TRAIN_FREQ_STEPS}"
            command += " --z_inference_samples 10_000"
            command += f" {eval_flag} "
            command += " ".join(common_args)
            commands.append(command)

        # VariBAD
        for s in range(N_SEEDS):
            command = "uv run python3"
            command += " crl/algos/varibad.py"
            command += f" --seed {s} "
            command += " --device cuda "
            command += f" --batch_size {VARIBAD_ROLLOUT_STEPS} "
            command += f" --minibatch_size {VARIBAD_MINIBATCH_SIZE} "
            command += " --z_dim 100 "
            command += f" {eval_flag} "
            command += " ".join(common_args)
            commands.append(command)

    print(
        "[run] | "
        f"num_envs={NUM_ENVS}, varibad_update_every={NUM_ENVS * VARIBAD_ROLLOUT_STEPS}, "
        f"fb_update_every={FB_TRAIN_FREQ_STEPS}, "
        f"varibad_batch={NUM_ENVS * VARIBAD_ROLLOUT_STEPS}, fb_batch={FB_TRAIN_BATCH_SIZE}"
    )
    print(f"[run] Launching {len(commands)} commands")
    run_parallel(commands, shell=True, quiet=False, gpus="auto", jobs_per_gpu=1)
