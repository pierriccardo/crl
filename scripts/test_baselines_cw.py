import numpy as np
import os

from run_parallel import run_parallel
from crl.envs import make_vec_env, get_task_sequence


ENVS = [
    ("continualworld", "cw10"),
    #("continualworld", "cw20"),
]

N_SEEDS = 1
NUM_ENVS = 32
STEPS_PER_TASK = 2_000_000
MAX_EPISODE_STEPS = 500
RUN_VEC_SMOKE_TEST = False
SKIP_JOBS = os.getenv("CRL_SKIP_JOBS", "0") == "1"

# Fair-comparison preset:
# - Match update cadence in env-steps across algorithms.
# - Keep similar batch scale per optimization step.
VARIBAD_ROLLOUT_STEPS = 32                 # varibad --batch_size (T)
VARIBAD_MINIBATCH_SIZE = 256
FB_TRAIN_FREQ_STEPS = NUM_ENVS * VARIBAD_ROLLOUT_STEPS
FB_TRAIN_BATCH_SIZE = NUM_ENVS * VARIBAD_ROLLOUT_STEPS
DO_EVAL = True
NUM_EVAL_EPISODES = 5


def _smoke_test_cw_vector_envs() -> None:
    seq = get_task_sequence("continualworld", "cw10")
    task = seq[0]

    for mode in ("sync", "async"):
        envs = make_vec_env(
            env_id="continualworld",
            task=task,
            seed=0,
            max_episode_steps=MAX_EPISODE_STEPS,
            num_envs=4,
            mode=mode,
        )
        obs, info = envs.reset(seed=0)
        assert getattr(obs, "shape", (0,))[0] == 4, f"{mode}: expected batch dim 4, got {getattr(obs, 'shape', None)}"

        actions = np.stack([envs.single_action_space.sample() for _ in range(4)], axis=0)
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        assert getattr(next_obs, "shape", (0,))[0] == 4, f"{mode}: invalid next_obs shape {getattr(next_obs, 'shape', None)}"
        assert np.asarray(rewards).shape[0] == 4, f"{mode}: invalid rewards shape {np.asarray(rewards).shape}"
        assert np.asarray(terminated).shape[0] == 4, f"{mode}: invalid terminated shape {np.asarray(terminated).shape}"
        assert np.asarray(truncated).shape[0] == 4, f"{mode}: invalid truncated shape {np.asarray(truncated).shape}"

        envs.close()
        print(f"[vec-smoke] continualworld {mode} ok | task={task}")


if __name__ == "__main__":
    if RUN_VEC_SMOKE_TEST:
        _smoke_test_cw_vector_envs()

    if SKIP_JOBS:
        raise SystemExit(0)

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
