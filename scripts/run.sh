#!/bin/bash

env_seed=0

# Run PPO on mujoco/humanoid for all tasks in HUMANOID_TASKS_SPEC (from crl.envs.tasks)
PIDS=()
tasks=(stand walk_forward run_forward crouch walk_backward run_backward turn_left turn_right jump)
for task in "${tasks[@]}"; do
    python3 crl/algos/ppo.py \
        --env.domain_name mujoco/humanoid \
        --env.task "${task}" \
        --env.seed ${env_seed} \
        --seed 0 \
        --epochs 2000 \
        --device cuda \
        --use_wandb &
    PIDS+=($!)
done

# Wait for all parallel jobs to finish
for pid in "${PIDS[@]}"; do
    wait $pid
done


# ==================================================
# Discrete
# ==================================================

#for algo_seed in {0..5}; do
#    uv run python3 crl/algos/varibad.py \
#        --env.domain_name minigrid \
#        --env.task_list basic \
#        --env.seed ${env_seed} \
#        --device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --use_wandb
#done

# PTDQN
#for algo_seed in {0..5}; do
#    uv run python3 crl/algos/ptdqn.py \
#        --env.domain_name minigrid \
#        --env.task_list basic \
#        --env.seed ${env_seed} \
#        --device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --eval_freq 500
#done

# OnlineFBcpr
#for algo_seed in {0..5}; do
#    uv run python3 crl/algos/fb_cpr_final.py \
#        --env.domain_name minigrid \
#        --env.task_list basic \
#        --env.seed ${env_seed} \
#        --device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --eval_freq 500
#done


#for algo_seed in {0..4}; do
#    uv run python3 crl/algos/varibad.py \
#        --env.domain_name dmc/walker \
#        --env.task_list full \
#        --env.seed ${env_seed} \
#        --device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --use_wandb
#done
#
# Run for seeds 0..4 in parallel
#PIDS=()
#for algo_seed in {0..4}; do
#    python3 crl/algos/fb.py \
#        --env.domain_name metaworld/reach-v3 \
#        --env.task_list 10 \
#        --env.task_switch_prob 0.001 \
#        --env.seed ${env_seed} \
#        --model.device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --expl.epsilon 0.5 \
#        --expl.num_z_samples 5 \
#        --eval_freq 500 \
#        --num_inference_samples 2000 &
#    PIDS+=($!)
#done
#
## Wait for all parallel jobs to finish
#for pid in "${PIDS[@]}"; do
#    wait $pid
#done


#PIDS=()
#for algo_seed in {0..4}; do
#    python3 crl/algos/varibad.py \
#        --env.domain_name metaworld/reach-v3 \
#        --env.task_list 10 \
#        --env.seed ${env_seed} \
#        --device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --use_wandb
#    PIDS+=($!)
#done
#
## Wait for all parallel jobs to finish
#for pid in "${PIDS[@]}"; do
#    wait $pid
#done

# ==================================================
# Synthetic tasks (switching distribution)
# ==================================================


#for algo_seed in {0..4}; do
#    uv run python3 crl/algos/varibad.py \
#        --env.domain_name switchingdist \
#        --env.task_list basic \
#        --env.seed ${env_seed} \
#        --device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --env.max_episode_steps 20 \
#        --use_wandb
#done

#for algo_seed in {0..4}; do
#    uv run python3 crl/algos/fb_cpr_final.py \
#        --env.domain_name switchingdist \
#        --env.task_list basic \
#        --env.seed ${env_seed} \
#        --model.device cuda \
#        --seed ${algo_seed} \
#        --num_episodes 10000 \
#        --expl.epsilon 0.5 \
#        --expl.num_z_samples 5 \
#        --eval_freq 500 \
#        --no-do_eval \
#        --num_inference_samples 2000 \
#        --env.max_episode_steps 20 \
#        --use_wandb
#done