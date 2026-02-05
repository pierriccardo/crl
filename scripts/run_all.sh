

# ==================================================
# Highway Parking
# ==================================================

python3 scripts/run.py \
  --env_name highway_parking \
  --task_list tasks_basic \
  --algos fb_cpr varibad \
  --seeds 0 1 2 3 4 \
  --task_switch_prob 0.01

python3 scripts/run.py \
  --env_name highway_parking \
  --task_list tasks_full \
  --algos fb_cpr varibad \
  --seeds 0 1 2 3 4 \
  --task_switch_prob 0.01

#python3 scripts/run.py \
#  --env_name dmc/walker \
#  --task_list full \
#  --algos fb_cpr \
#  --seeds 0 1 2 3 4 \
#  --task_switch_prob 0.01


#python3 scripts/run.py \
#  --env_name mujoco/walker2d \
#  --task_list full \
#  --algos fb_cpr varibad \
#  --seeds 0 1 2 3 4 \
#  --task_switch_prob 0.01


#python3 crl/algos/varibad.py \
#    --env.domain_name ogbench/cube-double-play-v0 \
#    --env.task_list default \
#    --env.task_switch_prob 0.08 \
#    --env.seed 0 \
#    --device cpu \
#    --seed 0 \
#    --num_episodes 10000


#
#python3 crl/algos/fb_cpr.py \
#  --env.domain_name highway_parking \
#  --env.task_list basic \
#  --env.max_episode_steps 300 \
#  --env.task_switch_prob 0.08 \
#  --env.seed 0 \
#  --model.device cuda \
#  --expl.epsilon 0.2 \
#  --seed 0 \
#  --num_episodes 10000 \
#  --buffer_size 10000 \
#  --warmup_episodes 50 \
#  --train.batch_size 512 \
#  --expl.epsilon 0.5 \
#  --expl.num_z_samples 5 \
#  --no-do_eval \
#  --eval_freq 500 \
#  --num_inference_samples 2000