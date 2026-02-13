
# Plot a single metric
#python plot.py --group "minigrid-basic-s0" --metric \
#    metrics/cumulative_reward \
#    metrics/reward_per_episode \
#
#python plot.py --group "switchingdist-basic-s0" --metric \
#    metrics/cumulative_reward \
#    metrics/reward_per_episode \
#
#python plot.py --group "dmc/walker-full-s0" --metric \
#    metrics/cumulative_reward \
#    metrics/reward_per_episode \

# Plot 3 metrics in composite
#python3 scripts/plot.py --group mujoco/walker2d-full-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
python3 scripts/plot.py --group highway_parking-tasks_basic-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
python3 scripts/plot.py --group highway_parking-tasks_full-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
python3 scripts/plot.py --group highway_parking-dists_full-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
python3 scripts/plot.py --group highway_parking-dists_basic-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
#python plot.py --group minigrid-basic-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
#python plot.py --group switchingdist-basic-s0 --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
#python plot.py --group "dmc/walker-full-s0" --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
#python plot.py --group "highway_parking-full-s0" --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id
#python plot.py --group "minihack-goals-s0" --composite --pdf --metrics metrics/cumulative_reward metrics/reward_per_episode metrics/task_id

