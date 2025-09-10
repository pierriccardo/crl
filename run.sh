uv run python3 crl/algos/dqn.py \
    --env_name goalenv \
    --task_sequence cardinal \
    --train_steps_per_task 5000 \
    --start_steps_per_task 500 \
    --eval_freq 100

uv run python3 crl/algos/ptdqn.py \
    --env_name minatar \
    --task_sequence classic \
    --train_steps_per_task 20000 \
    --start_steps_per_task 1000 \
    --eval_freq 1000