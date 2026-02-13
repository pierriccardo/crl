.PHONY: clean videos

clean:
	@echo "Cleaning experiment folders..."
	rm -rf wandb/
	rm -rf models/
	rm -rf results/
	rm -rf imgs/
	@echo "All experiment folders cleaned!"

videos:
	python scripts/record_humanoid_ppo_videos.py \
	--checkpoint models/mujoco-humanoid/walk_forward/ppo/seed_0 \
	--task walk_backward \
	--episodes 3

	python scripts/record_humanoid_ppo_videos.py \
	--checkpoint models/mujoco-humanoid/stand/ppo/seed_0 \
	--task crouch \
	--episodes 3