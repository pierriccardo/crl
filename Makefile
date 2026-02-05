.PHONY: clean ppo_humanoid

clean:
	@echo "Cleaning experiment folders..."
	rm -rf wandb/
	rm -rf models/
	rm -rf results/
	@echo "All experiment folders cleaned!"

ppo_humanoid:
	python3 crl/algos/ppo.py \
		--env.domain_name mujoco/humanoid \
		--env.task crouch \
		--env.seed 0 \
		--seed 0
