.PHONY: clean

clean:
	@echo "Cleaning experiment folders..."
	rm -rf wandb/
	rm -rf models/
	rm -rf results/
	@echo "All experiment folders cleaned!"
