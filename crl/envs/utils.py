from dataclasses import dataclass

@dataclass
class EnvConfig:
    domain_name: str = "dmc/walker"
    task: str = "walk" # default task
    task_list: str = "full"
    seed: int = 0
    max_episode_steps: int = 1000
    task_switch_prob: float = .01  # Probability of switching task at each episode reset (1.0 = always switch)