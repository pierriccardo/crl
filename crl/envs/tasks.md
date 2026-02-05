
# Parking Environment

## state space
In parking env an observation $s$ (and desired goal $s_g$) are vectors of this form:

$$s = [x, y, vx, vy, \cos h, \sin h]$$

where:
- $x, y$ position of the agent
- $vx, vy$ velocities
- $\cos h, \sin h$ heading

check PARKING_OBS dict [here](https://highway-env.farama.org/_modules/highway_env/envs/parking_env/). The goal, **is sampled randomly at each reset** in the function `_create_vehicles` on the ParkingEnv class.

## reward
Reward computation is splitted on two methods in [ParkingEnv](https://highway-env.farama.org/_modules/highway_env/envs/parking_env/): `compute_reward` and `_reward`. The `compute_reward` method in [ParkingEnv](https://highway-env.farama.org/_modules/highway_env/envs/parking_env/) computes the reward using a weighted p-norm of the difference between the current state $s$ and a desired goal state $s_g$:

$$ R(s,a) = -| s - s_g |_{W,p}^p $$

Where the p-norm is defined as:

$$|x|_{W,p} = (\sum_i |W_i x_i|^p)^{1/p}$$

is a weighted p-norm. The weights $W_i$ are provided by `self.config["reward_weights"]` and $p$ is a parameter that can be set in `compute_reward` method (default is 0.5) to influence the [kurtosis](https://en.wikipedia.org/wiki/Kurtosis) of rewards.

The `_reward` method return the overall reward, is the sum of this proximity reward and a penalty for collisions:

```python

def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict,
        p: float = 0.5,
    ) -> float:
        return -np.power(
            np.dot(
                np.abs(achieved_goal - desired_goal),
                np.array(self.config["reward_weights"]),
            ),
            p,
        )
reward = sum(
    self.compute_reward(
        agent_obs["achieved_goal"], agent_obs["desired_goal"], {}
    )
    for agent_obs in obs
)
reward += self.config["collision_reward"] * sum(
    v.crashed for v in self.controlled_vehicles
)
```
The overall reward form `_reward` method should be negative, and
a successful parking is defined when the compute_reward is greater than `-self.config["success_goal_reward"]`, a trhreshold to define when a parking is successful.

## modify the reward
Reward can be modifyied by passing a dictionary to the environment's constructor. The ParkingEnv's default configuration, which can be overridden, includes several parameters relevant to the reward function:

```python
config = {
    "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02], # used in p-norm
    "success_goal_reward": 0.12,
    "collision_reward": -5
}

env = gym.make("parking-v0", config)
```

## Config
```python

env = ParkingEnv(config={...})
config = {
    'observation': {
        'type': 'KinematicsGoal',
        'features': ['x', 'y', 'vx', 'vy', 'cos_h', 'sin_h'],
        'scales': [100, 100, 5, 5, 1, 1],
        'normalize': False
    },
    'action': {
        'type': 'ContinuousAction'
    },
    'simulation_frequency': 15,
    'policy_frequency': 5,
    'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
    'screen_width': 600,
    'screen_height': 300,
    'centering_position': [0.5, 0.5],
    'scaling': 7,
    'show_trajectories': False,
    'render_agent': True,
    'offscreen_rendering': False,
    'manual_control': False,
    'real_time_rendering': False,
    'reward_weights': [1, 0.3, 0, 0, 0.02, 0.02],
    'success_goal_reward': 0.12,
    'collision_reward': -5,
    'steering_range': np.float64(0.7853981633974483),
    'duration': 100,
    'controlled_vehicles': 1,
    'vehicles_count': 0,
    'add_walls': True
}
```



# Tasks for the parking environment

parking:
  - name: parking-v0
    description: "Park the car in the parking space"
    config:
      reward_weights: [(1.3, 1.6), (1.1, 1.4), (0.0, 0.01), (0.0, 0.01), (0.0, 0.01), (0.0, 0.01)]
      collision_reward: (-3.5, -2.0)
      success_goal_reward: (0.03, 0.06)


