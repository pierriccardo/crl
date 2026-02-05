
# ==========================================================================================
# Synthetic Distributions for SimpleDistAsTaskEnv
# ==========================================================================================

SYNTHETIC_DISTS_SPECS = {
    # Slow drift distribution
    "dist:0": {
        "w_std": 0.2,
        "mu_drift_std": 0.01,
        "reset_mu": True,
    },
    # Fast drift distribution
    "dist:1": {
        "w_std": 0.2,
        "mu_drift_std": 0.10,
        "reset_mu": True,
    },
    # Higher variability in w (harder)
    "dist:2": {
        "w_std": 0.8,
        "mu_drift_std": 0.03,
        "reset_mu": True,
    },
}

# ==========================================================================================
# Parking-v0 reward/config distributions (highway-env)
# ==========================================================================================

PARKING_TASKS_SPECS = {

    # Config 1: Default-like, balanced focus
    # Emphasizes x, y position with minor heading importance.
    "task_1": {
        "description": "Default-like: Balanced position and heading.",
        "config": {
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -5,
        }
    },

    # Config 2: High precision on position (x, y)
    # Increases weights for x and y, making it crucial to be precise in placement.
    "task_2": {
        "description": "High Precision Position: Strong emphasis on exact x,y placement.",
        "config": {
            "reward_weights": [3, 3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.08, # Slightly harder success
            "collision_reward": -7,
        }
    },

    # Config 3: Strong heading alignment importance
    # Increases weights for cos_h and sin_h, encouraging accurate vehicle orientation.
    "task_3": {
        "description": "Heading Alignment: Focus on correct orientation.",
        "config": {
            "reward_weights": [1, 0.3, 0, 0, 0.5, 0.5],
            "success_goal_reward": 0.1,
            "collision_reward": -5,
        }
    },

    # Config 4: Prioritize speed (less penalty for vx, vy mismatch, but generally faster parking)
    # Lower `success_goal_reward` incentivizes reaching the goal quicker. Velocity weights are kept low.
    "task_4": {
        "description": "Fast Parking: Incentivizes reaching goal quickly due to low success threshold.",
        "config": {
            "reward_weights": [1, 0.3, 0.01, 0.01, 0.02, 0.02],  # Minor velocity penalty
            "success_goal_reward": 0.05,  # Very strict success, encourages fast parking
            "collision_reward": -8,       # Increased collision penalty for taking risks
        }
    },

    # Config 5: High collision penalty
    # Makes collisions extremely costly, promoting cautious behavior.
    "task_5": {
        "description": "Cautious Driver: Very high collision penalty.",
        "config": {
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -20,      # Very high penalty
        }
    },

    # Config 6: Very easy success condition
    # Agent can succeed even if slightly off target.
    "task_6": {
        "description": "Easy Success: For initial learning or forgiving tasks.",
        "config": {
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.2,   # Easier success
            "collision_reward": -5,
        }
    },

    # Config 7: Mix of high position and heading importance
    # Combines emphasis on both accurate position and orientation.
    "task_7": {
        "description": "Combined Precision: High importance for position AND heading.",
        "config": {
            "reward_weights": [2.5, 2.5, 0, 0, 0.4, 0.4],
            "success_goal_reward": 0.07,
            "collision_reward": -10,
        }
    },

    # Config 8: Minimal reward differentiation, only collision matters
    # Almost uniform reward for being close, only large deviations or collisions penalize.
    "task_8": {
        "description": "Collision Avoidance Only: Proximity reward is minimal, collision is key.",
        "config": {
            "reward_weights": [0.1, 0.1, 0, 0, 0.01, 0.01],
            "success_goal_reward": 0.15,  # Relatively easy success threshold
            "collision_reward": -15,      # Still a significant penalty
        }
    },

    # Config 9: Focus on smooth parking (penalize velocity mismatch at goal)
    # Increases weights for vx and vy, penalizing if the car isn't stopped at the goal.
    "task_9": {
        "description": "Smooth Parking: Penalizes non-zero velocity at goal.",
        "config": {
            "reward_weights": [1.5, 1.5, 1.0, 1.0, 0.05, 0.05],
            "success_goal_reward": 0.09,
            "collision_reward": -6,
        }
    },

    # Config 10: Aggressive behavior encouraged (low collision penalty, moderate success)
    # A smaller penalty for collisions might lead to more aggressive maneuvering.
    "task_10": {
        "description": "Aggressive Driver: Low collision penalty, encouraging bolder moves.",
        "config": {
            "reward_weights": [1, 0.3, 0, 0, 0.02, 0.02],
            "success_goal_reward": 0.12,
            "collision_reward": -2,       # Very low collision penalty
        }
    },
}


# Parking-v0 reward/config distributions (highway-env). Each task samples from (low, high) at reset.
# reward_weights: [x, y, vx, vy, cos_h, sin_h]. success_goal_reward = termination threshold.
# Intervals are chosen so tasks are very distant: no overlap between tasks on key parameters.
#
# Max reward per episode = 0 (reach goal, no collision). success_goal_reward is not added to return.
PARKING_DISTS_SPECS = {
    # Position-focused: x,y high; v,heading near zero; mild collision; loose success
    "park:0": {
        "reward_weights": [(1.3, 1.6), (1.1, 1.4), (0.0, 0.01), (0.0, 0.01), (0.0, 0.01), (0.0, 0.01)],
        "collision_reward": (-3.5, -2.0),
        "success_goal_reward": (0.03, 0.06),
    },
    # Heading-focused: x,y low; cos_h,sin_h high; harsh collision; tight success
    "park:1": {
        "reward_weights": [(0.08, 0.18), (0.08, 0.18), (0.0, 0.01), (0.0, 0.01), (0.28, 0.38), (0.28, 0.38)],
        "collision_reward": (-22.0, -16.0),
        "success_goal_reward": (0.24, 0.32),
    },
    # Velocity-focused: vx,vy high; position/heading mid; moderate collision/success
    "park:2": {
        "reward_weights": [(0.35, 0.50), (0.30, 0.45), (0.18, 0.28), (0.18, 0.28), (0.04, 0.08), (0.04, 0.08)],
        "collision_reward": (-11.0, -7.0),
        "success_goal_reward": (0.10, 0.14),
    },
    # "At any cost": position high, rest low; very harsh collision; very loose success
    "park:3": {
        "reward_weights": [(1.5, 1.9), (1.2, 1.6), (0.0, 0.02), (0.0, 0.02), (0.0, 0.02), (0.0, 0.02)],
        "collision_reward": (-35.0, -25.0),
        "success_goal_reward": (0.01, 0.04),
    },
    # Balanced: all components in mid range
    "park:4": {
        "reward_weights": [(0.55, 0.75), (0.50, 0.70), (0.06, 0.12), (0.06, 0.12), (0.10, 0.18), (0.10, 0.18)],
        "collision_reward": (-14.0, -10.0),
        "success_goal_reward": (0.14, 0.20),
    },
    # Position-only: x,y only; v,heading zero; mid collision; mid-loose success
    "park:5": {
        "reward_weights": [(1.0, 1.25), (1.0, 1.25), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "collision_reward": (-7.0, -5.0),
        "success_goal_reward": (0.06, 0.10),
    },
    # Precision parking: heading + position; mild collision; very tight success
    "park:6": {
        "reward_weights": [(0.65, 0.85), (0.55, 0.75), (0.02, 0.06), (0.02, 0.06), (0.20, 0.30), (0.20, 0.30)],
        "collision_reward": (-2.5, -1.0),
        "success_goal_reward": (0.30, 0.42),
    },
}

# ==========================================================================================
# Humanoid-v5 task specifications
# ==========================================================================================

HUMANOID_TASKS_SPECS = {
  "task_wrapper_spec": {
    "type": "humanoid_task_reward_v1",
    "signals": {
      "x_velocity": { "source": "com_velocity_xy", "index": 0 },
      "y_velocity": { "source": "com_velocity_xy", "index": 1 },
      "torso_height": { "source": "qpos", "index": 2 },
      "yaw_rate": { "source": "qvel", "index": 5 }
    },
    "notes": {
      "com_velocity_xy": "Compute from MuJoCo subtree COM position difference over dt, consistent with how forward reward uses dx/dt.",
      "dt": "Use env.unwrapped.dt (depends on frame_skip)."
    }
  },
  "tasks": [
    {
      "name": "stand",
      "env_kwargs_override": {
        "forward_reward_weight": 0.0,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "velocity_height_tracking",
        "target_x_velocity": 0.0,
        "target_y_velocity": 0.0,
        "sigma_v": 0.2,
        "target_height": 1.4,
        "sigma_h": 0.2,
        "weights": { "w_vel": 2.0, "w_height": 2.0 }
      },
      "success_metric": {
        "type": "time_in_zone",
        "min_fraction": 0.6,
        "zone": {
          "abs_x_vel_max": 0.3,
          "abs_y_vel_max": 0.3,
          "height_range": [1.2, 2.0]
        }
      }
    },
    {
      "name": "walk_forward",
      "env_kwargs_override": {
        "forward_reward_weight": 1.25,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "velocity_tracking",
        "target_x_velocity": 1.0,
        "target_y_velocity": 0.0,
        "sigma_v": 0.5,
        "weights": { "w_vel": 1.0 }
      },
      "success_metric": {
        "type": "mean_abs_error",
        "signal": "x_velocity",
        "threshold": 0.6
      }
    },
    {
      "name": "run_forward",
      "env_kwargs_override": {
        "forward_reward_weight": 1.25,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "velocity_tracking",
        "target_x_velocity": 3.0,
        "target_y_velocity": 0.0,
        "sigma_v": 0.8,
        "weights": { "w_vel": 1.0 }
      },
      "success_metric": {
        "type": "mean_abs_error",
        "signal": "x_velocity",
        "threshold": 1.0
      }
    },
    {
      "name": "walk_backward",
      "env_kwargs_override": {
        "forward_reward_weight": 0.0,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "velocity_tracking",
        "target_x_velocity": -1.0,
        "target_y_velocity": 0.0,
        "sigma_v": 0.5,
        "weights": { "w_vel": 2.0 }
      },
      "success_metric": {
        "type": "mean_abs_error",
        "signal": "x_velocity",
        "threshold": 0.6
      }
    },
    {
      "name": "strafe_left",
      "env_kwargs_override": {
        "forward_reward_weight": 0.0,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "velocity_tracking",
        "target_x_velocity": 0.0,
        "target_y_velocity": 1.0,
        "sigma_v": 0.5,
        "weights": { "w_vel": 2.0 }
      },
      "success_metric": {
        "type": "mean_abs_error",
        "signal": "y_velocity",
        "threshold": 0.6
      }
    },
    {
      "name": "turn_in_place_left",
      "env_kwargs_override": {
        "forward_reward_weight": 0.0,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "yaw_rate_tracking",
        "target_yaw_rate": 1.0,
        "sigma_yaw": 0.6,
        "weights": { "w_yaw": 2.0 }
      },
      "success_metric": {
        "type": "time_in_zone",
        "min_fraction": 0.5,
        "zone": {
          "abs_x_vel_max": 0.5,
          "abs_y_vel_max": 0.5,
          "yaw_rate_range": [0.5, 1.5]
        }
      }
    },
    {
      "name": "crouch",
      "env_kwargs_override": {
        "forward_reward_weight": 0.0,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0,
        "terminate_when_unhealthy": False,
        "healthy_z_range": [0.7, 2.0]
      },
      "objective": {
        "type": "height_tracking",
        "target_height": 0.9,
        "sigma_h": 0.15,
        "weights": { "w_height": 2.0 }
      },
      "success_metric": {
        "type": "time_in_zone",
        "min_fraction": 0.5,
        "zone": { "height_range": [0.8, 1.0] }
      }
    },
    {
      "name": "pose_match_standing_reference",
      "env_kwargs_override": {
        "forward_reward_weight": 0.0,
        "ctrl_cost_weight": 0.1,
        "contact_cost_weight": 5e-7,
        "healthy_reward": 5.0
      },
      "objective": {
        "type": "qpos_pose_distance",
        "pose_path": "poses/standing.npy",
        "mask": {
          "exclude_root_xy": True,
          "exclude_root_quat": False
        },
        "weights": {
          "mode": "groups_by_joint_name_substrings",
          "groups": {
            "torso": { "substrings": ["abdomen"], "weight": 1.0 },
            "legs": { "substrings": ["hip", "knee", "ankle"], "weight": 2.0 },
            "arms": { "substrings": ["shoulder", "elbow"], "weight": 0.2 }
          }
        },
        "distance": {
          "type": "weighted_l2",
          "success_threshold": 0.7
        }
      },
      "success_metric": {
        "type": "min_distance_below",
        "threshold": 0.7
      }
    }
  ]
}

# Lookup by task name for factory and CLI
HUMANOID_TASK_BY_NAME = {t["name"]: t for t in HUMANOID_TASKS_SPECS["tasks"]}
