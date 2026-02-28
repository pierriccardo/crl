# ==========================================================================================
# MJX (mujoco_playground) continual-task specifications
# ==========================================================================================
#
# Domains map to mujoco_playground env names:
#   mjx/cheetah   -> CheetahRun
#   mjx/humanoid  -> HumanoidRun / HumanoidWalk / HumanoidStand
#   mjx/walker    -> WalkerRun / WalkerWalk / WalkerStand   (replaces brax/ant)
#
# Physics keys (gravity, friction) are applied directly on mj_model before JIT.
# action_coefficient and action_mask are applied by BraxTaskTransformWrapper.
#

MJX_TASKS_SPECS = {
    "cheetah": {
        "normal":               {"env_name": "CheetahRun"},
        "moon":                 {"env_name": "CheetahRun",  "gravity": 0.15},
        "rainfall":             {"env_name": "CheetahRun",  "friction": 0.4},
        "hugegravity":          {"env_name": "CheetahRun",  "gravity": 1.5},
        "inverted_actions":     {"env_name": "CheetahRun",  "action_coefficient": -1.0},
        "moon_inverted":        {"env_name": "CheetahRun",  "gravity": 0.15, "action_coefficient": -1.0},
    },
    "humanoid": {
        "normal":               {"env_name": "HumanoidRun"},
        "stand":                {"env_name": "HumanoidStand"},
        "walk":                 {"env_name": "HumanoidWalk"},
        "moon":                 {"env_name": "HumanoidRun",  "gravity": 0.15},
        "rainfall":             {"env_name": "HumanoidRun",  "friction": 0.4},
        "hugegravity":          {"env_name": "HumanoidRun",  "gravity": 1.5},
        "inverted_actions":     {"env_name": "HumanoidRun",  "action_coefficient": -1.0},
    },
    "walker": {
        "run":                  {"env_name": "WalkerRun"},
        "walk":                 {"env_name": "WalkerWalk"},
        "stand":                {"env_name": "WalkerStand"},
        "moon":                 {"env_name": "WalkerRun",   "gravity": 0.15},
        "rainfall":             {"env_name": "WalkerRun",   "friction": 0.4},
        "hugegravity":          {"env_name": "WalkerRun",   "gravity": 1.5},
        "inverted_actions":     {"env_name": "WalkerRun",   "action_coefficient": -1.0},
        # leg-disable tasks (6 actuators: r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle)
        "noleg_right":          {"env_name": "WalkerRun",   "action_mask": [0, 0, 0, 1, 1, 1]},
        "noleg_left":           {"env_name": "WalkerRun",   "action_mask": [1, 1, 1, 0, 0, 0]},
        "noknees":              {"env_name": "WalkerRun",   "action_mask": [1, 0, 1, 1, 0, 1]},
        "noankles":             {"env_name": "WalkerRun",   "action_mask": [1, 1, 0, 1, 1, 0]},
    },
}

MJX_SCENARIO_SEQUENCES = {
    "mjx/cheetah": {
        "forgetting":       ["hugegravity", "moon",     "rainfall",      "normal"],
        "transfer":         ["moon",         "rainfall", "inverted_actions", "hugegravity"],
        "robustness":       ["normal", "inverted_actions", "normal", "inverted_actions"],
        "compositionality": ["moon", "inverted_actions", "moon_inverted", "normal"],
    },
    "mjx/humanoid": {
        "default":          ["normal", "moon", "walk", "stand"],
        "robustness":       ["normal", "inverted_actions", "normal", "inverted_actions"],
    },
    "mjx/walker": {
        "forgetting":       ["run",        "hugegravity", "rainfall",   "moon"],
        "transfer":         ["noleg_right", "noleg_left", "noknees",    "noankles"],
        "robustness":       ["run", "inverted_actions",   "run",        "inverted_actions"],
        "compositionality": ["noleg_right", "noleg_left", "moon",       "rainfall"],
    },
}

