import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DMControlEnv(gym.Env):
    """Gymnasium wrapper for DeepMind Control Suite."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, domain="cartpole", task="swingup", render_mode=None,
                 from_pixels=False, height=84, width=84, camera_id=0, frame_skip=1):
        super().__init__()
        from dm_control import suite

        self._env = suite.load(domain_name=domain, task_name=task)
        self.render_mode = render_mode
        self.from_pixels = from_pixels
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.frame_skip = frame_skip

        # Action space
        action_spec = self._env.action_spec()
        if action_spec.shape:
            self.action_space = spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                shape=action_spec.shape,
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=action_spec.minimum,
                high=action_spec.maximum,
                shape=(1,),
                dtype=np.float32
            )

        # Observation space
        if from_pixels:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(height, width, 3),
                dtype=np.uint8
            )
        else:
            # Flatten state observations
            obs_spec = self._env.observation_spec()
            obs_dim = int(sum(np.prod(spec.shape) for spec in obs_spec.values()))
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )

    def _get_obs(self, time_step):
        """Extract observation from dm_control TimeStep."""
        if self.from_pixels:
            obs = self._env.physics.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            return obs
        else:
            # Flatten and concatenate all observation components
            obs_list = []
            for key in sorted(time_step.observation.keys()):
                obs_list.append(np.array(time_step.observation[key]).flatten())
            return np.concatenate(obs_list).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # dm_control uses random_state for seeding
            # Set both numpy and dm_control's random state
            np.random.seed(seed)
            # dm_control environments have their own random state
            # This is a best-effort seeding approach
            try:
                self._env.task.random.seed(seed)
            except AttributeError:
                # Some tasks may not have random attribute
                pass
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs, {}

    def step(self, action):
        # Ensure action is the right shape
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)

        reward = 0.0
        for _ in range(self.frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0.0
            if time_step.last():
                break

        obs = self._get_obs(time_step)
        terminated = time_step.last()
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.physics.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
        elif self.render_mode == "human":
            # For human rendering, you might want to use matplotlib or similar
            # This is a simple placeholder
            img = self._env.physics.render(
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            return img

    def close(self):
        self._env.close()


class ContinualWorldEnv(gym.Env):
    """Gymnasium wrapper for Continual World (MetaWorld) environments."""

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, task="hammer-v1", render_mode=None):
        super().__init__()
        try:
            from metaworld import ML1
        except ImportError:
            raise ImportError(
                "metaworld is required for ContinualWorldEnv. "
                "Install it with: pip install metaworld"
            )

        # Create ML1 environment with the specified task
        ml1 = ML1(task)
        self._env = ml1.train_classes[task]()
        self._task = ml1.train_tasks[0]  # Use first training task
        self._env.set_task(self._task)

        self.render_mode = render_mode
        self.task_name = task

        # Action space
        action_spec = self._env.action_space
        self.action_space = spaces.Box(
            low=action_spec.low,
            high=action_spec.high,
            shape=action_spec.shape,
            dtype=np.float32
        )

        # Observation space (state-based, not pixels)
        obs_spec = self._env.observation_space
        self.observation_space = spaces.Box(
            low=obs_spec.low,
            high=obs_spec.high,
            shape=obs_spec.shape,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # MetaWorld uses numpy random state
            np.random.seed(seed)
            try:
                self._env.random.seed(seed)
            except AttributeError:
                pass

        obs = self._env.reset()
        return obs.astype(np.float32), {}

    def step(self, action):
        # Ensure action is the right shape and type
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        else:
            action = action.astype(np.float32)

        # MetaWorld returns (obs, reward, done, info) not (obs, reward, terminated, truncated, info)
        obs, reward, done, info = self._env.step(action)
        # Convert to Gymnasium format: done -> terminated, no truncation
        terminated = bool(done)
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.render(offscreen=True, camera_name="corner")
        elif self.render_mode == "human":
            return self._env.render(offscreen=False, camera_name="corner")
        return None

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()
