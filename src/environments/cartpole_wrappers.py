"""
CartPole environment wrappers for Multi-Objective Reinforcement Learning.
"""
import gymnasium as gym
import numpy as np

from src.environments.base_env import BaseMORLEnv


class ScalarRewardWrapper(gym.Wrapper, BaseMORLEnv):
    """
    CartPole wrapper that augments observation with preference weights w.
    This wrapper exposes a **vector reward** via `get_reward()`.

    Note: Gymnasium `step()` must return a scalar reward; we return a scalar log
    (sum of objectives), while trainers should use `env.get_reward()` for MORL.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        BaseMORLEnv.__init__(self)
        original_shape = env.observation_space.shape[0]
        
        # Add 2 dimensions for w (w1, w2)
        # 4 + 2 = 6
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(original_shape + 2,),
            dtype=np.float32
        )
        # Initialize w
        self.current_w = np.array([0.5, 0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if options and 'w' in options:
            self.current_w = np.array(options['w'], dtype=np.float32)
        else:
            w = np.random.rand(2)
            self.current_w = w / w.sum()

        obs, info = self.env.reset(seed=seed)
        self._store_info(info)
        self._store_vector_reward(None)
        return np.concatenate([obs, self.current_w]), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x, x_dot, theta, theta_dot = obs

        # x range [-2.4, 2.4]
        norm_left = (2.4 - x) / 4.8
        norm_right = (x + 2.4) / 4.8

        r1 = norm_left ** 2
        r2 = norm_right ** 2

        if terminated:
            r1 = 0.0
            r2 = 0.0

        # Reward as vector
        vec_reward = np.array([r1, r2], dtype=np.float32)

        new_obs = np.concatenate([obs, self.current_w])

        self._store_info(info)
        self._store_vector_reward(vec_reward)
        # Return a scalar log (sum) to satisfy gym API, but trainer will use vector via get_reward
        scalar_log = float(vec_reward.sum())
        return new_obs, scalar_log, terminated, truncated, info

    def get_reward(self):
        # Return stored vector reward
        return self._last_vector_reward

    def get_reward_dimension(self) -> int:
        return 2


class SteerableCartPoleWrapper(gym.Wrapper, BaseMORLEnv):
    """
    Variant B & C: Value/Q-Space Scalarization
    Returns vector rewards and combines them in value/advantage space.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        BaseMORLEnv.__init__(self)
        original_shape = env.observation_space.shape[0]
        # State space: 4+2=6 (State + w)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(original_shape + 2,),
            dtype=np.float32
        )
        self.current_w = np.array([0.5, 0.5], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if options and 'w' in options:
            self.current_w = np.array(options['w'], dtype=np.float32)
        else:
            w = np.random.rand(2)
            self.current_w = w / w.sum()

        obs, info = self.env.reset(seed=seed)
        # Combine State + w
        self._store_info(info)
        self._store_vector_reward(None)
        return np.concatenate([obs, self.current_w]), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x, x_dot, theta, theta_dot = obs

        norm_left = (2.4 - x) / 4.8
        norm_right = (x + 2.4) / 4.8

        r1 = norm_left ** 2
        r2 = norm_right ** 2

        if terminated:
            r1 = 0.0
            r2 = 0.0

        # Reward as a vector
        vec_reward = np.array([r1, r2], dtype=np.float32)

        # Gym requires step must return a scalar reward
        # but trainer will use vector via get_reward
        scalar_reward_log = float(vec_reward.sum())

        new_obs = np.concatenate([obs, self.current_w])

        self._store_info(info)
        self._store_vector_reward(vec_reward)
        return new_obs, scalar_reward_log, terminated, truncated, info

    def get_reward(self):
        # Return stored vector reward
        return self._last_vector_reward

    def get_reward_dimension(self) -> int:
        return 2

