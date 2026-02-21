"""
Walker2d environment wrapper for Multi-Objective Reinforcement Learning.
Supports 3 objectives: velocity, survival, and energy efficiency.
"""
import gymnasium as gym
import numpy as np

from src.environments.base_env import BaseMORLEnv


class SteerableWalkerWrapper(gym.Wrapper, BaseMORLEnv):
    """
    Walker2d wrapper with 3 objectives:
    1. Velocity (Forward Reward)
    2. Survival (Healthy Reward)
    3. Energy Efficiency (Negative Control Cost)
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        BaseMORLEnv.__init__(self)
        original_shape = env.observation_space.shape[0]

        # 3 objectives
        self.num_objectives = 3

        # Observation = State(17) + Preference(3) = 20
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(original_shape + self.num_objectives,),
            dtype=np.float32
        )
        # Initialize weights (w1, w2, w3)
        self.current_w = np.array([0.33, 0.33, 0.33], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if options and 'w' in options:
            self.current_w = np.array(options['w'], dtype=np.float32)
        else:
            # Random sample 3 weights and normalize
            w = np.random.rand(self.num_objectives)
            self.current_w = w / w.sum()

        obs, info = self.env.reset(seed=seed, options=options)
        self._store_info(info)
        self._store_vector_reward(None)
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self.current_w]).astype(np.float32), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        """
        Gymnasium Walker2d-v4 info only provides x_position/x_velocity.
        Reward components must be reconstructed from the unwrapped env:
          forward_reward = forward_reward_weight * x_velocity
          healthy_reward = env.healthy_reward
          ctrl_cost      = env.control_cost(action)
        """
        # Match reward decomposition used in `variant_*` scripts:
        # - r_velocity = reward_forward
        # - r_survive  = reward_survive
        # - r_energy   = -reward_ctrl * 10.0
        # Some gym versions expose these directly in info; otherwise reconstruct.
        if ("reward_forward" in info) or ("reward_survive" in info) or ("reward_ctrl" in info):
            r_velocity = float(info.get("reward_forward", 0.0))
            r_survive = float(info.get("reward_survive", 0.0))
            r_energy = -float(info.get("reward_ctrl", 0.0))
        else:
            unwrapped = self.env.unwrapped

            x_velocity = float(info.get("x_velocity", 0.0))
            forward_weight = float(getattr(unwrapped, "_forward_reward_weight", 1.0))
            r_velocity = forward_weight * x_velocity

            r_survive = float(getattr(unwrapped, "healthy_reward", 0.0))

            # control_cost(action) exists on Walker2dEnv; fallback to 0.0 if missing
            if hasattr(unwrapped, "control_cost"):
                ctrl_cost = float(unwrapped.control_cost(action))
            else:
                ctrl_cost = 0.0
            r_energy = -ctrl_cost

        # Assemble into 3D vector
        vec_reward = np.array([r_velocity, r_survive, r_energy], dtype=np.float32)

        # Scalar log (sum) to satisfy gym API; trainer will use vector via get_reward
        scalar_log = float(vec_reward.sum())

        obs = np.asarray(obs, dtype=np.float32)
        new_obs = np.concatenate([obs, self.current_w]).astype(np.float32)

        self._store_info(info)
        self._store_vector_reward(vec_reward)
        return new_obs, scalar_log, terminated, truncated, info

    def get_reward(self):
        return self._last_vector_reward

    def get_reward_dimension(self) -> int:
        return self.num_objectives

