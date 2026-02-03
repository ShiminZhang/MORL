"""
Deep Sea Treasure environment wrapper for MORL.
"""
import gymnasium as gym
import numpy as np
import mo_gymnasium as mo_gym

from src.environments.base_env import BaseMORLEnv


class SteerableDeepSeaTreasureWrapper(gym.Wrapper, BaseMORLEnv):
    """
    Objectives (K=2): treasure value [0, 23.7], time penalty (-1 per step)
    Observation: position (2) + preferences (2) = (4,)
    Action: Discrete(4) - up, down, left, right
    """
    
    def __init__(self, env=None, normalize_rewards: bool = True):
        if env is None:
            env = mo_gym.make("deep-sea-treasure-v0")
        
        gym.Wrapper.__init__(self, env)
        BaseMORLEnv.__init__(self)
        
        self.normalize_rewards = normalize_rewards
        self.num_objectives = 2
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(4,),  # position (2) + preferences (2)
            dtype=np.float32
        )
        
        self.current_w = np.array([0.5, 0.5], dtype=np.float32)
        self.treasure_max = 23.7
    
    def reset(self, seed=None, options=None):
        if options and 'w' in options:
            self.current_w = np.array(options['w'], dtype=np.float32)
        else:
            w = np.random.rand(self.num_objectives)
            self.current_w = (w / w.sum()).astype(np.float32)
        
        obs, info = self.env.reset(seed=seed)
        obs = np.asarray(obs, dtype=np.float32) / 10.0
        
        self._store_info(info)
        self._store_vector_reward(None)
        
        return np.concatenate([obs, self.current_w]).astype(np.float32), info
    
    def step(self, action):
        obs, vector_reward, terminated, truncated, info = self.env.step(action)
        
        vec_reward = np.asarray(vector_reward, dtype=np.float32)
        
        if self.normalize_rewards:
            r_treasure = vec_reward[0] / self.treasure_max
            r_time = vec_reward[1] / 100.0
            vec_reward_normalized = np.array([r_treasure, r_time], dtype=np.float32)
        else:
            vec_reward_normalized = vec_reward
        
        obs = np.asarray(obs, dtype=np.float32) / 10.0
        augmented_obs = np.concatenate([obs, self.current_w]).astype(np.float32)
        
        self._store_info(info)
        self._store_vector_reward(vec_reward_normalized)
        
        scalar_reward = float(vec_reward_normalized.sum())
        return augmented_obs, scalar_reward, terminated, truncated, info
    
    def get_reward(self):
        return self._last_vector_reward
    
    def get_reward_dimension(self) -> int:
        return self.num_objectives
