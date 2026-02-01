"""
Humanoid environment wrapper for Multi-Objective Reinforcement Learning.

Targets Gymnasium MuJoCo `Humanoid-v5` and exposes 3 objectives:
1) Velocity (forward reward)
2) Survival (healthy reward)
3) Energy/contact efficiency (negative control + contact costs)
"""

import gymnasium as gym
import numpy as np

from src.environments.base_env import BaseMORLEnv


class SteerableHumanoidWrapper(gym.Wrapper, BaseMORLEnv):
    """
    Humanoid wrapper with 3 objectives:
    1. Velocity (Forward Reward)
    2. Survival (Healthy Reward)
    3. Energy/Contact Efficiency (-(Control Cost + Contact Cost))
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        BaseMORLEnv.__init__(self)

        original_shape = env.observation_space.shape[0]

        # 3 objectives
        self.num_objectives = 3

        # Observation = State + Preference(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_shape + self.num_objectives,),
            dtype=np.float32,
        )

        # Initialize weights (w1, w2, w3)
        self.current_w = np.array([0.33, 0.33, 0.33], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if options and "w" in options:
            self.current_w = np.array(options["w"], dtype=np.float32)
        else:
            w = np.random.rand(self.num_objectives)
            self.current_w = (w / w.sum()).astype(np.float32)

        obs, info = self.env.reset(seed=seed, options=options)
        self._store_info(info)
        self._store_vector_reward(None)
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self.current_w]).astype(np.float32), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # Prefer reward components from info when available (varies by gym version).
        if (
            ("reward_forward" in info)
            or ("reward_survive" in info)
            or ("reward_ctrl" in info)
            or ("reward_contact" in info)
        ):
            r_velocity = float(info.get("reward_forward", 0.0))
            r_survive = float(info.get("reward_survive", 0.0))
            ctrl = float(info.get("reward_ctrl", 0.0))
            contact = float(info.get("reward_contact", 0.0))
            r_energy = -(ctrl + contact) * 10.0
        else:
            unwrapped = self.env.unwrapped

            x_velocity = float(info.get("x_velocity", 0.0))
            forward_weight = float(getattr(unwrapped, "_forward_reward_weight", 1.0))
            r_velocity = forward_weight * x_velocity

            r_survive = float(getattr(unwrapped, "healthy_reward", 0.0))

            # control_cost(action) exists on HumanoidEnv; fallback to 0.0 if missing
            if hasattr(unwrapped, "control_cost"):
                ctrl_cost = float(unwrapped.control_cost(action))
            else:
                ctrl_cost = 0.0

            contact_cost = 0.0
            if hasattr(unwrapped, "contact_cost"):
                # Gymnasium MuJoCo sometimes defines contact_cost(external_contact_forces)
                try:
                    contact_cost = float(unwrapped.contact_cost())
                except TypeError:
                    try:
                        data = getattr(unwrapped, "data", None)
                        forces = getattr(data, "cfrc_ext", None) if data is not None else None
                        if forces is not None:
                            contact_cost = float(unwrapped.contact_cost(forces))
                    except Exception:
                        contact_cost = 0.0

            r_energy = -(ctrl_cost + contact_cost) * 10.0

        vec_reward = np.array([r_velocity, r_survive, r_energy], dtype=np.float32)
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

