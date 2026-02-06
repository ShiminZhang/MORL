"""
DM Control Walker environment wrapper for Multi-Objective Reinforcement Learning.
"""
import gymnasium as gym
import numpy as np
from dm_control import suite
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

from src.environments.base_env import BaseMORLEnv


class SteerableDMControlWalkerWrapper(gym.Wrapper, BaseMORLEnv):
    """
    DM Control Walker wrapper with 2 objectives:
    1. Velocity (horizontal velocity - forward movement)
    2. Energy Efficiency (negative of control cost)
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        BaseMORLEnv.__init__(self)

        # Store reference to the underlying dm_control environment
        # shimmy wraps it as env.unwrapped._env
        self._dm_env = env.unwrapped._env

        # The observation space of the wrapped env is a dict.
        # We flatten it to a single vector.
        # Handle both scalar (shape=()) and vector observations
        original_shape = sum(
            np.prod(v.shape) if v.shape else 1
            for v in env.observation_space.values()
        )

        # 2 objectives
        self.num_objectives = 2

        # Observation = State + Preference(2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_shape + self.num_objectives,),
            dtype=np.float64,
        )

        # Initialize weights (w1, w2)
        self.current_w = np.array([0.5, 0.5], dtype=np.float32)

        # Store last action for energy computation
        self._last_action = None

    def _flatten_obs(self, obs):
        return np.concatenate([obs[k].flatten() for k in sorted(obs.keys())])

    def _compute_velocity_reward(self):
        """Compute velocity reward from dm_control physics."""
        # Get horizontal velocity from the walker
        # The walker task uses center of mass velocity
        physics = self._dm_env.physics
        # Horizontal velocity (x direction)
        velocity = physics.named.data.sensordata['torso_subtreelinvel'][0]
        # Normalize to [0, 1] range (typical walking speed is 0-1 m/s)
        return float(np.clip(velocity / 1.0, 0.0, 1.0))

    def _compute_energy_reward(self, action):
        """Compute energy efficiency reward (negative control cost)."""
        if action is None:
            return 0.0
        # Control cost: sum of squared actions
        control_cost = float(np.sum(np.square(action)))
        # Convert to reward: less energy = higher reward
        # Normalize assuming action range is [-1, 1] and 6 actuators
        max_cost = len(action)  # sum of 1^2 for each actuator
        energy_reward = 1.0 - (control_cost / max_cost)
        return float(np.clip(energy_reward, 0.0, 1.0))

    def reset(self, seed=None, options=None):
        if options and "w" in options:
            self.current_w = np.array(options["w"], dtype=np.float32)
        else:
            w = np.random.rand(self.num_objectives)
            self.current_w = (w / w.sum()).astype(np.float32)

        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = None
        self._store_info(info)
        self._store_vector_reward(None)
        obs = self._flatten_obs(obs)
        return np.concatenate([obs, self.current_w]).astype(np.float64), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._last_action = action

        # Compute multi-objective rewards from physics state
        r_velocity = self._compute_velocity_reward()
        r_energy = self._compute_energy_reward(action)

        vec_reward = np.array([r_velocity, r_energy], dtype=np.float32)
        scalar_log = float(vec_reward.sum())

        obs = self._flatten_obs(obs)
        new_obs = np.concatenate([obs, self.current_w]).astype(np.float64)

        self._store_info(info)
        self._store_vector_reward(vec_reward)
        return new_obs, scalar_log, terminated, truncated, info

    def get_reward(self):
        return self._last_vector_reward

    def get_reward_dimension(self) -> int:
        return self.num_objectives


def make_dm_control_walker():
    """Helper function to create the wrapped dm_control walker environment."""
    dm_env = suite.load(domain_name="walker", task_name="walk")
    env = DmControlCompatibilityV0(dm_env)
    return SteerableDMControlWalkerWrapper(env)
