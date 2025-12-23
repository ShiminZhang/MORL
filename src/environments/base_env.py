"""
Abstract base class for MORL environments.
Provides a minimal interface and stores the last step reward internally.
"""
from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseMORLEnv(ABC):
    """
    Minimal MORL environment contract.
    Implementations should store the latest reward (scalar or vector)
    so that `get_reward` and `get_reward_dimension` work without relying
    on info keys.
    """

    def __init__(self):
        # Keep last rewards for retrieval
        self._last_scalar_reward = None  # float
        self._last_vector_reward = None  # list/np.ndarray
        self._last_info = None

    @abstractmethod
    def reset(self, seed: int = None, options: dict = None) -> Tuple[Any, dict]:
        """Reset the environment."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Any, bool, bool, dict]:
        """Take a step in the environment."""
        raise NotImplementedError

    @abstractmethod
    def get_reward(self) -> Any:
        """
        Return the reward from the last step.
        If vector reward is available, return it; otherwise return scalar.
        """
        if self._last_vector_reward is not None:
            return self._last_vector_reward
        return self._last_scalar_reward

    @abstractmethod
    def get_reward_dimension(self) -> int:
        """Return the dimensionality of the reward (e.g., 1 for scalar, N for vector)."""
        raise NotImplementedError

    def _store_info(self, info: dict):
        """Store last info for retrieval."""
        self._last_info = info

    def _store_scalar_reward(self, reward: float | None):
        if reward is None:
            self._last_scalar_reward = None
        else:
            self._last_scalar_reward = float(reward)
        self._last_vector_reward = None

    def _store_vector_reward(self, reward_vec: Any):
        if reward_vec is None:
            self._last_vector_reward = None
        else:
            # store as list to keep JSON-serializable shape if needed
            self._last_vector_reward = (
                reward_vec.tolist() if hasattr(reward_vec, "tolist") else list(reward_vec)
            )
        self._last_scalar_reward = None


