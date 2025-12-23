"""
Discrete action space agents for CartPole.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for neural network layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Scalar Critic Agent for Variant A (Reward Scalarization).
    """
    def __init__(self, env):
        super().__init__()
        # Critic: estimate V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(6, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor: action prob distribution
        self.actor = nn.Sequential(
            layer_init(nn.Linear(6, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        """Get value estimate."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Get action, log_prob, entropy, and value."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class VectorAgent(nn.Module):
    """
    Vector Critic Agent for Variant B & C (Value/Q-Space Scalarization).
    """
    def __init__(self, env):
        super().__init__()
        # Critic
        # Input: 6 (State + w)
        # Output: 2 (V1 and V2)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(6, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2), std=1.0),  # 2D
        )
        # Actor: the same
        self.actor = nn.Sequential(
            layer_init(nn.Linear(6, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        """Get vector value estimate. Shape: [batch_size, 2]"""
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Get action, log_prob, entropy, and vector value."""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

