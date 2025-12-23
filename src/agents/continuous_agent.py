"""
Continuous action space agents for Walker2d.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for neural network layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ContinuousScalarAgent(nn.Module):
    """
    Continuous Scalar Critic Agent for Variant A (Reward Scalarization).
    """
    def __init__(self, env):
        super().__init__()
        # Observation = 17 (state) + 3 (weights) = 20
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]  # Walker2d = 6

        # Critic (Scalar Output)
        # Variant A: Critic only predicts a scalar Value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor (Continuous Mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Get scalar value estimate. Supports 1D or batched input."""
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        out = self.critic(x)
        return out.squeeze(0) if squeeze_out else out

    def get_action_and_value(self, x, action=None):
        """Get action, log_prob, entropy, and value. Supports 1D or batched input."""
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        else:
            if action.dim() == 1:
                action = action.unsqueeze(0)

        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)

        if squeeze_out:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value


class ContinuousVectorAgent(nn.Module):
    """
    Continuous Vector Critic Agent for Variant B & C (Value/Q-Space Scalarization).
    """
    def __init__(self, env, num_objectives=3):
        super().__init__()
        # Get dimensions
        # Observation = 17 (state) + 3 (weights) = 20
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]  # Walker2d = 6

        self.num_objectives = num_objectives

        # Critic (Vector Output)
        # Output dimension = 3 (Vel, Survive, Energy)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_objectives), std=1.0),
        )

        # Actor (Continuous Mean)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        # Learnable Log Std (independent parameter, not state-dependent)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """Get vector value estimate. Supports 1D or batched input."""
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        out = self.critic(x)
        return out.squeeze(0) if squeeze_out else out

    def get_action_and_value(self, x, action=None):
        """Get action, log_prob, entropy, and vector value. Supports 1D or batched input."""
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Use normal distribution
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()
        else:
            if action.dim() == 1:
                action = action.unsqueeze(0)

        # For continuous action space, log_prob needs to sum over all dimensions
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(x)

        if squeeze_out:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value

