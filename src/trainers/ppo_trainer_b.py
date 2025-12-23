"""
PPO Trainer for Variant B: Value/Q-Space Scalarization
Rewards are kept as vectors, scalarization happens in advantage space.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional

import gymnasium as gym
from src.trainers.morl_trainer import MORLTrainer


class PPOTrainerB(MORLTrainer):
    """
    PPO Trainer for Variant B: Value/Q-Space Scalarization.
    Rewards are vectors, advantages are scalarized before PPO update.
    """
    def __init__(
        self,
        agent,
        env,
        device,
        learning_rate=3e-4,
        num_steps=128,
        total_timesteps=80000,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
        clip_coef=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=64,
        num_objectives=2,
    ):
        super().__init__()
        self.agent = agent
        self.env = env
        self.device = device
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.num_objectives = num_objectives
        
        self.optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        self.num_updates = total_timesteps // num_steps
        
        # Determine if continuous action space (Box); Discrete has shape=() so we check type
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)
        
        # Initialize buffers
        self._init_buffers()

    def _init_buffers(self):
        """Initialize rollout buffers."""
        obs_dim = self.env.observation_space.shape[0]
        
        self.obs = torch.zeros((self.num_steps, obs_dim)).to(self.device)
        self.dones = torch.zeros((self.num_steps,)).to(self.device)
        
        if self.is_continuous:
            action_dim = self.env.action_space.shape[0]
            self.actions = torch.zeros((self.num_steps, action_dim)).to(self.device)
        else:
            self.actions = torch.zeros((self.num_steps,), dtype=torch.long).to(self.device)
            
        self.logprobs = torch.zeros((self.num_steps,)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)
        self.contexts = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)

    def train(self, train_loader=None):
        """Main training loop."""
        global_step = 0
        next_obs, _ = self.env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.tensor(0.0).to(self.device)

        print("Starting Training (Variant B: Q-Space Scalarization)...")

        for update in range(1, self.num_updates + 1):
            # Rollout
            for step in range(self.num_steps):
                global_step += 1
                self.obs[step] = next_obs
                self.dones[step] = next_done
                # Save context (weights)
                self.contexts[step] = next_obs[-self.num_objectives:]

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value

                self.actions[step] = action
                self.logprobs[step] = logprob

                # Step environment
                if self.is_continuous:
                    action_np = action.cpu().numpy()
                else:
                    action_np = action.item()
                    
                real_next_obs, _, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                # Get vector reward from env
                r_vec_raw = self.env.get_reward()
                if r_vec_raw is None:
                    r_vec_raw = [0.0] * self.num_objectives
                self.rewards[step] = torch.as_tensor(r_vec_raw, dtype=torch.float32, device=self.device)

                next_obs = torch.Tensor(real_next_obs).to(self.device)
                next_done = torch.tensor(float(done)).to(self.device)

                if done:
                    next_obs_np, _ = self.env.reset()
                    next_obs = torch.Tensor(next_obs_np).to(self.device)

            # Compute Vector GAE
            advantages, returns = self._compute_gae(next_obs, next_done)

            # Update
            loss = self._update(advantages, returns)

            if update % 20 == 0:
                train_scalar_rewards = (self.rewards * self.contexts).sum(dim=1).mean().item()
                print(f"Update {update}/{self.num_updates}, Loss: {loss.item():.4f}, Mean Scalar Reward: {train_scalar_rewards:.2f}")

        print("Training Finished!")
        return self.agent

    def _compute_gae(self, next_obs, next_done):
        """Compute Vector Generalized Advantage Estimation."""
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs)
            next_value = next_value.reshape(1, -1) if len(next_value.shape) == 1 else next_value
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = torch.zeros(self.num_objectives).to(self.device)

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value[0]
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]

                # Vector TD Error
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]

                # Vector GAE
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.values

        return advantages, returns

    def _update(self, advantages, returns):
        """Perform PPO update with advantage scalarization."""
        # Prepare batch data
        b_obs = self.obs.reshape((-1, self.env.observation_space.shape[0]))
        b_logprobs = self.logprobs.reshape(-1)
        
        if self.is_continuous:
            b_actions = self.actions.reshape((-1, self.env.action_space.shape[0]))
        else:
            b_actions = self.actions.reshape(-1)
            
        b_returns = returns.reshape((-1, self.num_objectives))
        b_values = self.values.reshape((-1, self.num_objectives))
        b_contexts = self.contexts.reshape((-1, self.num_objectives))
        
        # Scalarize advantages: w1*A1 + w2*A2
        scalar_advantages = (advantages * self.contexts).sum(dim=1).reshape(-1)

        # Mini-batch update
        b_inds = np.arange(self.num_steps)
        
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.num_steps, self.batch_size):
                end = start + self.batch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Variant B: Scalarize advantages before PPO
                mb_adv = scalar_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss: Vector MSE
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return loss

    def evaluate(self, eval_loader=None, test_weights=None, num_episodes=1):
        """Evaluate the agent with different preference weights."""
        self.agent.eval()
        
        if test_weights is None:
            test_weights = [
                [1.0, 0.0],
                [0.7, 0.3],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.0, 1.0]
            ]

        results = []
        
        print(f"{'Weight (w1, w2)':<20} | {'Avg Position':<15} | {'Steps':<10}")
        print("-" * 50)

        for w in test_weights:
            obs, _ = self.env.reset(options={'w': w})
            obs = torch.Tensor(obs).to(self.device)

            positions = []
            steps = 0

            for _ in range(num_episodes):
                episode_positions = []
                episode_steps = 0
                
                while True:
                    with torch.no_grad():
                        action, _, _, _ = self.agent.get_action_and_value(obs)

                    step_result = self.env.step(action.item() if not self.is_continuous else action.cpu().numpy())
                    real_next_obs = step_result[0]
                    term = step_result[2]
                    trunc = step_result[3]

                    episode_positions.append(real_next_obs[0])
                    episode_steps += 1

                    obs = torch.Tensor(real_next_obs).to(self.device)
                    if term or trunc:
                        break
                
                positions.extend(episode_positions)
                steps += episode_steps

            avg_pos = np.mean(positions)
            print(f"{str(w):<20} | {avg_pos: .4f}          | {steps:<10}")
            results.append({
                'weight': [float(x) for x in w],
                'avg_position': float(avg_pos),
                'steps': int(steps),
            })

        self.agent.train()
        return results

