"""
PPO Trainer for Variant A: Reward Scalarization
Rewards are scalarized before training.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional

import gymnasium as gym
from src.trainers.morl_trainer import MORLTrainer
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir, get_figures_dir, get_saved_agents_dir
from src.utils.weights import sample_poisson_weights


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for neural network layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class VariantA_DiscreteAgent(nn.Module):
    """
    Variant A (Discrete): scalar critic + categorical policy.
    Architecture is intentionally simple and distinct from B/C.
    """

    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            return self.critic(x).squeeze(0)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        from torch.distributions.categorical import Categorical

        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True

        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        else:
            if action.dim() == 0:
                action = action.unsqueeze(0)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(x)

        if squeeze_out:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value


class VariantA_ContinuousAgent(nn.Module):
    """
    Variant A (Continuous): scalar critic + diagonal Gaussian policy.
    """

    def __init__(self, env):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        out = self.critic(x)
        return out.squeeze(0) if squeeze_out else out

    def get_action_and_value(self, x, action=None):
        from torch.distributions.normal import Normal

        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True

        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()
        else:
            if action.dim() == 1:
                action = action.unsqueeze(0)

        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(x)

        if squeeze_out:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value


class PPOTrainerA(MORLTrainer):
    """
    PPO Trainer for Variant A: Reward Scalarization.
    Rewards are combined into a single scalar before training.
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
        
        # Determine if continuous action space (Box); Discrete has shape=() so we check type
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)

        # Explicit default model definition (if agent is not provided)
        if agent is None:
            if self.is_continuous:
                agent = VariantA_ContinuousAgent(self.env)
            else:
                agent = VariantA_DiscreteAgent(self.env)
            agent = agent.to(self.device)

        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.num_updates = total_timesteps // num_steps
        
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
        self.rewards = torch.zeros((self.num_steps,)).to(self.device)  # scalarized reward
        self.values = torch.zeros((self.num_steps,)).to(self.device)
        # store preference/context to scalarize rewards
        self.contexts = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)

    def train(self, train_loader=None):
        """Main training loop."""
        logger = get_logger("morl.trainer.A", level=logging.INFO)
        global_step = 0
        next_obs, _ = self.env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.tensor(0.0).to(self.device)

        logger.info(
            "[bold]Starting Training[/bold] (Variant A: Reward Scalarization) | steps=%s | updates=%s",
            self.total_timesteps,
            self.num_updates,
        )

        for update in range(1, self.num_updates + 1):
            # Rollout
            for step in range(self.num_steps):
                global_step += 1
                self.obs[step] = next_obs
                self.dones[step] = next_done
                # save current preference weights from obs tail
                self.contexts[step] = next_obs[-self.num_objectives:]

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()

                self.actions[step] = action
                self.logprobs[step] = logprob

                # Step environment
                if self.is_continuous:
                    action_np = action.cpu().numpy()
                else:
                    action_np = action.item()
                    
                real_next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                done = terminated or truncated

                # Get vector reward from env and scalarize using current context
                r_vec_raw = self.env.get_reward()
                if r_vec_raw is None:
                    r_vec_raw = [0.0] * self.num_objectives
                r_vec = torch.as_tensor(r_vec_raw, dtype=torch.float32, device=self.device)
                scalar_reward = (r_vec * self.contexts[step]).sum()
                self.rewards[step] = scalar_reward

                next_obs = torch.Tensor(real_next_obs).to(self.device)
                next_done = torch.tensor(float(done)).to(self.device)

                if done:
                    next_obs_np, _ = self.env.reset()
                    next_obs = torch.Tensor(next_obs_np).to(self.device)

            # Compute GAE
            advantages, returns = self._compute_gae(next_obs, next_done)

            # Update
            loss = self._update(advantages, returns)

            if update % 10 == 0:
                mean_reward = self.rewards.mean().item()
                logger.info(
                    "Update %s/%s | loss=%.4f | mean_scalar_reward=%.2f",
                    update,
                    self.num_updates,
                    loss.item(),
                    mean_reward,
                )

        logger.info("[bold green]Training Finished![/bold green]")
        return self.agent

    def _compute_gae(self, next_obs, next_done):
        """Compute Generalized Advantage Estimation."""
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0

            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value[0]
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]

                # TD Error
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]

                # GAE
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + self.values

        return advantages, returns

    def _update(self, advantages, returns):
        """Perform PPO update."""
        # Prepare batch data
        b_obs = self.obs.reshape((-1, self.env.observation_space.shape[0]))
        b_logprobs = self.logprobs.reshape(-1)
        
        if self.is_continuous:
            b_actions = self.actions.reshape((-1, self.env.action_space.shape[0]))
        else:
            b_actions = self.actions.reshape(-1)
            
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

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

                # Variant A: Standard PPO with scalar advantage
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - self.ent_coef * entropy.mean() + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return loss

    def evaluate(self, eval_loader=None, test_weights=None, num_episodes=1):
        """Evaluate the agent with different preference weights."""
        logger = get_logger("morl.trainer.A", level=logging.INFO)
        self.agent.eval()
        
        if test_weights is None:
            test_weights = sample_poisson_weights(num_samples=10, dim=self.num_objectives, lam=1.0)

        results = []
        
        logger.info(
            "Evaluation started | episodes=%s | weights=%s | metric=mean(w·r) per step",
            num_episodes,
            len(test_weights),
        )

        for w in test_weights:
            obs, _ = self.env.reset(options={'w': w})
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

            # Align observation dimension with the agent's expected input size
            if hasattr(self.agent, "actor_mean"):
                expected_dim = self.agent.actor_mean[0].in_features
            elif hasattr(self.agent, "actor"):
                expected_dim = self.agent.actor[0].in_features
            else:
                expected_dim = obs.shape[0]

            positions = []
            steps = 0
            weighted_returns = []
            episode_curves = []

            for _ in range(num_episodes):
                episode_positions = []
                episode_steps = 0
                episode_weighted_sum = 0.0
                step_curve = []
                
                while True:
                    with torch.no_grad():
                        if obs.shape[0] != expected_dim:
                            if obs.shape[0] < expected_dim:
                                pad = torch.zeros(expected_dim - obs.shape[0], device=self.device, dtype=obs.dtype)
                                obs_in = torch.cat([obs, pad], dim=0)
                            else:
                                obs_in = obs[-expected_dim:]
                        else:
                            obs_in = obs

                        action, _, _, _ = self.agent.get_action_and_value(obs_in)

                    step_result = self.env.step(action.item() if not self.is_continuous else action.cpu().numpy())
                    real_next_obs = step_result[0]
                    term = step_result[2]
                    trunc = step_result[3]

                    episode_positions.append(real_next_obs[0])
                    episode_steps += 1

                    # Metric: weighted average reward using latest vector reward from env
                    r_vec_raw = self.env.get_reward()
                    if r_vec_raw is None:
                        r_vec_raw = [0.0] * self.num_objectives
                    r_vec = np.asarray(r_vec_raw, dtype=np.float32)
                    step_wdotr = float(np.dot(np.asarray(w, dtype=np.float32), r_vec))
                    episode_weighted_sum += step_wdotr
                    step_curve.append(step_wdotr)

                    obs = torch.Tensor(real_next_obs).to(self.device)
                    if term or trunc:
                        break
                
                positions.extend(episode_positions)
                steps += episode_steps
                weighted_returns.append(episode_weighted_sum / max(episode_steps, 1))
                episode_curves.append(step_curve)

            avg_pos = float(np.mean(positions)) if len(positions) else 0.0
            mean_weighted = float(np.mean(weighted_returns)) if len(weighted_returns) else 0.0

            # Build a single curve for this weight: average across episodes per timestep (ragged -> pad with NaN)
            max_len = max((len(c) for c in episode_curves), default=0)
            if max_len > 0:
                padded = np.full((len(episode_curves), max_len), np.nan, dtype=np.float32)
                for i, c in enumerate(episode_curves):
                    padded[i, : len(c)] = np.asarray(c, dtype=np.float32)
                w_curve = np.nanmean(padded, axis=0).astype(np.float32).tolist()
            else:
                w_curve = []
            logger.info("w=%s | mean_wdotr=%.4f | avg_pos=%.4f | steps=%s", w, mean_weighted, avg_pos, int(steps))
            # Cast to native Python types for JSON serialization
            results.append({
                'weight': [float(x) for x in w],
                'mean_weighted_reward': mean_weighted,
                'mean_wdotr_curve': w_curve,
                'avg_position': avg_pos,
                'steps': int(steps),
            })

        self.agent.train()
        return results


def _resolve_checkpoint_path(path_str: Optional[str]) -> Optional[str]:
    """
    Resolve save/load paths:
    - If user passes a bare filename, store under project_root/saved_agents/
    - If user passes a path with directories (or absolute path), respect it as-is
    - If no suffix is provided, default to ".pth"
    """
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() or p.parent != Path("."):
        return str(p)
    filename = p.name
    if Path(filename).suffix == "":
        filename = f"{filename}.pth"
    return str(ensure_dir(get_saved_agents_dir()) / filename)


def _resolve_plot_path(path_str: Optional[str], env_name: str) -> str:
    """
    Resolve plot output paths:
    - Default to project_root/figures/eval_curves_A_{env}.png
    - If user passes a bare filename, place it under figures/
    - If user passes a path with directories (or absolute path), respect it as-is
    """
    figures_dir = ensure_dir(get_figures_dir())
    if not path_str:
        return str(figures_dir / f"eval_curves_A_{env_name}.png")
    p = Path(path_str)
    if p.is_absolute() or p.parent != Path("."):
        return str(p)
    return str(figures_dir / p.name)


def main():
    parser = argparse.ArgumentParser(description="Run PPOTrainerA standalone (colored logs).")
    parser.add_argument("--env", type=str, default="Walker2d-v5", choices=["CartPole-v1", "Walker2d-v5", "Humanoid-v5"])
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint path before training.")
    parser.add_argument("--save", type=str, default=None, help="Save checkpoint path after training.")
    parser.add_argument("--eval_only", action="store_true", help="Load and run evaluation only (skip training).")
    parser.add_argument("--plot", action="store_true", help="Save evaluation curves plot as PNG.")
    parser.add_argument("--plot_path", type=str, default=None, help="Output PNG path for --plot.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("morl.trainer.A", level=logging.INFO)
    logger.info("Device: %s", device)

    from src.environments import ScalarRewardWrapper, SteerableHumanoidWrapper, SteerableWalkerWrapper

    base_env = gym.make(args.env)
    if args.env == "CartPole-v1":
        env = ScalarRewardWrapper(base_env)
    elif args.env == "Walker2d-v5":
        env = SteerableWalkerWrapper(base_env)
    else:
        env = SteerableHumanoidWrapper(base_env)

    # Default config matches existing experiment defaults
    is_cartpole = args.env == "CartPole-v1"
    num_objectives = 2 if is_cartpole else 3
    num_steps = 128 if is_cartpole else 2048
    update_epochs = 4 if is_cartpole else 10
    ent_coef = 0.001 if is_cartpole else 0.0
    # Standalone Walker2d script uses vf_coef=0.05
    vf_coef = 0.5 if is_cartpole else 0.05
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else (80000 if is_cartpole else 1000000)

    trainer = PPOTrainerA(
        agent=None,
        env=env,
        device=device,
        learning_rate=args.learning_rate,
        num_steps=num_steps,
        total_timesteps=total_timesteps,
        update_epochs=update_epochs,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        num_objectives=num_objectives,
    )

    load_path = _resolve_checkpoint_path(args.load)
    save_path = _resolve_checkpoint_path(args.save)
    if load_path:
        ckpt = torch.load(load_path, map_location=device)
        if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
            trainer.agent.load_state_dict(ckpt["agent_state_dict"])
        else:
            # allow loading raw state_dict for backward compatibility
            trainer.agent.load_state_dict(ckpt)
        logger.info("[bold cyan]Loaded checkpoint[/bold cyan]: %s", load_path)

    if not args.eval_only:
        trainer.train()
    results = trainer.evaluate(test_weights=None)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            curves = [r for r in results if isinstance(r, dict) and r.get("mean_wdotr_curve") and r.get("weight") is not None]
            if not curves:
                logger.warning("No curves found to plot.")
            else:
                plt.figure(figsize=(10, 5))
                for i, r in enumerate(curves):
                    y = r["mean_wdotr_curve"]
                    plt.plot(y, linewidth=1.5, label=f"w{i}")
                plt.title(f"Variant A eval: mean(w·r_t) per step ({args.env})")
                plt.xlabel("timestep")
                plt.ylabel("w·r_t")
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()

                out_path = _resolve_plot_path(args.plot_path, args.env)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                plt.savefig(out_path, dpi=200)
                plt.close()
                logger.info("[bold green]Saved plot[/bold green]: %s", out_path)
        except Exception as e:
            logger.error("Plot failed (matplotlib missing?): %s", e)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(
            {
                "agent_state_dict": trainer.agent.state_dict(),
                "env": args.env,
                "num_objectives": num_objectives,
            },
            save_path,
        )
        logger.info("[bold green]Saved checkpoint[/bold green]: %s", save_path)


if __name__ == "__main__":
    main()
