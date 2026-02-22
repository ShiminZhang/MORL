"""
PPO Trainer for Variant C: Gradient-Space Combination
Rewards are kept as vectors, scalarization happens at gradient level.
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


class VariantC_DiscreteAgent(nn.Module):
    """
    Variant C (Discrete): vector critic with shared trunk + per-objective heads,
    actor uses FiLM modulation by preference weights (structurally distinct from A/B).
    """

    def __init__(self, env, num_objectives: int):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Match ContinuousVectorAgent-style 64-64 MLP (vector critic + standard actor)
        self.num_objectives = num_objectives

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_objectives), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        out = self.critic(x)
        return out.squeeze(0) if squeeze_out else out

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
        value = self.get_value(x)

        if squeeze_out:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value


class VariantC_ContinuousAgent(nn.Module):
    """
    Variant C (Continuous): vector critic (shared trunk + heads),
    actor uses FiLM modulation by preference weights.
    """

    def __init__(self, env, num_objectives: int):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Match ContinuousVectorAgent from variant_c_gradient:
        # critic: obs -> 64 -> 64 -> K
        # actor_mean: obs -> 64 -> 64 -> action_dim
        self.num_objectives = num_objectives

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_objectives), std=1.0),
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
        value = self.get_value(x)

        if squeeze_out:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            entropy = entropy.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, entropy, value


class GradientAttentionSynthesizer(nn.Module):
    """
    Learns to output alpha (objective mixing weights) based on:
      - query: average state + preference
      - key: singular values S and right singular vectors V of gradient matrix G
    """

    def __init__(self, state_dim: int, pref_dim: int, num_objectives: int, n_bins: int = 100):
        super().__init__()
        self.n_bins = n_bins
        self.num_objectives = num_objectives

        self.query_net = nn.Sequential(
            nn.Linear(state_dim + pref_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        key_dim = num_objectives + (num_objectives * num_objectives)  # S + vec(V)
        self.key_net = nn.Sequential(
            nn.Linear(key_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_bins),
        )

        self.codebook = nn.Parameter(self._init_codebook(n_bins, num_objectives), requires_grad=True)

    @staticmethod
    def _init_codebook(n_bins: int, num_objectives: int) -> torch.Tensor:
        cb = torch.rand(n_bins, num_objectives)
        return cb / cb.sum(dim=-1, keepdim=True)

    def forward(self, state: torch.Tensor, pref: torch.Tensor, S: torch.Tensor, V: torch.Tensor):
        # Ensure 1D inputs
        if state.dim() != 1:
            state = state.view(-1)
        if pref.dim() != 1:
            pref = pref.view(-1)
        if S.dim() != 1:
            S = S.view(-1)
        if V.dim() != 2:
            V = V.view(self.num_objectives, self.num_objectives)

        q = self.query_net(torch.cat([state, pref], dim=-1))  # [32]
        sv_feat = torch.cat([S, V.flatten()], dim=-1)  # [K + K*K]
        k = self.key_net(sv_feat)  # [32]

        latent = q * k
        logits = self.classifier(latent)  # [n_bins]

        probs = torch.softmax(logits / 0.8, dim=-1)  # [n_bins]
        alpha = torch.matmul(probs, self.codebook)  # [K]
        return alpha, logits


def _get_actor_params(agent) -> list[torch.nn.Parameter]:
    # Continuous agents use actor_mean (+ logstd); discrete uses actor
    params: list[torch.nn.Parameter] = []
    if hasattr(agent, "actor_mean"):
        params += list(agent.actor_mean.parameters())
        if hasattr(agent, "actor_logstd"):
            params += [agent.actor_logstd]
    elif hasattr(agent, "actor"):
        params += list(agent.actor.parameters())
    else:
        raise AttributeError("Agent has neither `actor_mean` nor `actor`.")
    return params


def _get_grad_vector(params: list[torch.nn.Parameter]) -> Optional[torch.Tensor]:
    grads = []
    for p in params:
        if p.grad is not None:
            grads.append(p.grad.view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
    return torch.cat(grads) if grads else None


def _set_grad_vector(params: list[torch.nn.Parameter], grad_vector: torch.Tensor) -> None:
    pointer = 0
    for p in params:
        numel = p.numel()
        p.grad = grad_vector[pointer : pointer + numel].view_as(p).clone()
        pointer += numel


class PPOTrainerC(MORLTrainer):
    """
    PPO Trainer for Variant C: Gradient-Space Combination.
    Rewards are vectors, loss is computed per objective and gradients are mixed.
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
        use_gradient_synthesizer: bool = False,
        synthesizer_bins: int = 100,
        synthesizer_learning_rate: float = 1e-4,
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
        self.use_gradient_synthesizer = use_gradient_synthesizer

        # Determine if continuous action space (Box); Discrete has shape=() so we check type
        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)

        # Explicit default model definition (if agent is not provided)
        if agent is None:
            if self.is_continuous:
                agent = VariantC_ContinuousAgent(self.env, num_objectives=self.num_objectives)
            else:
                agent = VariantC_DiscreteAgent(self.env, num_objectives=self.num_objectives)
            agent = agent.to(self.device)

        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.actor_params = _get_actor_params(self.agent)
        self.synthesizer: Optional[GradientAttentionSynthesizer] = None
        self.synth_optimizer: Optional[optim.Optimizer] = None
        if self.use_gradient_synthesizer:
            state_dim = self.env.observation_space.shape[0]
            pref_dim = self.num_objectives
            self.synthesizer = GradientAttentionSynthesizer(
                state_dim=state_dim,
                pref_dim=pref_dim,
                num_objectives=self.num_objectives,
                n_bins=synthesizer_bins,
            ).to(self.device)
            self.synth_optimizer = optim.Adam(self.synthesizer.parameters(), lr=synthesizer_learning_rate)
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
        self.rewards = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)
        self.contexts = torch.zeros((self.num_steps, self.num_objectives)).to(self.device)

    def train(self, train_loader=None):
        """Main training loop."""
        logger = get_logger("morl.trainer.C", level=logging.INFO)  # 获取带名字空间的 logger（用于打印训练进度/指标）
        global_step = 0  # 全局环境交互步数计数器（跨 update/step 累加）
        next_obs, _ = self.env.reset()  # 重置环境，拿到初始观测（Gymnasium 返回 obs, info）
        next_obs = torch.Tensor(next_obs).to(self.device)  # 把 numpy 观测转成 torch tensor 并搬到 device
        next_done = torch.tensor(0.0).to(self.device)  # 当前 episode 是否结束的标记（0.0/1.0），作为 rollout buffer 的 done

        logger.info(  # 打印训练开始信息（总步数、update 次数、是否启用 synthesizer）
            "[bold]Starting Training[/bold] (Variant C: Gradient Mixing) | steps=%s | updates=%s | synthesizer=%s",
            self.total_timesteps,
            self.num_updates,
            bool(self.use_gradient_synthesizer),
        )

        for update in range(1, self.num_updates + 1):  # 外层循环：每次 update 先采样一段 rollout，再做若干 epoch 的 PPO 更新
            # Rollout（采样 num_steps 步数据写入 buffer）
            for step in range(self.num_steps):  # 内层循环：与环境交互 num_steps 次，收集一段轨迹
                global_step += 1  # 全局交互步数 +1（用于统计/日志）
                self.obs[step] = next_obs  # 记录当前时刻的观测 obs_t
                self.dones[step] = next_done  # 记录当前时刻的 done_t（表示 obs_t 是否是 episode 结束后的状态）
                # Save context (weights)
                self.contexts[step] = next_obs[-self.num_objectives:]  # 从观测尾部取出偏好/权重向量 w_t（环境 wrapper 约定）

                with torch.no_grad():  # rollout 采样不需要梯度（省显存/提速）
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)  # 用当前策略采样动作 a_t，并得到 logπ(a_t|s_t) 与向量价值 V(s_t)
                    self.values[step] = value  # 保存向量 value 估计（K 维），用于计算 GAE/returns

                self.actions[step] = action  # 保存动作 a_t（连续为向量，离散为标量）
                self.logprobs[step] = logprob  # 保存旧策略下的 logprob（PPO ratio 的分母部分）

                # Step environment（把 torch action 转为 env 能吃的 numpy/scalar）
                if self.is_continuous:  # 连续动作空间：Box
                    action_np = action.cpu().numpy()  # tensor -> numpy（放到 CPU）
                else:  # 离散动作空间：Discrete
                    action_np = action.item()  # 0-dim tensor -> Python int

                real_next_obs, _, terminated, truncated, info = self.env.step(action_np)  # 与环境交互一步，拿到下一个观测等信息（奖励这里用 env.get_reward 取向量）
                done = terminated or truncated  # Gymnasium 里 episode 结束由 terminated 或 truncated 任一为 True 决定

                # Get vector reward from env（Variant C：奖励保持为向量）
                r_vec_raw = self.env.get_reward()  # 从 wrapper 获取多目标向量奖励 r_t（长度 K）
                if r_vec_raw is None:  # 兜底：环境没返回向量奖励时，用 0 向量避免崩溃
                    r_vec_raw = [0.0] * self.num_objectives  # 构造长度 K 的零奖励
                    logger.error("No vector reward returned from environment")
                self.rewards[step] = torch.as_tensor(r_vec_raw, dtype=torch.float32, device=self.device)  # 保存向量奖励到 buffer

                next_obs = torch.Tensor(real_next_obs).to(self.device)  # 更新 next_obs 为 obs_{t+1}（tensor 化并搬到 device）
                next_done = torch.tensor(float(done)).to(self.device)  # 更新 next_done 为 done_{t+1}（0.0/1.0）

                if done:  # 若 episode 结束，立刻 reset，为后续 step 继续采样准备新的初始状态
                    next_obs_np, _ = self.env.reset()  # 重置环境，得到新的初始观测
                    next_obs = torch.Tensor(next_obs_np).to(self.device)  # 新初始观测转成 tensor

            # Compute Vector GAE（向量形式的 GAE：每个 objective 各算一份 advantage）
            advantages, returns = self._compute_gae(next_obs, next_done)  # 用 rollout buffer + bootstrap value 计算 advantages 和 returns

            # Update（用 rollout 数据进行 PPO 更新；Variant C 这里可能做“梯度空间混合”）
            loss = self._update(advantages, returns)  # 执行若干 epoch 的 mini-batch 更新，返回用于日志的 loss

            if update % 20 == 0:  # 每 20 次 update 打印一次训练指标
                train_scalar_rewards = (self.rewards * self.contexts).sum(dim=1).mean().item()  # 用 w_t·r_t 得到标量奖励并取平均（用于快速观察训练趋势）
                logger.info(  # 输出当前 update 的 loss 和均值标量奖励
                    "Update %s/%s | loss=%.4f | mean_scalar_reward=%.2f",
                    update,
                    self.num_updates,
                    float(loss.item()) if hasattr(loss, "item") else float(loss),  # 兼容 loss 可能是 tensor 或 float
                    train_scalar_rewards,
                )

        logger.info("[bold green]Training Finished![/bold green]")  # 训练结束日志
        return self.agent  # 返回训练后的 agent（策略/价值网络）

    def _compute_gae(self, next_obs, next_done):
        """Compute Vector Generalized Advantage Estimation."""
        with torch.no_grad():  # GAE 计算不需要梯度（纯后处理）
            next_value = self.agent.get_value(next_obs)  # 用 critic 估计最后一个 next_obs 的向量价值（用于 bootstrap）
            next_value = next_value.reshape(1, -1) if len(next_value.shape) == 1 else next_value  # 统一形状为 [1, K]（便于索引）
            advantages = torch.zeros_like(self.rewards).to(self.device)  # 初始化 advantages buffer，形状 [T, K]
            lastgaelam = torch.zeros(self.num_objectives).to(self.device)  # 反向递推用的上一时刻 GAE 累积项（K 维）

            for t in reversed(range(self.num_steps)):  # 反向遍历时间步 t=T-1...0（GAE 的标准实现）
                if t == self.num_steps - 1:  # 最后一个时间步：bootstrap 来自外部传入的 next_obs/next_done
                    nextnonterminal = 1.0 - next_done  # 若 next_done=1 表示终止，则 nextnonterminal=0（不 bootstrap）
                    nextvalues = next_value[0]  # 取出形状 [K] 的 next value
                else:  # 非最后时间步：bootstrap 来自 buffer 中 t+1 的 value/done
                    nextnonterminal = 1.0 - self.dones[t + 1]  # done_{t+1} 决定是否 bootstrap
                    nextvalues = self.values[t + 1]  # V(s_{t+1})（向量）

                # Vector TD Error（每个 objective 都算一份 TD 残差）
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]  # δ_t = r_t + γ V_{t+1} - V_t（向量）

                # Vector GAE（按 objective 逐维递推）
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam  # A_t = δ_t + γλ A_{t+1}（向量）

            returns = advantages + self.values  # 计算 returns：R_t = A_t + V_t（向量）

        return advantages, returns  # 返回向量 advantages 和向量 returns

    def _update(self, advantages, returns):
        """Perform PPO update with gradient mixing."""
        # Prepare batch data（把 [T, ...] rollout buffer 展平成 batch 维度，便于随机取 mini-batch）
        b_obs = self.obs.reshape((-1, self.env.observation_space.shape[0]))  # 观测 batch：形状 [T, obs_dim]
        b_logprobs = self.logprobs.reshape(-1)  # 旧策略 logprob：形状 [T]

        if self.is_continuous:  # 连续动作：动作是向量
            b_actions = self.actions.reshape((-1, self.env.action_space.shape[0]))  # 动作 batch：形状 [T, act_dim]
        else:  # 离散动作：动作是标量
            b_actions = self.actions.reshape(-1)  # 动作 batch：形状 [T]

        b_returns = returns.reshape((-1, self.num_objectives))  # 向量 returns：形状 [T, K]
        b_values = self.values.reshape((-1, self.num_objectives))  # 向量 values：形状 [T, K]（这里可能用于诊断/扩展）
        b_vector_advantages = advantages.reshape((-1, self.num_objectives))  # 向量 advantages：形状 [T, K]
        b_contexts = self.contexts.reshape((-1, self.num_objectives))  # 偏好/权重 w：形状 [T, K]

        # Mini-batch update（按 PPO 的方式做多轮 epoch，打乱索引分 batch 更新）
        b_inds = np.arange(self.num_steps)  # 构造时间步索引 [0..T-1]（这里 T=num_steps）

        for epoch in range(self.update_epochs):  # PPO 的多轮更新（对同一批 rollout 反复优化）
            np.random.shuffle(b_inds)  # 每个 epoch 随机打乱样本顺序（提升优化稳定性）
            for start in range(0, self.num_steps, self.batch_size):  # 以 batch_size 为步长遍历
                end = start + self.batch_size  # 计算当前 mini-batch 的结束位置
                mb_inds = b_inds[start:end]  # 取出当前 mini-batch 的样本索引

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(  # 用当前策略重新评估 mini-batch（得到新 logprob、熵、以及新 value）
                    b_obs[mb_inds], b_actions[mb_inds]  # 输入 s 与固定的 a（PPO 需要对同一动作算新旧概率比）
                )
                logratio = newlogprob - b_logprobs[mb_inds]  # 计算 log π_new(a|s) - log π_old(a|s)
                ratio = logratio.exp()  # 得到概率比 r_t(θ)=π_new/π_old（PPO clip 用）

                if self.use_gradient_synthesizer:  # 分支：启用“梯度合成器”，在梯度空间做目标混合
                    assert self.synthesizer is not None and self.synth_optimizer is not None  # 确保 synthesizer/optimizer 已初始化

                    # Compute per-objective gradient vectors for the actor（对每个 objective 单独构造 PPO policy loss，并提取 actor 的梯度向量）
                    grad_vectors: dict[int, torch.Tensor] = {}  # 用于保存每个 objective 的 actor 梯度 g_k（展平向量）
                    for k in range(self.num_objectives):  # 遍历每个 objective k
                        adv_k = b_vector_advantages[mb_inds, k]  # 取出第 k 维 advantage：形状 [B]
                        adv_k = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)  # 对 advantage 做标准化（常见技巧，稳定训练）

                        loss_k_element = torch.max(  # PPO clipped surrogate（逐样本）
                            -adv_k * ratio,  # unclipped surrogate：-A * r
                            -adv_k * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef),  # clipped surrogate：-A * clip(r)
                        )

                        self.optimizer.zero_grad()  # 清空上一轮累积的梯度（准备对当前 k 计算梯度）
                        loss_k = loss_k_element.mean()
                        loss_k.backward(retain_graph=True)  # 反传得到梯度；retain_graph=True 因为还要对其他 k 以及 value loss 继续用同一图
                        grad_vectors[k] = _get_grad_vector(self.actor_params)  # 把 actor 参数的梯度展平成向量 g_k（长度 D）

                    # Entropy gradient for actor（熵正则的梯度：鼓励探索）
                    self.optimizer.zero_grad()  # 清空梯度，单独对 entropy loss 求梯度
                    entropy_loss = -self.ent_coef * entropy.mean()  # 熵项：最大化 entropy 等价于最小化 -ent；乘以系数 ent_coef
                    entropy_loss.backward(retain_graph=True)  # 反传得到熵项对应的 actor 梯度
                    g_ent = _get_grad_vector(self.actor_params)  # 提取熵项的 actor 梯度向量 g_ent（长度 D）

                    # Assemble gradient matrix G: [D, K]（把各 objective 的梯度拼成矩阵）
                    G = torch.stack([grad_vectors[k] for k in range(self.num_objectives)], dim=1)  # G 的列是 g_k，形状 [D, K]
                    # Column-wise L2 normalization（与 notebook vc_svd 对齐：只在分解时看方向，不让模长主导）
                    G_norm = G / (torch.norm(G, p=2, dim=0, keepdim=True) + 1e-8)

                    # SVD to get U, S, V (all detached from autograd)（对归一化后的梯度矩阵做 SVD，得到可解释的梯度子空间基）
                    U, S, Vh = torch.linalg.svd(G_norm, full_matrices=False)  # G_norm=U diag(S) Vh（这里 Vh 是转置形式）
                    V = Vh  # torch.linalg.svd 返回的是 Vh；这里直接当作 V（后续按该实现使用）

                    # Sign alignment (stabilize U/V, match vc_svd notebook)（用 U 与 G_norm 的点积符号对齐）
                    dot_products = torch.sum(U * G_norm, dim=0)
                    signs = torch.sign(dot_products)
                    signs = torch.where(signs == 0, torch.ones_like(signs), signs)  # 避免 0 符号
                    U = U * signs
                    V = V * signs.unsqueeze(1)

                    mb_obs_avg = b_obs[mb_inds].mean(dim=0)  # 计算当前 mini-batch 的平均 state（作为 synthesizer 的输入之一）
                    mb_pref_avg = b_contexts[mb_inds].mean(dim=0)  # 计算当前 mini-batch 的平均偏好 w（作为 synthesizer 的输入之一）

                    alpha, _ = self.synthesizer(mb_obs_avg, mb_pref_avg, S, V)  # synthesizer 输出 alpha（对梯度子空间/目标的混合权重）
                    g_task = torch.matmul(U, alpha)  # 把 alpha 映射回参数空间得到最终任务梯度 g_task（形状 [D]）
                    g_final_actor = g_task + (g_ent if g_ent is not None else 0.0)  # 把熵梯度加到最终 actor 梯度里（若存在）

                    # Critic update via value loss; then inject actor gradient（先用 value loss 反传出 critic 梯度，再把 actor 梯度强行替换为 g_final_actor）
                    self.optimizer.zero_grad()  # 清空梯度，准备计算 value loss 的梯度（critic 需要正常反传）
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()  # 向量 value 的 MSE（对 K 维一起求均值）
                    (self.vf_coef * v_loss).backward()  # 只通过 value loss 反传（得到 critic 梯度，同时也会对 actor 产生梯度但后面会覆盖掉）

                    _set_grad_vector(self.actor_params, g_final_actor)  # 关键：把 actor 参数的梯度替换成我们合成后的 g_final_actor（实现“梯度空间混合”）

                    if self.max_grad_norm is not None:  # 若启用梯度裁剪，避免更新过大导致不稳定
                        nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)  # 对 agent 全部参数做范数裁剪
                    self.optimizer.step()  # 执行一次优化器 step（同时更新 actor + critic）

                    # Train synthesizer to align with linear target gradient G @ w（训练 synthesizer：让它输出的 g_task 贴近线性加权目标梯度 G w）
                    with torch.no_grad():  # 目标梯度只用于监督，不需要梯度
                        target_g = torch.matmul(G, mb_pref_avg)  # 线性组合得到 target 梯度（形状 [D]）

                    cos_sim = torch.nn.functional.cosine_similarity(  # 计算 g_task 和 target_g 的余弦相似度（越大越对齐）
                        g_task.unsqueeze(0), target_g.unsqueeze(0)  # 扩一维变成 batch=1 的形式
                    )
                    synth_loss = 1.0 - cos_sim.mean()  # 以 1-cos 作为损失（最大化 cos 等价于最小化 1-cos）
                    self.synth_optimizer.zero_grad()  # 清空 synthesizer 参数梯度
                    synth_loss.backward()  # 反传更新 synthesizer（注意：g_task 依赖 alpha，而 alpha 来自 synthesizer）
                    self.synth_optimizer.step()  # 更新 synthesizer 参数

                    loss = v_loss.detach()  # 用于日志：这里返回 value loss（detach 防止外部误用梯度）
                else:  # 分支：不使用 synthesizer，退化为“按 w 对各 objective policy loss 加权求和”的标量 PPO
                    raise ValueError("Synthesizer is not enabled")

        return loss  # 返回最后一次 mini-batch 的 loss（主要用于训练日志）

    def evaluate(self, eval_loader=None, test_weights=None, num_episodes=1):
        """Evaluate the agent with different preference weights."""
        logger = get_logger("morl.trainer.C", level=logging.INFO)
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
            obs = torch.Tensor(obs).to(self.device)

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
                        action, _, _, _ = self.agent.get_action_and_value(obs)

                    step_result = self.env.step(action.item() if not self.is_continuous else action.cpu().numpy())
                    real_next_obs = step_result[0]
                    term = step_result[2]
                    trunc = step_result[3]

                    episode_positions.append(real_next_obs[0])
                    episode_steps += 1

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

            max_len = max((len(c) for c in episode_curves), default=0)
            if max_len > 0:
                padded = np.full((len(episode_curves), max_len), np.nan, dtype=np.float32)
                for i, c in enumerate(episode_curves):
                    padded[i, : len(c)] = np.asarray(c, dtype=np.float32)
                w_curve = np.nanmean(padded, axis=0).astype(np.float32).tolist()
            else:
                w_curve = []
            logger.info("w=%s | mean_wdotr=%.4f | avg_pos=%.4f | steps=%s", w, mean_weighted, avg_pos, int(steps))
            results.append({
                'weight': [float(x) for x in w],
                'mean_weighted_reward': mean_weighted,
                'mean_wdotr_curve': w_curve,
                'avg_position': float(avg_pos),
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
    - Default to project_root/figures/eval_curves_C_{env}.png
    - If user passes a bare filename, place it under figures/
    - If user passes a path with directories (or absolute path), respect it as-is
    """
    figures_dir = ensure_dir(get_figures_dir())
    if not path_str:
        return str(figures_dir / f"eval_curves_C_{env_name}.png")
    p = Path(path_str)
    if p.is_absolute() or p.parent != Path("."):
        return str(p)
    return str(figures_dir / p.name)


def main():
    parser = argparse.ArgumentParser(description="Run PPOTrainerC standalone (colored logs).")
    parser.add_argument("--env", type=str, default="Walker2d-v5", choices=["CartPole-v1", "Walker2d-v5", "Humanoid-v5", "dm_control_walker-walk", "dm_control_hopper-hop", "dm_control_humanoid-walk"])
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--use_synth", action="store_true", help="Enable gradient synthesizer (Variant C gradient).")
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint path before training.")
    parser.add_argument("--save", type=str, default=None, help="Save checkpoint path after training.")
    parser.add_argument("--eval_only", action="store_true", help="Load and run evaluation only (skip training).")
    parser.add_argument("--plot", action="store_true", help="Save evaluation curves plot as PNG.")
    parser.add_argument("--plot_path", type=str, default=None, help="Output PNG path for --plot.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("morl.trainer.C", level=logging.INFO)
    logger.info("Device: %s", device)

    from src.environments import SteerableCartPoleWrapper, SteerableHumanoidWrapper, SteerableWalkerWrapper, make_dm_control_walker, make_dm_control_hopper, make_dm_control_humanoid

    if args.env == "dm_control_walker-walk":
        env = make_dm_control_walker()
    elif args.env == "dm_control_hopper-hop":
        env = make_dm_control_hopper()
    elif args.env == "dm_control_humanoid-walk":
        env = make_dm_control_humanoid()
    else:
        base_env = gym.make(args.env)
        if args.env == "CartPole-v1":
            env = SteerableCartPoleWrapper(base_env)
        elif args.env == "Walker2d-v5":
            env = SteerableWalkerWrapper(base_env)
        else:
            env = SteerableHumanoidWrapper(base_env)

    is_cartpole = args.env == "CartPole-v1"
    is_dm_control = args.env.startswith("dm_control_")
    num_objectives = 2 if (is_cartpole or is_dm_control) else 3
    num_steps = 128 if is_cartpole else 2048
    update_epochs = 4 if is_cartpole else 10
    ent_coef = 0.001 if is_cartpole else 0.0
    # Standalone Variant C gradient script uses vf_coef=0.05 for Walker2d
    vf_coef = 0.5 if is_cartpole else 0.05
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else (80000 if is_cartpole else 1000000)

    # Default behavior: on Walker2d we enable gradient synthesizer (matches your Variant C gradient).
    trainer = PPOTrainerC(
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
        use_gradient_synthesizer=(bool(args.use_synth) if args.env == "CartPole-v1" else True),
    )

    load_path = _resolve_checkpoint_path(args.load)
    save_path = _resolve_checkpoint_path(args.save)
    if load_path:
        ckpt = torch.load(load_path, map_location=device)
        if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
            trainer.agent.load_state_dict(ckpt["agent_state_dict"])
            # Synthesizer is optional
            if "synth_state_dict" in ckpt and trainer.synthesizer is not None:
                trainer.synthesizer.load_state_dict(ckpt["synth_state_dict"])
        else:
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
                plt.title(f"Variant C eval: mean(w·r_t) per step ({args.env})")
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
        payload = {
            "agent_state_dict": trainer.agent.state_dict(),
            "env": args.env,
            "num_objectives": num_objectives,
            "use_gradient_synthesizer": bool(trainer.use_gradient_synthesizer),
        }
        if trainer.synthesizer is not None:
            payload["synth_state_dict"] = trainer.synthesizer.state_dict()
        torch.save(payload, save_path)
        logger.info("[bold green]Saved checkpoint[/bold green]: %s", save_path)


if __name__ == "__main__":
    main()
