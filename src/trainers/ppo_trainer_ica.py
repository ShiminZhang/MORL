"""
PPO Trainer for Variant D (ICA): Gradient-space mixing with ICA whitening.

Based on `variant_d_ica.py`:
- Compute per-objective PPO policy gradients (vector reward, no scalarization in return/GAE).
- Form gradient matrix G in R^{D x K}, whiten it via Newton–Schulz inverse square-root:
    Z = G_norm @ W, where W ~ (G_norm^T G_norm)^(-1/2)
- Learn a synthesizer to output alpha in R^K (mixture weights) conditioned on (state, preference, W).
- Inject g_task = Z @ alpha (plus entropy) into actor gradients; optionally regularize with ICA kurtosis loss.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.trainers.morl_trainer import MORLTrainer
from src.utils.logger import get_logger
from src.utils.paths import ensure_dir, get_figures_dir, get_saved_agents_dir
from src.utils.weights import sample_poisson_weights


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal initialization for neural network layers."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class VariantICA_DiscreteAgent(nn.Module):
    """Vector critic + categorical policy (Discrete)."""

    def __init__(self, env, num_objectives: int):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
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

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        out = self.critic(x)
        return out.squeeze(0) if squeeze_out else out

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
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


class VariantICA_ContinuousAgent(nn.Module):
    """Vector critic + diagonal Gaussian policy (Continuous)."""

    def __init__(self, env, num_objectives: int):
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
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

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        out = self.critic(x)
        return out.squeeze(0) if squeeze_out else out

    def get_action_and_value(self, x: torch.Tensor, action: Optional[torch.Tensor] = None):
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


class NewtonSchulzIndependenceLoss(nn.Module):
    """
    ICA-style independence loss on gradient matrix G in R^{D x K}.

    Implements:
      A = G^T G  (KxK)
      W ~ A^(-1/2) via Newton–Schulz
      Z = G W
      loss = -sum_k log( E[ z_k^4 ] + 1 ) * kurtosis_weight
    """

    def __init__(
        self,
        num_objectives: int,
        ns_steps: int = 6,
        kurtosis_weight: float = 0.1,
        eps: float = 1e-6,
        clamp_z: float = 5.0,
    ):
        super().__init__()
        self.num_objectives = num_objectives
        self.ns_steps = ns_steps
        self.kurtosis_weight = kurtosis_weight
        self.eps = eps
        self.clamp_z = clamp_z

    def _newton_schulz_inverse_sqrt(self, A: torch.Tensor) -> torch.Tensor:
        dim = A.shape[0]
        I = torch.eye(dim, device=A.device, dtype=A.dtype)

        norm_A = torch.norm(A, p="fro")
        # Avoid divide-by-zero; in practice A should be PD due to eps*I anyway.
        norm_A = torch.clamp(norm_A, min=self.eps)

        Y = A / norm_A
        X = I.clone()
        for _ in range(self.ns_steps):
            T = X @ (Y @ X)
            X = 0.5 * X @ (3.0 * I - T)
        return X / torch.sqrt(norm_A)

    def forward(self, G: torch.Tensor):
        """
        Args:
            G: [D, K] gradient matrix
        Returns:
            ica_loss: scalar tensor
            Z: [D, K] whitened gradients
            W: [K, K] inverse-sqrt transform
            m4: [K] kurtosis-like moments used by the loss
        """
        if G.dim() != 2:
            raise ValueError(f"Expected G to have shape [D, K], got {tuple(G.shape)}")
        D, K = G.shape
        if K != self.num_objectives:
            raise ValueError(f"G has K={K}, but num_objectives={self.num_objectives}")

        A = G.t() @ G
        A = A + self.eps * torch.eye(K, device=G.device, dtype=G.dtype)

        W = self._newton_schulz_inverse_sqrt(A)
        Z = G @ W

        z_mean = torch.mean(Z, dim=0, keepdim=True)
        z_std = torch.std(Z, dim=0, keepdim=True) + 1e-8
        z_standardized = (Z - z_mean) / z_std
        z_standardized = torch.clamp(z_standardized, -self.clamp_z, self.clamp_z)

        m4 = torch.mean(z_standardized**4, dim=0)
        ica_loss = -torch.sum(torch.log(m4 + 1.0))
        return ica_loss * self.kurtosis_weight, Z, W, m4


class GradientAttentionSynthesizer(nn.Module):
    """
    Outputs alpha in R^K based on:
      - query: average state + preference
      - key: dummy S (K,) and a matrix (K,K) (here we use ICA's W)
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

        key_dim = num_objectives + (num_objectives * num_objectives)
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
        if p.grad is None:
            grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads) if grads else None


def _flatten_grads(grads: tuple[Optional[torch.Tensor], ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    out = []
    for g, p in zip(grads, params, strict=True):
        if g is None:
            out.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            out.append(g.reshape(-1))
    return torch.cat(out) if out else torch.zeros(0)


def _add_grad_vector(params: list[torch.nn.Parameter], grad_vector: torch.Tensor) -> None:
    pointer = 0
    for p in params:
        numel = p.numel()
        payload = grad_vector[pointer : pointer + numel].view_as(p)
        if p.grad is None:
            p.grad = payload.clone()
        else:
            p.grad.add_(payload)
        pointer += numel


class PPOTrainerICA(MORLTrainer):
    """
    PPO Trainer for Variant D (ICA).

    Notes:
    - Designed for vector reward settings where observation ends with preference weights.
    - Supports both discrete/continuous policies, but ICA whitening is most commonly used with continuous tasks.
    """

    def __init__(
        self,
        agent,
        env,
        device,
        learning_rate: float = 3e-4,
        num_steps: int = 128,
        total_timesteps: int = 80_000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 4,
        clip_coef: float = 0.2,
        ent_coef: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        num_objectives: int = 3,
        # ICA specifics
        ns_steps: int = 6,
        kurtosis_weight: float = 0.1,
        actor_clip_norm_small: float = 0.05,
        # Synthesizer
        use_gradient_synthesizer: bool = True,
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
        self.actor_clip_norm_small = actor_clip_norm_small

        self.use_gradient_synthesizer = use_gradient_synthesizer

        self.is_continuous = isinstance(env.action_space, gym.spaces.Box)
        if agent is None:
            if self.is_continuous:
                agent = VariantICA_ContinuousAgent(self.env, num_objectives=self.num_objectives)
            else:
                agent = VariantICA_DiscreteAgent(self.env, num_objectives=self.num_objectives)
            agent = agent.to(self.device)

        self.agent = agent
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate)
        self.actor_params = _get_actor_params(self.agent)

        self.ica_criterion = NewtonSchulzIndependenceLoss(
            num_objectives=self.num_objectives,
            ns_steps=ns_steps,
            kurtosis_weight=kurtosis_weight,
        ).to(self.device)

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
        self._init_buffers()

    def _init_buffers(self):
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

    def train(self, train_loader=None):  # 训练入口；train_loader 为接口保留（这里不使用）
        logger = get_logger("morl.trainer.ICA", level=logging.INFO)  # 创建 logger，打印训练过程
        global_step = 0  # 统计与环境交互的总步数（跨 update 累加）
        next_obs, _ = self.env.reset()  # 重置环境，拿到初始观测 obs（Gymnasium 返回 (obs, info)）
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)  # 转成 float32 tensor 并搬到 device
        next_done = torch.tensor(0.0, device=self.device)  # 当前状态是否为“episode 结束后”的标记（0/1），rollout buffer 要存

        logger.info(  # 打印训练配置摘要（总步数、update 次数、是否启用 synthesizer）
            "[bold]Starting Training[/bold] (Variant D: ICA Gradient Mixing) | steps=%s | updates=%s | synth=%s",
            self.total_timesteps,  # 总交互步数预算
            self.num_updates,  # update 次数 = total_timesteps // num_steps
            bool(self.use_gradient_synthesizer),  # 是否启用梯度 synthesizer
        )

        for update in range(1, self.num_updates + 1):  # 外层：每次 update 先 rollout，再做 PPO 更新
            # Rollout（采样 num_steps 步并写入 buffer）  # 说明：先收集一段轨迹，再训练
            for step in range(self.num_steps):  # 内层：与环境交互 T=num_steps 次
                global_step += 1  # 全局交互步数 +1
                self.obs[step] = next_obs  # 存 s_t
                self.dones[step] = next_done  # 存 done_t（表示当前 s_t 是否处在 episode 终止之后）
                self.contexts[step] = next_obs[-self.num_objectives :]  # 从观测尾部取偏好/权重 w_t（环境 wrapper 的约定）

                with torch.no_grad():  # rollout 不需要梯度，节省显存/加速
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs)  # 采样动作 a_t，得到 logπ_old(a_t|s_t) 和向量 V(s_t)
                    self.values[step] = value  # 保存向量 value（K 维），用于后续 GAE/returns

                self.actions[step] = action  # 保存动作 a_t（连续为向量，离散为标量）
                self.logprobs[step] = logprob  # 保存旧策略下 logprob（PPO ratio 的分母部分）

                if self.is_continuous:  # 连续动作空间（Box）
                    action_np = action.cpu().numpy()  # env.step 需要 numpy
                else:  # 离散动作空间（Discrete）
                    action_np = action.item()  # env.step 需要 int

                real_next_obs, _, terminated, truncated, info = self.env.step(action_np)  # 与环境交互一步（奖励这里不用 step 返回的标量）
                done = terminated or truncated  # episode 是否结束（Gymnasium 终止或截断都算 done）

                r_vec_raw = self.env.get_reward()  # 从 wrapper 获取“上一步”的向量 reward（长度 K）
                if r_vec_raw is None:  # 兜底：环境没有返回向量 reward 时，避免崩溃
                    r_vec_raw = [0.0] * self.num_objectives  # 用 0 向量代替
                self.rewards[step] = torch.as_tensor(r_vec_raw, dtype=torch.float32, device=self.device)  # 保存向量奖励 r_t 到 buffer

                next_obs = torch.tensor(real_next_obs, dtype=torch.float32, device=self.device)  # 更新 next_obs 为 s_{t+1}
                next_done = torch.tensor(float(done), device=self.device)  # 更新 next_done 为 done_{t+1}（0.0/1.0）

                if done:  # 如果 episode 结束
                    next_obs_np, _ = self.env.reset()  # 立刻 reset，继续采样
                    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)  # reset 后的初始 obs 转 tensor
                    next_done = torch.tensor(0.0, device=self.device)  # reset 后 done=0

            advantages, returns = self._compute_gae(next_obs, next_done)  # 计算向量优势 A（GAE）和向量 returns（每个 objective 一份）
            loss, metrics = self._update(advantages, returns)  # PPO 更新（包含 ICA 白化 + synthesizer + actor 梯度注入）

            if update % 20 == 0:  # 每 20 次 update 打印一次训练指标
                train_scalar_rewards = (self.rewards * self.contexts).sum(dim=1).mean().item()  # 用 w·r 得到标量 reward 并取平均（用于快速观察趋势）
                logger.info(  # 打印 loss + 平均标量 reward + ICA loss + synthesizer 对齐度
                    "Update %s/%s | loss=%.4f | mean_scalar_reward=%.2f | ica=%.4f | synth_align=%.3f",
                    update,  # 当前 update
                    self.num_updates,  # 总 update
                    float(loss.item()) if hasattr(loss, "item") else float(loss),  # loss 可能是 tensor，这里转 float 便于打印
                    train_scalar_rewards,  # 平均 w·r
                    float(metrics.get("ica_loss", 0.0)),  # 平均 ICA loss（mini-batch 聚合后）
                    float(metrics.get("synth_align", 0.0)),  # 平均对齐度 cos(g_task, G@w)
                )

        logger.info("[bold green]Training Finished![/bold green]")  # 训练结束提示
        return self.agent  # 返回训练后的 agent（便于外部保存/评估）

    def _compute_gae(self, next_obs: torch.Tensor, next_done: torch.Tensor):  # 计算向量版 GAE（每个 objective 各算一份优势）
        with torch.no_grad():  # GAE 是后处理，不需要梯度
            next_value = self.agent.get_value(next_obs)  # 估计最后一个 next_obs 的向量价值 V(s_T)
            next_value = next_value.reshape(1, -1) if next_value.dim() == 1 else next_value  # 保证形状为 [1, K] 便于索引

            advantages = torch.zeros_like(self.rewards).to(self.device)  # 初始化 advantages buffer，shape=[T, K]
            lastgaelam = torch.zeros(self.num_objectives).to(self.device)  # 反向递推的累积项 lastgaelam（K 维）

            for t in reversed(range(self.num_steps)):  # 反向遍历时间步 t=T-1..0（GAE 标准实现）
                if t == self.num_steps - 1:  # 最后一个时间步：bootstrap 来自外部 next_obs/next_done
                    nextnonterminal = 1.0 - next_done  # 如果 next_done=1 则不 bootstrap（乘 0）
                    nextvalues = next_value[0]  # 取出 [K] 的 V(s_T)
                else:  # 非最后时间步：bootstrap 来自 buffer 中 t+1
                    nextnonterminal = 1.0 - self.dones[t + 1]  # done_{t+1} 决定是否 bootstrap
                    nextvalues = self.values[t + 1]  # V(s_{t+1})（向量，K 维）

                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]  # 向量 TD 残差 δ_t（逐 objective）
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam  # 向量 GAE 递推公式

            returns = advantages + self.values  # 返回向量 returns：R_t = A_t + V_t
        return advantages, returns  # 返回 [T,K] advantages 与 [T,K] returns

    def _update(self, advantages: torch.Tensor, returns: torch.Tensor):  # PPO 更新（核心：每 objective 梯度→ICA 白化→注入 actor 梯度）
        b_obs = self.obs.reshape((-1, self.env.observation_space.shape[0]))  # 展平观测 buffer 为 batch：[T, obs_dim]
        b_logprobs = self.logprobs.reshape(-1)  # 展平旧策略 logprob：[T]

        if self.is_continuous:  # 连续动作空间（Box）
            b_actions = self.actions.reshape((-1, self.env.action_space.shape[0]))  # 展平动作：[T, act_dim]
        else:  # 离散动作空间（Discrete）
            b_actions = self.actions.reshape(-1)  # 展平动作：[T]

        b_vector_advantages = advantages.reshape((-1, self.num_objectives))  # 展平向量 advantages：[T, K]
        b_returns = returns.reshape((-1, self.num_objectives))  # 展平向量 returns：[T, K]
        b_contexts = self.contexts.reshape((-1, self.num_objectives))  # 展平偏好/权重 w：[T, K]

        b_inds = np.arange(self.num_steps)  # 构造索引 [0..T-1]，用于随机抽 mini-batch

        last_loss = torch.tensor(0.0, device=self.device)  # 用于返回：记录最后一次 mini-batch 的 loss
        metrics: dict[str, float] = {"ica_loss": 0.0, "synth_align": 0.0}  # 聚合统计：ICA loss 与 synthesizer 对齐度
        metrics_count = 0  # mini-batch 计数，用于求平均

        for _ in range(self.update_epochs):  # PPO 多轮 epoch（对同一段 rollout 反复优化）
            np.random.shuffle(b_inds)  # 打乱样本顺序，提升优化稳定性
            for start in range(0, self.num_steps, self.batch_size):  # 以 batch_size 为步长遍历
                end = start + self.batch_size  # 当前 mini-batch 的结束位置
                mb_inds = b_inds[start:end]  # 取出当前 mini-batch 索引

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(  # 用当前策略重新评估 mini-batch
                    b_obs[mb_inds],  # 输入状态 s
                    b_actions[mb_inds],  # 输入固定动作 a（PPO 需要同一动作下的新旧概率）
                )
                logratio = newlogprob - b_logprobs[mb_inds]  # log(π_new(a|s) / π_old(a|s))
                ratio = logratio.exp()  # 概率比 r_t(θ)=π_new/π_old

                # Per-objective policy-gradient vectors (create_graph=True for ICA loss)  # 说明：对每个 objective 单独算 actor 梯度向量
                grad_vectors: list[torch.Tensor] = []  # 存每个 objective 的 actor 梯度 g_k（展平向量）
                for k in range(self.num_objectives):  # 遍历目标维度 k=0..K-1
                    adv_k = b_vector_advantages[mb_inds, k]  # 取出第 k 个 objective 的 advantage：[B]
                    adv_k = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-8)  # 标准化 advantage，稳定训练

                    loss_k_element = torch.max(  # PPO clipped surrogate（逐样本）
                        -adv_k * ratio,  # unclipped：-A * r
                        -adv_k * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef),  # clipped：-A * clip(r)
                    )
                    w_k = b_contexts[mb_inds, k]  # 取出偏好权重 w_k：[B]
                    loss_k = (w_k * loss_k_element).mean()  # 用 w_k 对该 objective 的 policy loss 加权并求均值（vd 的做法）

                    grads_k = torch.autograd.grad(  # 直接对 actor_params 求梯度（不使用 backward），并保留计算图给 ICA loss
                        loss_k,  # 被求导的 loss
                        self.actor_params,  # 只对 actor 参数求导（critic 不参与）
                        create_graph=True,  # 让“梯度”本身可求导（ICA 正则会对梯度再反传）
                        retain_graph=True,  # 还要继续对其它 objective 求梯度，必须保留图
                        allow_unused=True,  # 允许某些参数梯度为 None（不会报错）
                    )
                    grad_vectors.append(_flatten_grads(grads_k, self.actor_params))  # 把分层梯度拼成一条向量 g_k（长度 D）

                # Entropy gradient vector (for injection)  # 说明：熵正则的梯度也要加到最终 actor 梯度里
                self.optimizer.zero_grad()  # 清空上一轮梯度
                entropy_loss = -self.ent_coef * entropy.mean()  # 熵项：最大化 entropy 等价于最小化 -ent_coef * entropy
                entropy_loss.backward(retain_graph=True)  # 反传得到熵项对应的 actor 梯度（写入 .grad）
                g_ent = _get_grad_vector(self.actor_params)  # 将当前 actor 参数梯度展平成向量 g_ent（长度 D）
                if g_ent is None:  # 防御：理论上不会为 None（我们会填 0），这里兜底
                    g_ent = torch.zeros_like(grad_vectors[0])  # 用 0 向量代替

                G = torch.stack(grad_vectors, dim=1)  # 拼成梯度矩阵 G=[D, K]（每列对应一个 objective 的梯度）
                G_norm = G / (torch.norm(G, p=2, dim=0, keepdim=True) + 1e-8)  # 每列 L2 归一化，减少尺度差异对 ICA 的影响

                ica_loss_val, Z, W, _m4 = self.ica_criterion(G_norm)  # ICA：得到正则 loss、白化后的 Z、以及变换矩阵 W

                mb_obs_avg = b_obs[mb_inds].mean(dim=0)  # 当前 mini-batch 的平均 state（synthesizer 的 query 输入）
                mb_pref_avg = b_contexts[mb_inds].mean(dim=0)  # 当前 mini-batch 的平均偏好 w（synthesizer 的 query 输入）

                if self.use_gradient_synthesizer:  # 若启用 synthesizer，用它输出 alpha
                    assert self.synthesizer is not None and self.synth_optimizer is not None  # 确保 synthesizer/optimizer 已初始化
                    dummy_S = torch.ones(self.num_objectives, device=self.device, dtype=W.dtype)  # 按 vd：S 用全 1 占位（接口保持一致）
                    alpha, _ = self.synthesizer(mb_obs_avg, mb_pref_avg, dummy_S, W)  # synthesizer 输出 alpha（混合系数）
                else:  # 否则退化：直接用偏好 w 作为 alpha
                    alpha = mb_pref_avg  # alpha=w（简单线性混合）

                g_task = torch.matmul(Z, alpha)  # 用白化基 Z 组合得到任务梯度 g_task=[D]
                g_final_actor = g_task + g_ent  # 最终要注入 actor 的梯度：任务梯度 + 熵梯度

                # Critic update + ICA regularization (produces actor grads via ICA loss)  # 说明：先反传 value loss + ICA loss 得到 critic 梯度与 ICA 的高阶梯度
                self.optimizer.zero_grad()  # 清空 agent 参数梯度
                if self.synth_optimizer is not None:  # 若有 synthesizer optimizer
                    self.synth_optimizer.zero_grad()  # 同步清空 synthesizer 梯度（后面会训练它）

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()  # 向量 value 的 MSE（对 K 维一起平均）
                total_loss = self.vf_coef * v_loss + ica_loss_val  # 总损失：vf_coef * v_loss + ICA 正则项
                total_loss.backward(retain_graph=True)  # 反传：critic 得到梯度；ICA loss 通过 create_graph 路径回到 actor

                # Small clip on actor params (matches `variant_d_ica.py`)  # 说明：对 actor 参数单独小阈值裁剪（vd 用 0.05）
                if self.actor_clip_norm_small is not None:  # 若设置了小裁剪阈值
                    nn.utils.clip_grad_norm_(self.actor_params, max_norm=self.actor_clip_norm_small)  # 防止 actor 梯度过大导致不稳定

                # Train synthesizer to align with target gradient G @ w  # 说明：训练 synthesizer，使 g_task 对齐线性目标梯度 G@w
                synth_align = 0.0  # 记录余弦相似度（用于日志）
                if self.use_gradient_synthesizer and self.synth_optimizer is not None:  # 只有启用 synthesizer 才训练
                    with torch.no_grad():  # target_g 仅作监督信号，不需要梯度
                        target_g = torch.matmul(G, mb_pref_avg)  # 线性组合得到目标梯度 target_g=[D]
                    cos_sim = F.cosine_similarity(g_task.unsqueeze(0), target_g.unsqueeze(0))  # 计算 cos(g_task, target_g)
                    synth_loss = 1.0 - cos_sim.mean()  # 损失 = 1 - cos（越小越对齐）
                    synth_align = float(cos_sim.mean().detach().cpu().item())  # 记录当前对齐分数
                    synth_loss.backward()  # 反传更新 synthesizer（因为 g_task 依赖 alpha，alpha 依赖 synthesizer）

                # Inject actor gradient (additive, preserving ICA gradients)  # 说明：把 g_final_actor 以“加法”方式注入，避免覆盖 ICA loss 的梯度
                _add_grad_vector(self.actor_params, g_final_actor)  # actor.grad += g_final_actor（逐参数切片相加）

                if self.max_grad_norm is not None:  # 若启用全参数梯度裁剪
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)  # 对 agent 所有参数做裁剪（PPO 常用）
                self.optimizer.step()  # 更新 agent（actor + critic）
                if self.synth_optimizer is not None:  # 若 synthesizer 存在
                    self.synth_optimizer.step()  # 更新 synthesizer

                last_loss = total_loss.detach()  # 记录最后一次 mini-batch 的 total_loss（detach 只用于日志/返回）
                metrics["ica_loss"] += float(ica_loss_val.detach().cpu().item())  # 累加 ICA loss（转成 float）
                metrics["synth_align"] += float(synth_align)  # 累加对齐分数
                metrics_count += 1  # mini-batch 计数 +1

        if metrics_count > 0:  # 若至少有 1 个 mini-batch
            metrics["ica_loss"] /= metrics_count  # 求平均 ICA loss
            metrics["synth_align"] /= metrics_count  # 求平均对齐分数

        return last_loss, metrics  # 返回 loss（用于 train 打印）和 metrics（用于 train 打印）

    def evaluate(self, eval_loader=None, test_weights=None, num_episodes: int = 1):
        logger = get_logger("morl.trainer.ICA", level=logging.INFO)
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
            obs, _ = self.env.reset(options={"w": w})
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

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

                    episode_positions.append(float(real_next_obs[0]) if len(real_next_obs) > 0 else 0.0)
                    episode_steps += 1

                    r_vec_raw = self.env.get_reward()
                    if r_vec_raw is None:
                        r_vec_raw = [0.0] * self.num_objectives
                    r_vec = np.asarray(r_vec_raw, dtype=np.float32)
                    step_wdotr = float(np.dot(np.asarray(w, dtype=np.float32), r_vec))
                    episode_weighted_sum += step_wdotr
                    step_curve.append(step_wdotr)

                    obs = torch.tensor(real_next_obs, dtype=torch.float32, device=self.device)
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
            results.append(
                {
                    "weight": [float(x) for x in w],
                    "mean_weighted_reward": mean_weighted,
                    "mean_wdotr_curve": w_curve,
                    "avg_position": float(avg_pos),
                    "steps": int(steps),
                }
            )

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
    - Default to project_root/figures/eval_curves_D_ICA_{env}.png
    - If user passes a bare filename, place it under figures/
    - If user passes a path with directories (or absolute path), respect it as-is
    """
    figures_dir = ensure_dir(get_figures_dir())
    if not path_str:
        return str(figures_dir / f"eval_curves_D_ICA_{env_name}.png")
    p = Path(path_str)
    if p.is_absolute() or p.parent != Path("."):
        return str(p)
    return str(figures_dir / p.name)


def main():
    parser = argparse.ArgumentParser(description="Run PPOTrainerICA standalone (Variant D / ICA).")
    parser.add_argument("--env", type=str, default="Walker2d-v5", choices=["CartPole-v1", "Walker2d-v5", "Humanoid-v5", "dm_control_walker-walk", "dm_control_hopper-hop", "dm_control_humanoid-walk"])
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint path before training.")
    parser.add_argument("--save", type=str, default=None, help="Save checkpoint path after training.")
    parser.add_argument("--eval_only", action="store_true", help="Load and run evaluation only (skip training).")
    parser.add_argument("--plot", action="store_true", help="Save evaluation curves plot as PNG.")
    parser.add_argument("--plot_path", type=str, default=None, help="Output PNG path for --plot.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("morl.trainer.ICA", level=logging.INFO)
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
        elif args.env == "Humanoid-v5":
            env = SteerableHumanoidWrapper(base_env)
        else:
            raise ValueError(f"Unsupported environment: {args.env}")

    is_cartpole = args.env == "CartPole-v1"
    is_dm_control = args.env.startswith("dm_control_")
    num_objectives = 2 if (is_cartpole or is_dm_control) else 3
    num_steps = 128 if is_cartpole else 2048
    update_epochs = 4 if is_cartpole else 10
    ent_coef = 0.001 if is_cartpole else 0.0
    vf_coef = 0.5 if is_cartpole else 0.05
    total_timesteps = args.total_timesteps if args.total_timesteps is not None else (80_000 if is_cartpole else 1_000_000)

    trainer = PPOTrainerICA(
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
        use_gradient_synthesizer=(not is_cartpole),  # MuJoCo: True by default; CartPole: False (K=2 not matching original ICA setup)
    )

    load_path = _resolve_checkpoint_path(args.load)
    save_path = _resolve_checkpoint_path(args.save)
    if load_path:
        ckpt = torch.load(load_path, map_location=device)
        if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
            trainer.agent.load_state_dict(ckpt["agent_state_dict"])
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

            curves = [
                r
                for r in results
                if isinstance(r, dict)
                and r.get("mean_wdotr_curve")
                and r.get("weight") is not None
            ]
            if not curves:
                logger.warning("No curves found to plot.")
            else:
                plt.figure(figsize=(10, 5))
                for i, r in enumerate(curves):
                    y = r["mean_wdotr_curve"]
                    plt.plot(y, linewidth=1.5, label=f"w{i}")
                plt.title(f"Variant D (ICA) eval: mean(w·r_t) per step ({args.env})")
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
        }
        if trainer.synthesizer is not None:
            payload["synth_state_dict"] = trainer.synthesizer.state_dict()
        torch.save(payload, save_path)
        logger.info("[bold green]Saved checkpoint[/bold green]: %s", save_path)


if __name__ == "__main__":
    main()

