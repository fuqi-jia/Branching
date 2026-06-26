"""阶段二：solver-in-the-loop 微调 (plan 8.2)。

提供两种修正分布偏移的方式:

- **DAgger**：在策略当前访问到的状态上收集专家标签，复用 imitation 损失再训练。
- **REINFORCE**：用真实求解反馈（wall-clock time、节点数、objective gap AUC）作为
  奖励的策略梯度，带移动平均 baseline 与熵正则。

奖励建议见 plan 8.2：完整求解 ``-log(1+solve_time)`` 或 ``-nodes``；anytime OMT 用
incumbent improvement 与 final gap；局部 step 用 bound improvement / conflict quality。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn.functional as F

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, RankingExample, TrainConfig


@dataclass
class TrajectoryStep:
    """轨迹中一步：在 ``graph`` 上选择了候选 ``chosen_bool_local``，获得即时奖励。"""

    graph: HeteroGraph
    chosen_bool_local: int
    reward: float = 0.0


@dataclass
class Trajectory:
    """一次完整求解产生的决策序列与终局奖励。"""

    steps: list[TrajectoryStep] = field(default_factory=list)
    terminal_reward: float = 0.0  # 求解结束后的总体奖励 (如 -log(1+solve_time))


@dataclass
class FinetuneConfig:
    lr: float = 3e-4
    gamma: float = 0.99            # 折扣因子
    entropy_coef: float = 1e-2
    baseline_momentum: float = 0.9
    grad_clip: float = 5.0
    device: str = "cpu"


class SolverInLoopFinetuner:
    """对已训练策略做在线微调。"""

    def __init__(self, policy: BranchingPolicy, config: FinetuneConfig = FinetuneConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.opt = torch.optim.Adam(policy.parameters(), lr=config.lr)
        self._baseline = 0.0

    # ------------------------------------------------------------------ #
    # DAgger：在策略访问状态上用专家标签再训练
    # ------------------------------------------------------------------ #
    def dagger_update(self, labeled_examples: Iterable[RankingExample],
                      epochs: int = 1) -> list[dict[str, float]]:
        trainer = ImitationTrainer(
            self.policy, TrainConfig(lr=self.config.lr, device=self.config.device)
        )
        return trainer.fit(labeled_examples, epochs=epochs)

    # ------------------------------------------------------------------ #
    # REINFORCE：用求解反馈做策略梯度
    # ------------------------------------------------------------------ #
    def reinforce_update(self, trajectory: Trajectory) -> dict[str, float]:
        self.policy.train()
        dev = self.config.device
        returns = self._discounted_returns(trajectory)

        policy_loss = torch.zeros((), device=dev)
        entropy = torch.zeros((), device=dev)
        n = 0
        for step, G in zip(trajectory.steps, returns):
            g = step.graph.to(dev)
            out = self.policy(g)
            probs = out.masked_bool_probs()
            if probs.numel() == 0 or step.chosen_bool_local >= probs.numel():
                continue
            logp = torch.log(probs[step.chosen_bool_local].clamp_min(1e-12))
            advantage = G - self._baseline
            policy_loss = policy_loss - logp * advantage
            ent = -(probs.clamp_min(1e-12) * torch.log(probs.clamp_min(1e-12))).sum()
            entropy = entropy + ent
            n += 1

        if n == 0:
            return {"loss": 0.0, "baseline": self._baseline, "steps": 0}

        loss = (policy_loss - self.config.entropy_coef * entropy) / n
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.opt.step()

        mean_return = float(sum(returns) / len(returns))
        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1 - m) * mean_return
        return {"loss": float(loss), "baseline": self._baseline, "steps": n}

    # ------------------------------------------------------------------ #
    def _discounted_returns(self, traj: Trajectory) -> list[float]:
        out: list[float] = []
        running = traj.terminal_reward
        for step in reversed(traj.steps):
            running = step.reward + self.config.gamma * running
            out.append(running)
        out.reverse()
        return out
