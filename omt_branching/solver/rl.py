"""Solver-in-the-Loop 强化学习训练（阶段二，plan 8.2）。

在**真实 z3 GOMT 求解回路**中采集轨迹并用 REINFORCE 微调 GNN 分支策略。与 z3 的
所有交互都经 ``omt_branching.solver`` 的接口完成（``Z3Backend`` / ``Z3SnapshotExtractor``
/ ``GOMTSolver``），策略只在 calculus 唯一的启发式自由点 **F-Split** 处以**采样**方式
选择分支对象与方向，从而得到可微的动作对数似然用于策略梯度。

奖励设计：

- **单步 reward**：一次 F-Split 分支后，由后续 ``Solve`` 得到的**局部最优值
  (incumbent) 提升**。MAX 时 ``reward = 新值 - 旧值``；MIN 时 ``reward = 旧值 - 新值``
  （越优 reward 越大）。见 :meth:`SolverInLoopRLTrainer._build_episode`。
- **终局 loss/penalty**：整个 OMT 问题求解消耗的 **z3 rlimit count 增长**，作为负向
  终局奖励（rlimit 是与硬件/负载无关的确定性工作量计量，比 wall-clock 更稳定可复现），
  驱动策略以更少的求解开销到达最优。

训练时先在无梯度下 rollout 采样动作并记录动作对应的图；更新时对记录的每个图重跑
一次前向（图是固定输入，梯度只流向策略参数），按折扣回报做 REINFORCE，带移动平均
baseline 与熵正则。
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Optional

import torch

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver.calculus import GOMTConfig, GOMTResult, GOMTSolver
from omt_branching.solver.interfaces import Sense, SplitDecision
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.strategy import _bool_split, _numeric_split
from omt_branching.solver.z3_backend import Z3Backend


# --------------------------------------------------------------------------- #
# 配置与轨迹数据结构
# --------------------------------------------------------------------------- #
@dataclass
class RLConfig:
    """RL 训练配置。"""

    lr: float = 3e-4
    gamma: float = 0.99              # 折扣因子
    entropy_coef: float = 1e-2       # 熵正则权重（鼓励探索）
    baseline_momentum: float = 0.9   # 移动平均 baseline 动量
    grad_clip: float = 5.0
    device: str = "cpu"

    # 奖励塑形
    reward_scale: float = 1.0        # incumbent 提升的缩放
    rlimit_penalty_coef: float = 1.0  # 终局 rlimit 代价 penalty 权重
    use_log_cost: bool = True        # True: 用 log(1+rlimit) 压缩代价尺度

    # 求解回路
    max_steps: int = 10_000          # calculus 派生步数上限
    max_split_depth: int = 6         # 每个 Δ-round 的 split 预算（保证终止）


@dataclass
class RLStep:
    """一次被记录的 F-Split 决策（一个 RL 动作）。"""

    graph: HeteroGraph               # 该决策点的图（供更新时重跑前向）
    head: str                        # "numeric" | "bool"
    chosen_local: int                # 所选候选的图内局部索引
    direction: bool                  # numeric: branch_up；bool: phase_true
    value_at_decision: Optional[float] = None  # 决策时的 incumbent 目标值


@dataclass
class RLEpisode:
    """一次完整 OMT 求解产生的轨迹与奖励。"""

    steps: list[RLStep] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)  # 与 steps 对齐的单步 reward
    terminal_reward: float = 0.0     # rlimit 代价导致的终局负向奖励
    rlimit: int = 0                  # 本次求解累计的 z3 rlimit count
    runtime: float = 0.0             # wall-clock（仅供参考/日志）
    final_value: Optional[float] = None
    optimal: bool = False
    result_stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 采样式记录策略：在 F-Split 处采样动作并记录
# --------------------------------------------------------------------------- #
class RLRecordingStrategy:
    """训练/评估用的分支策略。

    - ``sample=True``（训练）：按策略分布采样候选与方向，用于探索并产生 on-policy 轨迹。
    - ``sample=False``（评估）：取 argmax / 阈值，等价于 :class:`NeuralStrategy` 的确定性行为。

    每次成功 split 都把 ``(graph, head, chosen_local, direction, incumbent_value)``
    追加到 ``self.steps``，供 :class:`SolverInLoopRLTrainer` 计算奖励与梯度。
    """

    def __init__(self, problem, policy: BranchingPolicy, config: RLConfig,
                 sample: bool = True):
        from omt_branching.solver.extractor import Z3SnapshotExtractor

        self.problem = problem
        self.policy = policy
        self.config = config
        self.sample = sample
        self.builder = GraphBuilder(DEFAULT_FEATURE_SPEC)
        self.extractor = Z3SnapshotExtractor(problem)
        self.steps: list[RLStep] = []

    def reset(self) -> None:
        self.steps = []

    def propose(self, state, backend) -> SplitDecision:
        depth = state.stats.get("branch_depth", 0)
        if depth >= self.config.max_split_depth:
            return SplitDecision.resolve()

        try:
            view = replace(state, hard=backend.conjoin(state.hard, state.top))
            extraction = self.extractor.extract(view, backend)
            graph = self.builder.build(extraction.snapshot)
            out = self.policy.infer(graph)
        except Exception as exc:  # 抽取/推理异常一律回退，不中断搜索
            warnings.warn(f"RLRecordingStrategy 抽取/推理失败，回退 resolve: {exc!r}")
            return SplitDecision.resolve()

        action = self._select_action(out)
        if action is None:
            return SplitDecision.resolve()
        head, chosen_local, direction = action

        psi = state.top
        subs: Optional[list] = None
        if head == "numeric":
            sid = graph.solver_id(NodeType.NUMERIC_VAR, chosen_local)
            handle = extraction.numeric_handles.get(sid) if sid is not None else None
            if handle is not None:
                subs = _numeric_split(handle, psi, backend, direction)
        else:
            sid = graph.solver_id(NodeType.BOOL_VAR, chosen_local)
            handle = extraction.atom_handles.get(sid) if sid is not None else None
            if handle is not None:
                subs = _bool_split(handle, psi, backend, direction)

        if subs is None:
            return SplitDecision.resolve()

        cur_val = self._incumbent_value(state, backend)
        self.steps.append(RLStep(
            graph=graph, head=head, chosen_local=chosen_local,
            direction=direction, value_at_decision=cur_val,
        ))
        state.stats["branch_depth"] = depth + 1
        return SplitDecision.split(subs, source="rl")

    # ------------------------------------------------------------------ #
    def _select_action(self, out) -> Optional[tuple[str, int, bool]]:
        """优先数值域切分（B&B / 整数 head），否则布尔原子切分。"""
        num_probs = out.masked_numeric_probs()
        if num_probs.numel() > 0 and out.candidate_numeric_local and float(num_probs.sum()) > 0:
            idx = self._sample_index(num_probs)
            dir_p = float(torch.sigmoid(out.int_dir_logits[idx]))
            direction = self._sample_bool(dir_p)
            return "numeric", idx, direction

        bool_probs = out.masked_bool_probs()
        if bool_probs.numel() > 0 and out.candidate_bool_local and float(bool_probs.sum()) > 0:
            idx = self._sample_index(bool_probs)
            ph_p = float(torch.sigmoid(out.phase_logits[idx]))
            direction = self._sample_bool(ph_p)
            return "bool", idx, direction

        return None

    def _sample_index(self, probs: torch.Tensor) -> int:
        if self.sample:
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(probs).item())

    def _sample_bool(self, p_true: float) -> bool:
        if self.sample:
            return float(torch.rand(())) < p_true
        return p_true >= 0.5

    @staticmethod
    def _incumbent_value(state, backend) -> Optional[float]:
        if state.incumbent is None:
            return None
        try:
            return float(backend.value(state.incumbent, state.objective))
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# 强化学习训练器
# --------------------------------------------------------------------------- #
class SolverInLoopRLTrainer:
    """在 z3 GOMT 回路中用 REINFORCE 微调分支策略。"""

    def __init__(self, policy: BranchingPolicy, config: RLConfig = RLConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.opt = torch.optim.Adam(policy.parameters(), lr=config.lr)
        self._baseline = 0.0

    # ------------------------------------------------------------------ #
    # 采集：跑一次真实求解，记录轨迹并算奖励
    # ------------------------------------------------------------------ #
    def collect_episode(self, hard_list, objective, sense: Sense) -> RLEpisode:
        backend = Z3Backend()
        problem = GOMTProblem(hard_list=tuple(hard_list), objective=objective, sense=sense)
        strategy = RLRecordingStrategy(problem, self.policy, self.config, sample=True)
        solver = GOMTSolver(
            problem, backend, strategy,
            GOMTConfig(max_steps=self.config.max_steps, f_sat_mode="plain"),
        )
        t0 = time.perf_counter()
        result = solver.run()
        runtime = time.perf_counter() - t0
        return self._build_episode(strategy.steps, result, sense,
                                   rlimit=backend.rlimit_count, runtime=runtime)

    def _build_episode(self, steps: list[RLStep], result: GOMTResult,
                       sense: Sense, rlimit: int, runtime: float) -> RLEpisode:
        final_val = float(result.value) if result.value is not None else None
        sign = 1.0 if sense is Sense.MAX else -1.0  # 使“更优”对应正 reward

        rewards: list[float] = []
        for i, step in enumerate(steps):
            nxt = steps[i + 1].value_at_decision if i + 1 < len(steps) else final_val
            cur = step.value_at_decision
            if cur is None or nxt is None:
                rewards.append(0.0)
            else:
                rewards.append(sign * (nxt - cur) * self.config.reward_scale)

        # 终局代价：用 z3 rlimit count 增长（替代 wall-clock）。
        cost = math.log1p(rlimit) if self.config.use_log_cost else float(rlimit)
        terminal = -self.config.rlimit_penalty_coef * cost

        return RLEpisode(
            steps=steps, rewards=rewards, terminal_reward=terminal, rlimit=rlimit,
            runtime=runtime, final_value=final_val, optimal=result.optimal,
            result_stats=dict(result.stats),
        )

    # ------------------------------------------------------------------ #
    # 更新：REINFORCE
    # ------------------------------------------------------------------ #
    def update(self, episode: RLEpisode) -> dict[str, float]:
        if not episode.steps:
            return {"loss": 0.0, "mean_return": episode.terminal_reward,
                    "baseline": self._baseline, "steps": 0}

        self.policy.train()
        dev = self.config.device
        returns = self._discounted_returns(episode)

        policy_loss = torch.zeros((), device=dev)
        entropy = torch.zeros((), device=dev)
        n = 0
        for step, G in zip(episode.steps, returns):
            g = step.graph.to(dev)
            out = self.policy(g)
            logp, ent = self._action_logp_entropy(out, step)
            if logp is None:
                continue
            advantage = G - self._baseline
            policy_loss = policy_loss - logp * advantage
            entropy = entropy + ent
            n += 1

        if n == 0:
            return {"loss": 0.0, "mean_return": float(sum(returns) / len(returns)),
                    "baseline": self._baseline, "steps": 0}

        loss = (policy_loss - self.config.entropy_coef * entropy) / n
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)
        self.opt.step()

        mean_return = float(sum(returns) / len(returns))
        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1 - m) * mean_return
        return {"loss": float(loss), "mean_return": mean_return,
                "baseline": self._baseline, "steps": n}

    def _discounted_returns(self, ep: RLEpisode) -> list[float]:
        out: list[float] = []
        running = ep.terminal_reward
        for r in reversed(ep.rewards):
            running = r + self.config.gamma * running
            out.append(running)
        out.reverse()
        return out

    def _action_logp_entropy(self, out, step: RLStep):
        """重算所记录动作的对数似然与该步分布熵（selection + direction）。"""
        if step.head == "numeric":
            probs = out.masked_numeric_probs()
            dir_logits = out.int_dir_logits
        else:
            probs = out.masked_bool_probs()
            dir_logits = out.phase_logits

        if probs.numel() == 0 or step.chosen_local >= probs.numel():
            return None, None

        p_sel = probs[step.chosen_local].clamp_min(1e-12)
        dir_p = torch.sigmoid(dir_logits[step.chosen_local]).clamp(1e-6, 1 - 1e-6)
        p_dir = dir_p if step.direction else (1.0 - dir_p)

        logp = torch.log(p_sel) + torch.log(p_dir)
        ent = _categorical_entropy(probs) + _bernoulli_entropy(dir_p)
        return logp, ent

    # ------------------------------------------------------------------ #
    # 训练主循环 & 评估
    # ------------------------------------------------------------------ #
    def train(self, instances, iterations: int = 1,
              log: bool = True) -> list[dict[str, float]]:
        """对一组实例反复采集+更新。

        ``instances``：``(hard_list, objective, sense)`` 三元组的可迭代对象。
        """
        instances = list(instances)
        history: list[dict[str, float]] = []
        for it in range(iterations):
            for j, (hard, obj, sense) in enumerate(instances):
                ep = self.collect_episode(hard, obj, sense)
                stats = self.update(ep)
                stats.update({
                    "iter": it, "instance": j, "rlimit": ep.rlimit,
                    "runtime": ep.runtime, "final_value": ep.final_value,
                    "optimal": ep.optimal,
                    "splits": ep.result_stats.get("splits", 0),
                    "solve_calls": ep.result_stats.get("solve_calls", 0),
                })
                history.append(stats)
                if log:
                    print(
                        f"[iter {it} inst {j}] loss={stats['loss']:.4f} "
                        f"return={stats['mean_return']:.4f} baseline={stats['baseline']:.4f} "
                        f"rlimit={ep.rlimit} splits={stats['splits']} "
                        f"solve_calls={stats['solve_calls']} value={ep.final_value}"
                    )
        return history

    @torch.no_grad()
    def evaluate(self, hard_list, objective, sense: Sense) -> tuple[GOMTResult, int]:
        """确定性评估（argmax 策略），返回 ``(result, rlimit_count)``。"""
        backend = Z3Backend()
        problem = GOMTProblem(hard_list=tuple(hard_list), objective=objective, sense=sense)
        strategy = RLRecordingStrategy(problem, self.policy, self.config, sample=False)
        solver = GOMTSolver(
            problem, backend, strategy,
            GOMTConfig(max_steps=self.config.max_steps, f_sat_mode="plain"),
        )
        result = solver.run()
        return result, backend.rlimit_count

    def make_service(self) -> BranchingPolicyService:
        """把当前策略包装成部署用的 :class:`BranchingPolicyService`。"""
        return BranchingPolicyService(policy=self.policy)

    # ------------------------------------------------------------------ #
    # 结果持久化
    # ------------------------------------------------------------------ #
    def save(self, path, history: Optional[list] = None) -> None:
        """保存策略权重 + RL 训练状态（baseline / 配置 / 训练历史）。"""
        from dataclasses import asdict

        from omt_branching.model.persistence import save_policy

        save_policy(self.policy, path, meta={
            "kind": "rl",
            "baseline": self._baseline,
            "rl_config": asdict(self.config),
            "history": history or [],
        })

    def load(self, path, map_location: Optional[str] = None) -> dict:
        """从磁盘恢复策略权重与 RL 训练状态，返回附带的 meta。"""
        from omt_branching.model.persistence import load_policy_into

        meta = load_policy_into(self.policy, path, map_location or self.config.device)
        self._baseline = float(meta.get("baseline", 0.0))
        return meta


def solve_and_measure(hard_list, objective, sense: Sense, strategy_factory,
                      max_steps: int = 10_000, f_sat_mode: str = "plain") -> dict:
    """用给定策略跑一次完整求解，返回代价/结果指标（含 rlimit）。

    ``strategy_factory(problem) -> BranchingStrategy``，便于对同一实例分别评测
    Neural 与 Baseline 策略。返回 dict 含 ``value / optimal / splits / solve_calls /
    rlimit / steps``。
    """
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=tuple(hard_list), objective=objective, sense=sense)
    strategy = strategy_factory(problem)
    solver = GOMTSolver(problem, backend, strategy,
                        GOMTConfig(max_steps=max_steps, f_sat_mode=f_sat_mode))
    result = solver.run()
    return {
        "value": result.value,
        "optimal": result.optimal,
        "splits": result.stats.get("splits", 0),
        "solve_calls": result.stats.get("solve_calls", 0),
        "steps": result.stats.get("steps", 0),
        "rlimit": backend.rlimit_count,
    }


def _categorical_entropy(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-12)
    return -(p * torch.log(p)).sum()


def _bernoulli_entropy(p_true: torch.Tensor) -> torch.Tensor:
    p = p_true.clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


__all__ = [
    "RLConfig",
    "RLStep",
    "RLEpisode",
    "RLRecordingStrategy",
    "SolverInLoopRLTrainer",
    "solve_and_measure",
]
