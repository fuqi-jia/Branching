"""桥接核心接口与数据类型（不依赖 z3）。

本模块定义把 OMT 求解器后端与分支策略解耦的两个 Protocol，以及 GOMT calculus
的状态对象。``calculus.py`` / ``problem.py`` / 各 strategy 只通过这些抽象工作，
对 ``Model`` / ``Term`` / ``Constraint`` 等句柄视为**不透明**（仅经 ``SolveBackend``
操作），从而保证 z3 被完全限制在 adapter 层（``z3_backend.py`` / ``extractor.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

# 不透明句柄类型别名：calculus / strategy 不窥探其内部，仅经 backend 操作。
Model = Any
Term = Any
Constraint = Any
Atom = Any
Value = Any


class Sense(Enum):
    """优化方向。"""

    MIN = "min"
    MAX = "max"


@dataclass
class SplitDecision:
    """策略对 ``Top(τ)`` 的处理建议。

    - ``kind == "split"``：执行 F-Split，把 ``Top(τ)`` 换成有序的 ``subformulas``
      （列表首项最先被探索，对应 GOMT §4.2 的 search order）。
    - ``kind == "resolve"``：不再细分，直接对当前 branch 调用 ``Solve`` /
      ``Optimize``，由其结果决定走 F-Sat 还是 F-Close。
    """

    kind: str
    subformulas: list[Constraint] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @classmethod
    def split(cls, formulas, **info) -> "SplitDecision":
        return cls(kind="split", subformulas=list(formulas), info=dict(info))

    @classmethod
    def resolve(cls, **info) -> "SplitDecision":
        return cls(kind="resolve", subformulas=[], info=dict(info))


@dataclass
class GOMTState:
    """GOMT calculus 状态 ``Ψ = ⟨I, Δ, τ⟩``（含求解上下文与统计）。

    - ``incumbent``：迄今最优解 ``I``（model）。
    - ``delta``：剩余搜索空间公式 ``Δ``。
    - ``tau``：把 ``Δ`` 划分成 branch 的公式序列；不变式 ``φ ⊨ (⋁τ_i ⇔ Δ)``。
    - ``objective`` / ``sense`` / ``hard``：问题的目标项 ``t`` / 方向 / 硬约束 ``φ``。
    - ``step`` / ``stats``：派生步数与统计（splits/sats/closes/...）。
    """

    incumbent: Optional[Model]
    delta: Constraint
    tau: list[Constraint]
    objective: Term
    sense: Sense
    hard: Constraint
    step: int = 0
    stats: dict = field(default_factory=dict)

    @property
    def top(self) -> Constraint:
        """``Top(τ)``：当前待处理 branch。"""
        return self.tau[0]

    @property
    def saturated(self) -> bool:
        """``τ = ∅`` 即饱和；此时 ``incumbent`` 为最优解（Thm 1）。"""
        return not self.tau


@runtime_checkable
class SolveBackend(Protocol):
    """OMT 后端契约：calculus / strategy 经它做理论求解与公式代数。

    z3 是其唯一实现（``Z3Backend``）。把公式代数（conjoin/negate/better/le/ge）
    也收进接口，使 calculus 与策略保持理论无关、不直接 import z3。
    """

    def solve(self, constraint: Constraint) -> Optional[Model]:
        """SMT 判定：SAT 返回 model，UNSAT 返回 ``None``。"""
        ...

    def optimize(self, constraint: Constraint, objective: Term,
                 sense: Sense) -> Optional[tuple[Model, Value]]:
        """更强的 Solve：返回 branch 内对 ``objective`` 最优的 ``(model, value)``。"""
        ...

    def solve_branch(self, base: Constraint, branch: Constraint) -> Optional[Model]:
        """增量版 ``solve``：``base`` 固定、``branch`` 逐节点变化，语义同 ``solve(base∧branch)``。"""
        ...

    def optimize_branch(self, base: Constraint, branch: Constraint, objective: Term,
                        sense: Sense) -> Optional[tuple[Model, Value]]:
        """增量版 ``optimize``：语义同 ``optimize(base∧branch, objective, sense)``。"""
        ...

    def value(self, model: Model, term: Term) -> Value:
        """求 ``term`` 在 ``model`` 下的取值。"""
        ...

    def is_true(self, model: Model, atom: Atom) -> bool:
        """``atom`` 在 ``model`` 下是否为真。"""
        ...

    def conjoin(self, *constraints: Constraint) -> Constraint:
        """合取；零参数返回逻辑真。"""
        ...

    def negate(self, constraint: Constraint) -> Constraint:
        """取非。"""
        ...

    def better(self, objective: Term, value: Value, sense: Sense) -> Constraint:
        """``Better(I)``：MIN 时 ``objective < value``，MAX 时 ``objective > value``。"""
        ...

    def top(self) -> Constraint:
        """逻辑真常量。"""
        ...

    def le(self, term: Term, bound) -> Atom:
        """``term <= bound``。"""
        ...

    def ge(self, term: Term, bound) -> Atom:
        """``term >= bound``。"""
        ...


@runtime_checkable
class BranchingStrategy(Protocol):
    """calculus 咨询的分支决策者：唯一的启发式自由点（F-Split）。"""

    def propose(self, state: GOMTState, backend: SolveBackend) -> SplitDecision:
        """给出对 ``Top(τ)`` 的处理建议（split 或 resolve）。"""
        ...


__all__ = [
    "Sense",
    "SplitDecision",
    "GOMTState",
    "SolveBackend",
    "BranchingStrategy",
    "Model",
    "Term",
    "Constraint",
    "Atom",
    "Value",
]
