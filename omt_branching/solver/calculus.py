"""GOMT calculus 引擎（理论无关，不依赖 z3）。

实现 ``GOMT.pdf`` 第 4 节的状态机 ``Ψ = ⟨I, Δ, τ⟩`` 与三条派生规则：

- **F-Split**：把 ``Top(τ)`` 换成策略给出的有序子公式（唯一的启发式自由点）。
- **F-Sat**：``Solve(φ∧ψ)`` 有解 ``I'`` 时切换到 ``I'`` 并收紧 ``Δ := Δ∧Better(I')``，
  重置 ``τ := (Δ)``。
- **F-Close**：``Solve(φ∧ψ) = ⊥`` 时排除该 branch：``Δ := Δ∧¬ψ``，``τ := Pop(τ)``。

是否 F-Split 由 ``BranchingStrategy`` 决定；F-Sat 与 F-Close 由 ``Solve`` 结果机械
触发。饱和（``τ=∅``）时 ``incumbent`` 为最优解（Theorem 1）。
"""

from __future__ import annotations

from dataclasses import dataclass

from omt_branching.solver.interfaces import (
    BranchingStrategy, GOMTState, Model, SolveBackend, Value,
)
from omt_branching.solver.problem import GOMTProblem


@dataclass(frozen=True)
class GOMTConfig:
    """求解配置。

    - ``max_steps``：派生步数上限（anytime 后备，超限返回当前最优并标记非最优）。
    - ``f_sat_mode``：``"plain"`` 用 ``Solve``（Neural F-Split 真正驱动搜索）；
      ``"hybrid"`` 用 ``Optimize`` 作 leaf 加速器（GOMT §4.2 / Thm 4，"更强的 Solve"）。
    - ``check_invariants``：预留位（v1 为 no-op，以保持本模块不 import z3）。F-Split
      子公式由 ``ψ∧a`` / ``ψ∧¬a`` 构造，前提 ``φ⊨ψ⇔⋁ψ_j`` 构造上即成立。
    """

    max_steps: int = 10_000
    f_sat_mode: str = "plain"
    check_invariants: bool = False


@dataclass
class GOMTResult:
    """求解结果。``optimal`` 为 ``True`` 表示到达饱和态（最优）。"""

    model: Model
    value: Value
    optimal: bool
    stats: dict


class GOMTSolver:
    """按 ``BranchingStrategy`` 驱动的 GOMT calculus 求解器。"""

    def __init__(self, problem: GOMTProblem, backend: SolveBackend,
                 strategy: BranchingStrategy, config: GOMTConfig = GOMTConfig()):
        self.problem = problem
        self.backend = backend
        self.strategy = strategy
        self.config = config

    def run(self) -> GOMTResult:
        """跑到饱和或 ``max_steps``，返回最优（或当前最优）解。"""
        b = self.backend
        st = self.problem.initial_state(b)
        obj, sense = st.objective, st.sense

        while not st.saturated and st.step < self.config.max_steps:
            decision = self.strategy.propose(st, b)

            if decision.kind == "split" and decision.subformulas:
                # F-Split：用有序子公式替换 Top(τ)。
                st.tau = list(decision.subformulas) + st.tau[1:]
                st.stats["splits"] += 1
                st.step += 1
                continue

            # resolve（含空 split 的退化）：对当前 branch 增量调用 Solve / Optimize
            # （base=φ 断言一次，branch=ψ 经 push/pop 变化，保留 z3 lemma 大幅提速）。
            psi = st.top
            st.stats["solve_calls"] += 1
            if self.config.f_sat_mode == "hybrid":
                res = b.optimize_branch(st.hard, psi, obj, sense)
                model = res[0] if res is not None else None
                value = res[1] if res is not None else None
            else:
                model = b.solve_branch(st.hard, psi)
                value = b.value(model, obj) if model is not None else None

            if model is not None:
                # F-Sat：切换到更优解，收紧 Δ，重置 τ 与本轮 split 预算。
                st.incumbent = model
                st.delta = b.conjoin(st.delta, b.better(obj, value, sense))
                st.tau = [st.delta]
                st.stats["sats"] += 1
                st.stats["branch_depth"] = 0
            else:
                # F-Close：排除该 branch。
                st.delta = b.conjoin(st.delta, b.negate(psi))
                st.tau = st.tau[1:]
                st.stats["closes"] += 1
            st.step += 1

        st.stats["steps"] = st.step
        optimal = st.saturated
        value = b.value(st.incumbent, obj)
        return GOMTResult(model=st.incumbent, value=value, optimal=optimal,
                          stats=dict(st.stats))


__all__ = ["GOMTSolver", "GOMTConfig", "GOMTResult"]
