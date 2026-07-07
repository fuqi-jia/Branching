"""``SolveBackend`` 的 z3 实现（adapter 层，集中所有 z3 依赖）。

通过 z3 的公开 Python API 提供 GOMT calculus 所需的 ``Solve`` / ``Optimize`` 与
公式代数。**不修改、不依赖系统 z3 二进制**，仅用 pip 安装的 ``z3-solver`` wheel
（与系统 ``/usr/local/bin/z3`` 版本一致，互不影响）。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

import z3

from omt_branching.solver.interfaces import Sense


class Unbounded(Exception):
    """目标在某 branch 上无界（v1 不处理无界优化）。"""


class Z3Backend:
    """以 z3 公开 API 实现的 ``SolveBackend``。

    额外累计 z3 的 **rlimit count**（``Solver.statistics()`` 的 ``"rlimit count"``）
    与 solve 次数：rlimit 是 z3 与硬件/负载无关的确定性工作量计量，用它的增长反映
    “求解耗时”比 wall-clock 更稳定可复现，适合作强化学习的代价信号。
    """

    def __init__(self, eps: float = 1e-9):
        self.eps = eps
        self.rlimit_count = 0    # 累计 rlimit（跨本 backend 的所有 solve/optimize 调用）
        self.solve_calls = 0     # 累计 check() 次数
        # 增量求解：持久的 solver/optimizer（base=φ 断言一次，branch=ψ 经 push/pop 逐节点变化）。
        self._inc_solver = None
        self._inc_solver_base = None
        self._inc_solver_rl = 0          # 该持久 solver 上次见到的累计 rlimit（用于取增量）
        self._inc_opt = None
        self._inc_opt_base = None
        self._inc_opt_obj = None
        self._inc_opt_sense = None
        self._inc_opt_rl = 0

    def reset_stats(self) -> None:
        """清零累计统计（复用同一 backend 跑多次时使用）。"""
        self.rlimit_count = 0
        self.solve_calls = 0

    # ---------------- 求解（一次性：native / 初始态）----------------
    def solve(self, constraint) -> Optional[z3.ModelRef]:
        s = z3.Solver()
        s.add(constraint)
        res = s.check()
        self.solve_calls += 1
        self.rlimit_count += self._rlimit(s)
        return s.model() if res == z3.sat else None

    def optimize(self, constraint, objective, sense: Sense):
        o = z3.Optimize()
        o.add(constraint)
        if sense is Sense.MIN:
            o.minimize(objective)
        else:
            o.maximize(objective)
        res = o.check()
        self.solve_calls += 1
        self.rlimit_count += self._rlimit(o)
        if res != z3.sat:
            return None
        m = o.model()
        # 用最优 model 上的目标取值作为最优值（LIA/LRA 闭最优时即为 bound，
        # 且避免 lower/upper 返回 epsilon/oo 表达式带来的解析问题）。
        return m, self.value(m, objective)

    # ---------------- 增量求解（GOMT 热回路 / strong-branching 标签）----------------
    def solve_branch(self, base, branch) -> Optional[z3.ModelRef]:
        """增量 ``Solve(base∧branch)``：``base`` 固定断言一次，``branch`` 经 push/pop 变化。

        复用持久 solver 保留 z3 已学到的 lemma，避免每个 GOMT 节点从零重解 φ。
        z3 模型取得后即使 pop / 再 check 仍有效（快照语义），可安全存作 incumbent。
        """
        s = self._inc_solver
        if s is None or self._inc_solver_base is not base:
            s = z3.Solver()
            s.add(base)
            self._inc_solver = s
            self._inc_solver_base = base
            self._inc_solver_rl = 0
        s.push()
        s.add(branch)
        res = s.check()
        model = s.model() if res == z3.sat else None
        s.pop()
        self.solve_calls += 1
        self._inc_solver_rl = self._accumulate_delta(s, self._inc_solver_rl)
        return model

    def optimize_branch(self, base, branch, objective, sense: Sense):
        """增量 ``Optimize(base∧branch)``：``base``+目标断言一次，``branch`` 经 push/pop 变化。"""
        o = self._inc_opt
        if (o is None or self._inc_opt_base is not base
                or self._inc_opt_obj is not objective or self._inc_opt_sense is not sense):
            o = z3.Optimize()
            o.add(base)
            if sense is Sense.MIN:
                o.minimize(objective)
            else:
                o.maximize(objective)
            self._inc_opt = o
            self._inc_opt_base = base
            self._inc_opt_obj = objective
            self._inc_opt_sense = sense
            self._inc_opt_rl = 0
        o.push()
        o.add(branch)
        res = o.check()
        if res != z3.sat:
            o.pop()
            self.solve_calls += 1
            self._inc_opt_rl = self._accumulate_delta(o, self._inc_opt_rl)
            return None
        m = o.model()
        val = self.value(m, objective)
        o.pop()
        self.solve_calls += 1
        self._inc_opt_rl = self._accumulate_delta(o, self._inc_opt_rl)
        return m, val

    def _rlimit(self, solver) -> int:
        """读取该 solver 当前累计 rlimit count（缺失返回 0）。"""
        try:
            st = solver.statistics()
            for key in st.keys():
                if key == "rlimit count":
                    return int(st.get_key_value(key))
        except Exception:  # pragma: no cover - 统计缺失不应影响求解
            pass
        return 0

    def _accumulate_delta(self, solver, prev: int) -> int:
        """持久 solver 的 rlimit 是**累计值**，只把本次增量计入 backend 统计；返回新累计值。"""
        cur = self._rlimit(solver)
        self.rlimit_count += max(0, cur - prev)
        return cur

    # ---------------- 取值 ----------------
    def value(self, model, term):
        return self._num(model.eval(term, model_completion=True))

    def is_true(self, model, atom) -> bool:
        return z3.is_true(model.eval(atom, model_completion=True))

    # ---------------- 公式代数 ----------------
    def conjoin(self, *constraints):
        if not constraints:
            return z3.BoolVal(True)
        if len(constraints) == 1:
            return constraints[0]
        return z3.And(*constraints)

    def negate(self, constraint):
        return z3.Not(constraint)

    def better(self, objective, value, sense: Sense):
        num = self._mk_numeral(objective, value)
        return objective < num if sense is Sense.MIN else objective > num

    def top(self):
        return z3.BoolVal(True)

    def le(self, term, bound):
        return term <= self._coerce_bound(term, bound)

    def ge(self, term, bound):
        return term >= self._coerce_bound(term, bound)

    def _coerce_bound(self, term, bound):
        """把 python int/Fraction 的界转成与 ``term`` sort 匹配的 z3 常量。"""
        if isinstance(bound, (int, Fraction)):
            return self._mk_numeral(term, bound)
        return bound

    # ---------------- 内部工具 ----------------
    def _num(self, ref):
        """z3 数值 ref -> python ``int`` / ``Fraction``；无穷/epsilon 抛 ``Unbounded``。"""
        if z3.is_int_value(ref):
            return ref.as_long()
        if z3.is_rational_value(ref):
            return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
        text = str(ref)
        if "oo" in text or "epsilon" in text:
            raise Unbounded(text)
        try:
            return Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:  # pragma: no cover
            raise Unbounded(text) from exc

    def _mk_numeral(self, term, value):
        """按 ``term`` 的 sort 构造与 ``value`` 对应的 z3 数值常量。"""
        if isinstance(value, Fraction) and value.denominator == 1:
            value = value.numerator
        if isinstance(value, int):
            return z3.IntVal(value) if z3.is_int(term) else z3.RealVal(value)
        # Fraction（实数）
        return z3.RealVal(f"{value.numerator}/{value.denominator}")


__all__ = ["Z3Backend", "Unbounded"]
