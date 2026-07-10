"""OMT = 单 z3.Solver 线性搜索回路（Solve + Better-cut，直到 UNSAT），可挂
LearnedDecidePropagator 接管内部布尔决策。z3.Optimize 不支持 propagator，故必须走此回路。

三臂对比：decider_factory=None -> VSIDS 臂；给 PolicyDecider -> learned 臂；native 见 solve_native。
"""

from __future__ import annotations

from fractions import Fraction
import math

import z3

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import collect_atoms


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


def _num(ref):
    if z3.is_int_value(ref):
        return ref.as_long()
    if z3.is_rational_value(ref):
        return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
    return Fraction(str(ref))


def solve_omt_with_decider(
    hard, objective, sense: Sense, decider_factory=None, max_iters: int = 100000
) -> dict:
    s = z3.Solver()
    prop = None
    if decider_factory is not None:
        atoms = collect_atoms(list(hard))
        decider = decider_factory(list(hard))
        prop = LearnedDecidePropagator(s, atoms, decider)
    s.add(*hard)

    if s.check() != z3.sat:
        raise ValueError("硬约束不可满足")
    m = s.model()
    best_val = m.eval(objective, model_completion=True)
    records = [(best_val, _stat(s, "rlimit count"))]

    iters = 0
    for iters in range(1, max_iters + 1):
        cut = objective > best_val if sense is Sense.MAX else objective < best_val
        s.add(cut)
        if s.check() != z3.sat:
            break
        m = s.model()
        best_val = m.eval(objective, model_completion=True)
        records.append((best_val, _stat(s, "rlimit count") - records[-1][1]))

    return {
        "value": _num(best_val),
        "rlimit": _stat(s, "rlimit count"),
        "weighted rlimit": sum(
            [math.log2(float(best_val) / local + 1) * cost for local, cost in records]
        ),
        "conflicts": _stat(s, "conflicts"),
        "decisions": (prop.n_decisions if prop is not None else None),
        "iters": iters,
    }


__all__ = ["solve_omt_with_decider"]
