from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import atom_key


def _sat_instance():
    xs = [z3.Bool(f"b{i}") for i in range(12)]
    clauses = [z3.Or(xs[i], z3.Not(xs[(i + 1) % 12]), xs[(i + 2) % 12]) for i in range(12)]
    return xs, clauses


def _solve(decider):
    xs, clauses = _sat_instance()
    s = z3.Solver()
    p = LearnedDecidePropagator(s, xs, decider)
    s.add(*clauses)
    return s.check(), p.n_decisions


def test_propagator_controls_decisions_and_preserves_correctness():
    idx = lambda k: int(k[1:])
    resA, nA = _solve(lambda und, asg: (min(und, key=idx), True))
    resB, nB = _solve(lambda und, asg: (max(und, key=idx), True))
    assert resA == resB == z3.sat        # 正确性不变
    assert nA > 0 and nB > 0             # 两个 decider 都真的强制了决策


def test_none_decider_falls_back():
    resN, nN = _solve(lambda und, asg: None)   # 永远 None = 退回 VSIDS
    assert resN == z3.sat
    assert nN == 0                        # 我们没强制任何决策
