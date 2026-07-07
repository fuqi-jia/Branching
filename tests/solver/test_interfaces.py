from __future__ import annotations

from omt_branching.solver.interfaces import (
    Sense, SplitDecision, GOMTState, SolveBackend, BranchingStrategy,
)


def test_sense_members():
    assert {s.name for s in Sense} == {"MIN", "MAX"}


def test_split_decision_helpers():
    s = SplitDecision.split(["a", "b"])
    assert s.kind == "split" and s.subformulas == ["a", "b"]
    r = SplitDecision.resolve()
    assert r.kind == "resolve" and r.subformulas == []


def test_gomt_state_top_and_saturated():
    st = GOMTState(incumbent=None, delta="D", tau=["t0", "t1"],
                   objective="t", sense=Sense.MIN, hard="phi")
    assert st.top == "t0"
    assert st.saturated is False
    st.tau = []
    assert st.saturated is True


def test_protocols_are_runtime_checkable():
    class B:
        def solve(self, c): ...
        def optimize(self, c, o, s): ...
        def solve_branch(self, base, br): ...
        def optimize_branch(self, base, br, o, s): ...
        def value(self, m, t): ...
        def is_true(self, m, a): ...
        def conjoin(self, *c): ...
        def negate(self, c): ...
        def better(self, o, v, s): ...
        def top(self): ...
        def le(self, t, b): ...
        def ge(self, t, b): ...
    assert isinstance(B(), SolveBackend)

    class S:
        def propose(self, state, backend): ...
    assert isinstance(S(), BranchingStrategy)
