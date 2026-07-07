from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver.policy_decider import PolicyDecider
from omt_branching.solver.propagator_snapshot import atom_key


def test_policy_decider_returns_valid_or_fallback():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=100)
    und = [atom_key(a), atom_key(b)]
    choice = dec(und, {})
    # 要么退回(None)，要么返回一个合法的未定原子键 + bool 相位
    assert choice is None or (choice[0] in und and isinstance(choice[1], bool))


def test_refocus_cadence():
    x = z3.Int("x")
    asserts = [x >= 0, x <= 10, z3.Or(x >= 5, x <= 2)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=3)
    calls = {"n": 0}
    orig = dec._refocus

    def counting(asg):
        calls["n"] += 1
        return orig(asg)

    dec._refocus = counting
    for _ in range(7):
        dec([atom_key(x >= 5)], {})
    assert calls["n"] == 3     # refocus 于第 1,4,7 次调用
