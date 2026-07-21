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


def test_add_hard_appends_and_forces_refocus():
    x = z3.Int("x")
    asserts = [x >= 0, x <= 10, z3.Or(x >= 5, x <= 2)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=100)
    n0 = len(dec.assertions)
    cut = x > 3
    dec([atom_key(x >= 5)], {})   # 消耗首次 refocus
    assert dec._since == 1
    dec.add_hard(cut)
    assert len(dec.assertions) == n0 + 1
    assert dec.assertions[-1] is cut
    assert dec._since == dec.refocus_every
    assert dec._pri is None


def test_add_hard_refreshes_root_fixed_for_graph():
    """跨 cut：add_hard 后 _root_fixed 非空，refocus 建图剪掉已定候选。"""
    from omt_branching.input.graph_builder import GraphBuilder
    from omt_branching.interfaces import EdgeType
    from omt_branching.solver.propagator_snapshot import (
        build_bool_snapshot,
        merge_root_assignment,
    )

    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=100)
    assert dec._root_fixed == {}
    dec.add_hard(x >= 6)
    assert dec._root_fixed.get(atom_key(a)) is True
    assert dec._root_fixed.get(atom_key(b)) is False

    seen = {}
    orig = dec._refocus

    def wrap(asg):
        merged = merge_root_assignment(dec._root_fixed, asg)
        seen["merged"] = dict(merged)
        snap, _ = build_bool_snapshot(dec.assertions, assignment=merged)
        seen["cand"] = list(snap.candidate_bool_ids or [])
        g = GraphBuilder().build(snap)
        seen["lit_edges"] = g.num_edges(EdgeType.LITERAL_IN_CLAUSE)
        return orig(asg)

    dec._refocus = wrap
    dec([atom_key(a), atom_key(b)], {})
    assert seen["merged"].get(atom_key(a)) is True
    assert atom_key(a) not in seen["cand"]
    assert atom_key(b) not in seen["cand"]
    assert seen["lit_edges"] == 0


def test_on_backtrack_forces_immediate_refocus():
    x = z3.Int("x")
    asserts = [x >= 0, x <= 10, z3.Or(x >= 5, x <= 2)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=100)
    calls = {"n": 0}
    orig = dec._refocus

    def counting(asg):
        calls["n"] += 1
        return orig(asg)

    dec._refocus = counting
    dec([atom_key(x >= 5)], {})  # 首次 refocus
    assert calls["n"] == 1
    assert dec._since == 1
    dec.on_backtrack(1)
    assert dec._since == dec.refocus_every
    dec([atom_key(x >= 5)], {})  # 回退后立刻再 refocus
    assert calls["n"] == 2


def test_on_backtrack_can_be_disabled():
    x = z3.Int("x")
    asserts = [x >= 0, x <= 10, z3.Or(x >= 5, x <= 2)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(
        svc, asserts, refocus_every=100, refocus_on_backtrack=False
    )
    dec([atom_key(x >= 5)], {})
    assert dec._since == 1
    dec.on_backtrack(2)
    assert dec._since == 1
