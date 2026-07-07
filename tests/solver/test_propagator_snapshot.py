from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator_snapshot import (
    atom_key, collect_atoms, build_bool_snapshot,
)


def test_collect_atoms_and_clause_cooccurrence():
    x = z3.Int("x")
    a, b, c = x >= 5, x <= 2, z3.Bool("c")
    asserts = [z3.Or(a, b), z3.Or(c, z3.Not(a))]
    atoms = collect_atoms(asserts)
    keys = {atom_key(t) for t in atoms}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= keys

    snap, amap = build_bool_snapshot(asserts)
    bkeys = {bv.var_id for bv in snap.bool_vars}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= bkeys
    # 每个顶层 assertion 的原子共现为一个 clause
    assert len(snap.clauses) == 2
    lits0 = {vid for vid, _ in snap.clauses[0].literals}
    assert lits0 == {atom_key(a), atom_key(b)}
    # ¬a 的极性为 False
    pol = dict((vid, pos) for vid, pos in snap.clauses[1].literals)
    assert pol[atom_key(a)] is False
    # 映射能取回 z3 原子
    assert amap[atom_key(a)] is not None


def test_assignment_and_candidates():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    snap, _ = build_bool_snapshot([z3.Or(a, b)], assignment={atom_key(a): True})
    amap = {bv.var_id: bv for bv in snap.bool_vars}
    assert amap[atom_key(a)].assignment is True
    assert amap[atom_key(b)].assignment is None
    assert set(snap.candidate_bool_ids) == {atom_key(a), atom_key(b)}
