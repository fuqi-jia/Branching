from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.z3_backend import Z3Backend
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.extractor import Z3SnapshotExtractor


def test_extract_counts_and_coeffs():
    b = Z3Backend()
    x, y = z3.Int("x"), z3.Int("y")
    hard = (x >= 0, x <= 10, y >= 0, y <= 10, 2 * x + 3 * y <= 12)
    prob = GOMTProblem(hard_list=hard, objective=x + y, sense=Sense.MAX)
    st = prob.initial_state(b)
    ex = Z3SnapshotExtractor(prob).extract(st, b)
    snap = ex.snapshot
    # two numeric vars
    ids = {nv.num_var_id for nv in snap.numeric_vars}
    assert len(ids) == 2
    # the 2x+3y<=12 atom has coeffs {x:2, y:3}, rhs 12
    atoms = [a for a in snap.theory_atoms
             if set(a.var_coeffs.values()) == {2.0, 3.0}]
    assert len(atoms) == 1 and atoms[0].rhs == 12.0
    # objective coeffs both 1
    assert snap.objective.var_coeffs and all(
        c == 1.0 for c in snap.objective.var_coeffs.values())
    assert snap.objective.sense_is_min is False
    # handles map back
    assert len(ex.numeric_handles) == 2


def test_extract_robust_on_incumbent_values():
    b = Z3Backend()
    x = z3.Int("x")
    prob = GOMTProblem(hard_list=(x >= 0, x <= 5), objective=x, sense=Sense.MIN)
    st = prob.initial_state(b)
    ex = Z3SnapshotExtractor(prob).extract(st, b)
    h = next(iter(ex.numeric_handles.values()))
    assert h.current_value is not None
    # candidate sets populated for the policy
    assert ex.snapshot.candidate_numeric_ids


def _atom(kind, coeffs, rhs):
    from omt_branching.input.solver_state import TheoryAtomInfo
    from omt_branching.interfaces import AtomKind
    return TheoryAtomInfo(atom_id="a", bool_var_id="b", kind=getattr(AtomKind, kind),
                          var_coeffs=coeffs, rhs=float(rhs))


def test_lp_relaxation_bounded_gives_optimum():
    """有界 LP：0<=x<=10 maximize x -> lp_value x=10。"""
    x = z3.Int("x")
    ext = Z3SnapshotExtractor(GOMTProblem(hard_list=(x >= 0, x <= 10), objective=x, sense=Sense.MAX))
    lp = ext._lp_relaxation([_atom("GE", {"x": 1.0}, 0.0), _atom("LE", {"x": 1.0}, 10.0)], {"x": 1.0})
    assert lp.get("x") == 10.0


def test_lp_relaxation_unbounded_returns_empty():
    """无界 LP（只有 x>=0，maximize x 无上界）：不得产出任意 model 值当 lp_value。"""
    x = z3.Int("x")
    ext = Z3SnapshotExtractor(GOMTProblem(hard_list=(x >= 0,), objective=x, sense=Sense.MAX))
    lp = ext._lp_relaxation([_atom("GE", {"x": 1.0}, 0.0)], {"x": 1.0})
    assert lp == {}


def test_lp_relaxation_no_objective_returns_empty():
    """无目标系数时 lp_value 无意义，返回空。"""
    x = z3.Int("x")
    ext = Z3SnapshotExtractor(GOMTProblem(hard_list=(x >= 0, x <= 5), objective=x, sense=Sense.MAX))
    assert ext._lp_relaxation([_atom("GE", {"x": 1.0}, 0.0), _atom("LE", {"x": 1.0}, 5.0)], {}) == {}
