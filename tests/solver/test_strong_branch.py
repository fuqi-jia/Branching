from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import OMTInstance, Sense
from omt_branching.solver.strong_branch import (
    StrongBranchConfig, oracle_bool_choice, strong_branch_scores, _root_extraction,
)


def _sep_instance():
    """maximize x, 0<=x<=10；Or(x<=3, x>=7) 的两原子分离目标(分数~7)，
    Or(x<=50, x>=-5) 与盒约束原子被蕴含(分数 0)。"""
    x = z3.Real("x")
    hard = [x >= 0, x <= 10, z3.Or(x <= 3, x >= 7), z3.Or(x <= 50, x >= -5)]
    return OMTInstance(instance_id="t", variables=[x], hard=hard, objective=x,
                       sense=Sense.MAX, obj_coeffs={"x": 1.0}, theory="LRA")


def test_separating_atom_outranks_entailed():
    inst = _sep_instance()
    extraction, phi, obj, sense, backend = _root_extraction(inst)
    scores, phases = strong_branch_scores(extraction, phi, obj, sense, backend)
    assert scores, "应有候选原子打分"
    # 分离原子分数约为 7（x∈[0,3] 的 max=3 vs x∈[7,10] 的 max=10）
    assert max(scores.values()) == pytest.approx(7.0, abs=1e-6)
    # 存在被蕴含原子（一侧 UNSAT）得 0 分
    assert min(scores.values()) == pytest.approx(0.0, abs=1e-9)


def test_oracle_bool_choice_returns_separating_atom():
    inst = _sep_instance()
    bid = oracle_bool_choice(inst)
    assert bid is not None
    # 该 bid 对应的原子确为分离原子（其目标分离度为最大）
    extraction, phi, obj, sense, backend = _root_extraction(inst)
    scores, _ = strong_branch_scores(extraction, phi, obj, sense, backend)
    assert scores[bid] == max(scores.values())


def test_oracle_none_when_no_separation():
    """无布尔结构（只有盒约束，所有原子被蕴含）时返回 None。"""
    x = z3.Real("x")
    hard = [x >= 0, x <= 10]
    inst = OMTInstance(instance_id="flat", variables=[x], hard=hard, objective=x,
                       sense=Sense.MAX, obj_coeffs={"x": 1.0}, theory="LRA")
    assert oracle_bool_choice(inst) is None


from omt_branching.solver.strong_branch import (
    strong_branch_numeric_scores, oracle_numeric_choice_sb,
)


def _lia_obj_instance():
    """maximize x；x,y∈[0,10]，无耦合。x 影响目标(分离~10)，y 不影响(分离 0)。"""
    x, y = z3.Int("x"), z3.Int("y")
    hard = [x >= 0, x <= 10, y >= 0, y <= 10]
    return OMTInstance(instance_id="li", variables=[x, y], hard=hard, objective=x,
                       sense=Sense.MAX, obj_coeffs={"x": 1.0, "y": 0.0}, theory="LIA")


def test_numeric_expert_ranks_objective_var_above_irrelevant():
    inst = _lia_obj_instance()
    extraction, phi, obj, sense, backend = _root_extraction(inst)
    scores, dirs = strong_branch_numeric_scores(extraction, phi, obj, sense, backend)
    assert scores.get("x", 0) > scores.get("y", 0)     # x 分离目标，y 不
    # 选 x 后 MAX 应先探高侧（x 越大越优）
    assert dirs["x"] is True


def test_oracle_numeric_choice_sb_picks_objective_var():
    inst = _lia_obj_instance()
    assert oracle_numeric_choice_sb(inst) == "x"
