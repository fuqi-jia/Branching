from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import generate_hard_lia_dataset, solve_native
from omt_branching.solver.decide_omt import solve_binary, solve_omt_with_decider
from omt_branching.solver.policy_decider import PolicyDecider


def test_vsids_arm_matches_native():
    """公平 VSIDS：挂 propagator，decide 恒 defer（不 next_split）。"""
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    r = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
    assert r["value"] == solve_native(hard, obj, sense)["value"]
    assert r["decisions"] == 0          # 挂 prop 但从未 next_split


def test_check_sat_loop_arm_matches_native():
    """check-sat-loop：同样预处理，但不挂 propagator。"""
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    r = solve_omt_with_decider(hard, obj, sense, attach_propagator=False)
    assert r["value"] == solve_native(hard, obj, sense)["value"]
    assert r["decisions"] is None       # 不挂 propagator


def test_solve_omt_isolated_context_per_call():
    """每次调用在独立 z3.Context 内求解，连续多实例互不干扰。"""
    from omt_branching.solver import generate_bool_lia_dataset

    insts = generate_bool_lia_dataset(3, seed=11, min_vars=4, max_vars=4)
    values = []
    for inst in insts:
        hard, obj, sense = inst.as_tuple()
        r = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
        values.append(r["value"])
    assert all(v is not None for v in values)
    assert len(set(values)) >= 1


def test_learned_arm_matches_native_and_fires():
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    svc = BranchingPolicyService(policy=BranchingPolicy())
    r = solve_omt_with_decider(
        hard, obj, sense,
        decider_factory=lambda a: PolicyDecider(svc, a, refocus_every=50))
    assert r["value"] == solve_native(hard, obj, sense)["value"]   # 正确性：== native
    assert r["decisions"] is not None                     # propagator 在回路里生效
    assert r["rlimit"] > 0


def test_omt_cuts_fed_to_gnn_assertions():
    """线性搜索 better-cut 会经 add_hard 并入 PolicyDecider 建图断言。"""
    from omt_branching.solver.interfaces import Sense

    x = z3.Int("x")
    hard = [x >= 0, x <= 5]
    obj = x
    holders = {}

    def factory(assertions):
        dec = PolicyDecider(
            BranchingPolicyService(policy=BranchingPolicy()),
            assertions,
            refocus_every=50,
        )
        holders["dec"] = dec
        holders["n0"] = len(dec.assertions)
        return dec

    r = solve_omt_with_decider(hard, obj, Sense.MAX, decider_factory=factory)
    assert r["value"] == 5
    assert r["iters"] >= 1
    dec = holders["dec"]
    # 每次成功改进都会 add_hard 一次 cut；最后一轮 UNSAT 前至少有 1 条
    assert len(dec.assertions) > holders["n0"]


def test_parse_get_value_after_objective_expr():
    from omt_branching.solver.decide_omt import _parse_get_value, _parse_z3_statistics

    obj = "(+ (* 4 x0) (* 2 x1))"
    stdout = (
        "sat\n"
        f"((({obj}) 66))\n"
        "(:conflicts 14\n"
        " :decisions 67\n"
        " :rlimit-count 7888)\n"
    )
    assert _parse_get_value(stdout, obj) == 66
    stats = _parse_z3_statistics(stdout)
    assert stats["conflicts"] == 14
    assert stats["decisions"] == 67
    assert stats["rlimit-count"] == 7888


def test_solve_binary_matches_native_on_bool_lia():
    import shutil
    from omt_branching.solver import generate_bool_lia_dataset, instance_to_smt2

    if not shutil.which("z3"):
        pytest.skip("z3 二进制不在 PATH")
    inst = generate_bool_lia_dataset(1, seed=99, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    smt2 = instance_to_smt2(inst)
    ref = solve_binary(inst, smt2=smt2)
    assert ref["status"] == "sat", ref.get("stderr")
    assert ref["value"] is not None, ref.get("stderr")
    assert ref["value"] == solve_native(hard, obj, sense)["value"]
    assert ref.get("returncode") == 0
