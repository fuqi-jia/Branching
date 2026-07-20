from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.propagator_snapshot import atom_key
from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat
from omt_branching.solver.vsids_trace import (
    VSIDSTraceConfig, build_vsids_examples_sat, collect_vsids_trajectory,
)


def test_collect_vsids_trajectory_records_registered_atoms():
    """PHP(UNSAT) 保证大量搜索;每条记录=VSIDS 落在未定注册原子上的决策。"""
    atoms, clauses = generate_php(4)               # PHP(5,4)，UNSAT
    records, ref_conflicts, info = collect_vsids_trajectory(clauses, atoms)
    assert info["result"] == "unsat"
    assert ref_conflicts >= 1                      # 有冲突 → 可作 RL 归一化参考
    assert len(records) >= 1
    keys = {atom_key(a) for a in atoms}
    for assignment, chosen_key, phase in records:
        assert chosen_key in keys                  # 只记已注册原子(跳过辅助变量)
        assert chosen_key not in assignment        # 决策落在未定原子
        assert isinstance(phase, bool)


def test_collect_stride_and_cap():
    atoms, clauses = generate_php(4)
    full, _, _ = collect_vsids_trajectory(clauses, atoms)
    capped, _, _ = collect_vsids_trajectory(clauses, atoms, VSIDSTraceConfig(max_examples=3))
    assert len(capped) <= 3
    strided, _, info = collect_vsids_trajectory(clauses, atoms, VSIDSTraceConfig(stride=3))
    # stride 下采样 → 不多于全量
    assert len(strided) <= len(full)


def test_build_vsids_examples_sat_labels_are_onehot():
    """标签近似 one-hot:被选原子记 weight(唯一最高分)、其余未定原子 0;相位一条。"""
    exs = build_vsids_examples_sat([generate_php(4)], VSIDSTraceConfig(max_examples=15))
    assert len(exs) >= 1
    w = VSIDSTraceConfig().weight
    for ex in exs:
        assert ex.bool_target_scores
        assert max(ex.bool_target_scores.values()) == w
        assert sum(1 for v in ex.bool_target_scores.values() if v == w) == 1
        assert len(ex.phase_targets) == 1


def test_vsids_examples_are_learnable():
    """learnability guard:VSIDS 轨迹标签在当前 snapshot 特征下**可学**(BC 损失下降) ——
    这是 warm-start 的前提;若冻结则说明缺冲突动态特征(需另加)。"""
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig

    torch.manual_seed(0)
    probs = [generate_php(4), generate_rand_3sat(20, 4.26, seed=1)]
    exs = [e for e in build_vsids_examples_sat(probs, VSIDSTraceConfig(max_examples=20))
           if e.bool_target_scores]
    assert len(exs) >= 4
    h = ImitationTrainer(BranchingPolicy(), TrainConfig(lr=5e-3)).fit(exs, epochs=40)
    assert h[-1]["branch"] < h[0]["branch"]
