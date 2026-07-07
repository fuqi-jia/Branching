from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import generate_lra_dataset
from omt_branching.solver.strong_branch import StrongBranchConfig
from omt_branching.solver.training_data import (
    build_imitation_example, bool_branch_hit,
)
from omt_branching.model.policy import BranchingPolicy


def _lra_inst():
    # 取一个含布尔结构、能产生分离原子的 LRA 实例
    ds = generate_lra_dataset(20, seed=3, min_vars=4, max_vars=5)
    for inst in ds:
        ex = build_imitation_example(inst)
        if ex is not None and ex.bool_target_scores:
            return inst
    pytest.skip("未生成含分离原子的实例")


def test_imitation_example_has_bool_labels():
    inst = _lra_inst()
    ex = build_imitation_example(inst)
    assert ex is not None
    assert ex.bool_target_scores, "bool head 应有非空 imitation 标签"
    # 标签键是合法的图内 BOOL_VAR 局部索引
    from omt_branching.interfaces import NodeType
    n_bool = ex.graph.num_nodes(NodeType.BOOL_VAR)
    assert all(0 <= k < n_bool for k in ex.bool_target_scores)


def test_bool_branch_hit_returns_bool():
    inst = _lra_inst()
    policy = BranchingPolicy()
    r = bool_branch_hit(policy, inst, StrongBranchConfig())
    assert r in (True, False)


def test_imitation_trains_bool_head_loss_decreases():
    from omt_branching.solver.training_data import build_imitation_examples
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig
    import torch

    torch.manual_seed(0)
    ds = generate_lra_dataset(24, seed=7, min_vars=4, max_vars=5)
    examples = [e for e in build_imitation_examples(ds) if e.bool_target_scores]
    assert examples, "应有带 bool 标签的样本"

    policy = BranchingPolicy()
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    hist = trainer.fit(examples, epochs=8)
    # bool head 有被训练：'branch' 损失项出现，且末轮 < 首轮
    assert "branch" in hist[0], "bool branching 损失项应存在（bool head 有梯度）"
    assert hist[-1]["branch"] < hist[0]["branch"]


def test_imitation_numeric_strong_labels():
    from omt_branching.solver.instance_gen import generate_hard_lia_dataset
    from omt_branching.solver.training_data import build_imitation_example

    ds = generate_hard_lia_dataset(8, seed=2, min_vars=5, max_vars=6)
    ex = None
    for inst in ds:
        e = build_imitation_example(inst, numeric_expert="strong")
        if e is not None and e.int_target_scores:
            ex = e
            break
    assert ex is not None, "应有含数值 strong 标签的样本"
    from omt_branching.interfaces import NodeType
    n_num = ex.graph.num_nodes(NodeType.NUMERIC_VAR)
    assert all(0 <= k < n_num for k in ex.int_target_scores)
    # 数值方向标签也应存在
    assert ex.int_dir_targets
