"""分支敏感实例生成器冒烟测试。"""
from __future__ import annotations

import random

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.instance_gen import (
    generate_branch_focus_lia_dataset,
    generate_branch_focus_lia_instance,
)


def test_branch_focus_sat_and_oracle_priority():
    rng = random.Random(0)
    bundle = generate_branch_focus_lia_instance(
        "bfocus_t0", rng, n_vars=7, n_modes=3, n_hard_disj=20, n_distractors=24
    )
    inst = bundle.instance
    assert inst.family == "branch_focus"
    assert len(bundle.oracle_priority) >= 1
    s = z3.Solver()
    s.add(*inst.hard)
    assert s.check() == z3.sat
    # 最优值应可达（MAX）
    o = z3.Optimize()
    o.add(*inst.hard)
    o.maximize(inst.objective)
    assert o.check() == z3.sat


def test_branch_focus_dataset_reproducible():
    a = generate_branch_focus_lia_dataset(3, seed=7, min_vars=6, max_vars=7)
    b = generate_branch_focus_lia_dataset(3, seed=7, min_vars=6, max_vars=7)
    assert [x.instance.instance_id for x in a] == [x.instance.instance_id for x in b]
    assert [len(x.instance.hard) for x in a] == [len(x.instance.hard) for x in b]
