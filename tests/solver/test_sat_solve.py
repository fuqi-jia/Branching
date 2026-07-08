from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.sat_instances import generate_php
from omt_branching.solver.sat_solve import solve_sat_with_decider
from omt_branching.solver.propagator_snapshot import atom_key


def test_vsids_arm_has_conflicts_and_correct():
    atoms, clauses = generate_php(6)                       # PHP(7,6) UNSAT
    r = solve_sat_with_decider(clauses, atoms, decider_factory=None)
    assert r["result"] == "unsat"                          # 正确性
    assert r["conflicts"] > 100                            # 附 propagator -> 纯 CDCL 大量冲突
    assert r["decisions"] == 0                             # VSIDS 臂我们不覆盖


def test_override_arm_controls_and_correct():
    atoms, clauses = generate_php(6)
    r = solve_sat_with_decider(
        clauses, atoms,
        decider_factory=lambda a: (lambda und, asg: (min(und), True)))
    assert r["result"] == "unsat"
    assert r["decisions"] > 0                               # 我们强制了决策


def test_build_lookahead_examples_sat_learnable():
    import torch
    from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat
    from omt_branching.solver.training_data import build_lookahead_examples_sat
    from omt_branching.model.policy import BranchingPolicy
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig

    torch.manual_seed(0)
    problems = [generate_php(4)] + [generate_rand_3sat(30, 4.26, s) for s in range(6)]
    exs = [e for e in build_lookahead_examples_sat(problems) if e.bool_target_scores]
    assert exs, "应有带 bool 标签的样本"
    policy = BranchingPolicy()
    h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=12)
    assert "branch" in h[0]
    assert h[-1]["branch"] < h[0]["branch"]        # 子句图=特征 -> look-ahead 可学
