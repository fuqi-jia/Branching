from __future__ import annotations

import statistics

import pytest

z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver import RLConfig, RLStep
from omt_branching.solver.rl import SolverInLoopRLTrainer
from omt_branching.solver.interfaces import Sense


def _trainer(**kw):
    return SolverInLoopRLTrainer(BranchingPolicy(), RLConfig(**kw))


def test_shaped_rewards_normalized_by_span():
    tr = _trainer(reward_scale=1.0)
    steps = [RLStep(graph=None, head="bool", chosen_local=0, direction=True, value_at_decision=0.0),
             RLStep(graph=None, head="bool", chosen_local=0, direction=True, value_at_decision=5.0)]
    r = tr._shaped_rewards(steps, final_val=10.0, sense=Sense.MAX)
    assert r[0] == pytest.approx(0.5) and r[1] == pytest.approx(0.5)


def test_shaped_rewards_zero_span_is_zero():
    tr = _trainer()
    steps = [RLStep(graph=None, head="bool", chosen_local=0, direction=True, value_at_decision=3.0)]
    r = tr._shaped_rewards(steps, final_val=3.0, sense=Sense.MAX)
    assert r == [0.0]


def test_per_instance_baseline_tracks_each_key_and_lowers_variance():
    tr = _trainer(baseline_momentum=0.5)
    for _ in range(30):
        tr._update_baseline_for("a", -1.0)
        tr._update_baseline_for("b", -100.0)
    ba, bb = tr._baseline_for("a"), tr._baseline_for("b")
    assert ba == pytest.approx(-1.0, abs=1e-3)
    assert bb == pytest.approx(-100.0, abs=1e-3)
    # per-instance 优势方差 << 单一全局 baseline 优势方差
    adv_per = [(-1.0) - ba, (-100.0) - bb]
    g = (-1.0 - 100.0) / 2
    adv_global = [(-1.0) - g, (-100.0) - g]
    assert statistics.pvariance(adv_per) < statistics.pvariance(adv_global)


def test_baseline_for_unknown_key_falls_back_to_global():
    tr = _trainer()
    tr._baseline = -7.0
    assert tr._baseline_for("never-seen") == -7.0
    assert tr._baseline_for(None) == -7.0
