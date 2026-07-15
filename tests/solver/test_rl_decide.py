from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.rl_decide import SamplingPolicyDecider
from omt_branching.solver.propagator_snapshot import atom_key


def test_sampling_decider_records_steps_and_valid_choice():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    policy = BranchingPolicy()
    defer = torch.zeros(())
    dec = SamplingPolicyDecider(policy, defer, asserts, refocus_every=100, sample=True)
    und = [atom_key(a), atom_key(b)]
    torch.manual_seed(0)
    outs = [dec(und, {}) for _ in range(5)]
    # 每次返回 None(defer) 或 合法未定原子+bool
    assert all(o is None or (o[0] in und and isinstance(o[1], bool)) for o in outs)
    assert len(dec.steps) == 5            # 记录了 5 步
    g, ls, idx = dec.steps[0]
    assert 0 <= idx <= len(ls)            # idx=0=defer, 1..len=原子


def test_decide_rl_collect_update_runs():
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig
    import math

    inst = generate_bool_lia_dataset(1, seed=3, min_vars=5, max_vars=5)[0]
    hard, obj, sense = inst.as_tuple()
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=30))
    steps, reward, res = tr.collect(hard, obj, sense)
    assert res["value"] is not None and res["rlimit"] > 0
    assert math.isfinite(reward)
    stats = tr.update(steps, reward, key=0)
    assert math.isfinite(stats["loss"])
    assert 0 in tr._baselines             # baseline 记录


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_decide_rl_cuda_graph_on_device():
    """RL collect/update 时图特征须与策略同设备（否则 addmm 报 device mismatch）。"""
    import math

    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    inst = generate_bool_lia_dataset(1, seed=7, min_vars=5, max_vars=5)[0]
    hard, obj, sense = inst.as_tuple()
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=20, device="cuda"))
    steps, reward, _ = tr.collect(hard, obj, sense)
    stats = tr.update(steps, reward, key=0)
    assert next(tr.policy.parameters()).device.type == "cuda"
    assert math.isfinite(stats["loss"])


def test_decide_rl_sat_collect_update():
    import math
    from omt_branching.solver.sat_instances import generate_rand_3sat
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    atoms, clauses = generate_rand_3sat(30, 4.26, seed=1)
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=40))
    steps, reward, res = tr.collect_sat(clauses, atoms)
    assert res["result"] in ("sat", "unsat")
    assert math.isfinite(reward)
    stats = tr.update(steps, reward, key=0)
    assert math.isfinite(stats["loss"])


def test_decide_rl_parallel_collect():
    """多进程 collect + 主进程 update。"""
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    insts = generate_bool_lia_dataset(4, seed=2, min_vars=5, max_vars=5)
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=40, workers=2))
    hist = tr.train(
        [i.as_tuple() for i in insts],
        iterations=1,
        workers=2,
        collect_seed=2,
        collect_min_vars=5,
        collect_max_vars=5,
    )
    assert len(hist) == 4
    assert all(h["steps"] >= 0 for h in hist)
