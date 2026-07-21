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
    dec = SamplingPolicyDecider(
        policy, defer, asserts, refocus_every=100, sample=True, sticky_window=True
    )
    und = [atom_key(a), atom_key(b)]
    torch.manual_seed(0)
    outs = [dec(und, {}) for _ in range(5)]
    # 每次返回 None(defer) 或 合法未定原子+bool
    assert all(o is None or (o[0] in und and isinstance(o[1], bool)) for o in outs)
    # sticky_window：整窗只采样/记 step 一次
    assert len(dec.steps) == 1
    g, ls, idx = dec.steps[0]
    assert 0 <= idx <= len(ls)            # idx=0=defer, 1..len=原子


def test_sampling_decider_sticky_reuses_scores_without_resample():
    """窗口内后续回调不再增加 steps，且在粘性原子仍未定时重复返回同一选择。"""
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    policy = BranchingPolicy()
    defer = torch.tensor(-10.0)  # 极低 defer → 几乎必采原子
    dec = SamplingPolicyDecider(
        policy, defer, asserts, refocus_every=10, sample=True, sticky_window=True
    )
    und = [atom_key(a), atom_key(b)]
    torch.manual_seed(1)
    o1 = dec(und, {})
    o2 = dec(und, {})
    o3 = dec(und, {})
    assert len(dec.steps) == 1
    assert o1 is not None and o1[0] in und
    assert o2 == o1 and o3 == o1


def test_sampling_decider_backtrack_forces_new_window():
    """on_backtrack 清空粘性窗，下次 decide 重新 refocus 并记新 step。"""
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    policy = BranchingPolicy()
    defer = torch.tensor(-10.0)
    dec = SamplingPolicyDecider(
        policy, defer, asserts, refocus_every=100, sample=True, sticky_window=True
    )
    und = [atom_key(a), atom_key(b)]
    torch.manual_seed(2)
    _ = dec(und, {})
    assert len(dec.steps) == 1
    dec.on_backtrack(1)
    assert dec._since == dec.refocus_every
    assert dec._graph is None
    _ = dec(und, {})
    assert len(dec.steps) == 2


def test_sampling_decider_nonsticky_records_every_call():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    policy = BranchingPolicy()
    defer = torch.zeros(())
    dec = SamplingPolicyDecider(
        policy, defer, asserts, refocus_every=100, sample=True, sticky_window=False
    )
    und = [atom_key(a), atom_key(b)]
    torch.manual_seed(0)
    _ = [dec(und, {}) for _ in range(5)]
    assert len(dec.steps) == 5


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


def test_sat_conflict_reward_normalized_and_monotone():
    """归一化 reward 恒 [-1,0]:0=零冲突,-0.5=与 VSIDS 持平,-1=≥2×ref;单调、有界。"""
    from omt_branching.solver.rl_decide import sat_conflict_reward

    assert sat_conflict_reward(0, 100) == 0.0
    assert sat_conflict_reward(100, 100, cap=2.0) == -0.5
    assert sat_conflict_reward(200, 100, cap=2.0) == -1.0
    assert sat_conflict_reward(10_000, 100) == -1.0          # 有界:离群不爆
    for c in (0, 1, 50, 100, 300, 5000):
        assert -1.0 <= sat_conflict_reward(c, 100) <= 0.0
    rs = [sat_conflict_reward(c, 100) for c in (0, 10, 50, 100, 200)]
    assert all(a >= b for a, b in zip(rs, rs[1:]))           # 单调不增
    # 无参考 → ref=1 保守惩罚
    assert sat_conflict_reward(0, None) == 0.0
    assert sat_conflict_reward(5, None) == -1.0


def test_collect_sat_reward_in_unit_interval():
    from omt_branching.solver.sat_instances import generate_rand_3sat
    from omt_branching.solver.sat_solve import solve_sat_with_decider
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    atoms, clauses = generate_rand_3sat(24, 4.26, seed=2)
    ref = solve_sat_with_decider(clauses, atoms, None)["conflicts"]
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=40))
    steps, reward, res = tr.collect_sat(clauses, atoms, ref_conflicts=ref)
    assert -1.0 <= reward <= 0.0
    stats = tr.update(steps, reward, key=0)
    assert stats["steps"] >= 0


def test_train_sat_normalized_reward_history():
    from omt_branching.solver.sat_instances import generate_rand_3sat
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    probs = [generate_rand_3sat(20, 4.26, seed=s) for s in (1, 2)]
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=40))
    hist = tr.train_sat(probs, iterations=1)
    assert len(hist) == 2
    for h in hist:
        assert -1.0 <= h["reward"] <= 0.0
        assert h["ref_conflicts"] >= 0
        assert "conflicts" in h


def test_decide_rl_parallel_collect():
    """多进程 collect + GpuInferService 排队推理 + 主进程 update。"""
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    insts = generate_bool_lia_dataset(8, seed=2, min_vars=5, max_vars=5)
    tr = DecideRLTrainer(
        BranchingPolicy(),
        DecideRLConfig(
            refocus_every=40,
            workers=2,
            min_instances_for_parallel=4,
            use_all_gpus=False,
        ),
    )
    hist = tr.train(
        [i.as_tuple() for i in insts],
        iterations=1,
        workers=2,
        collect_seed=2,
        collect_min_vars=5,
        collect_max_vars=5,
    )
    # train_end 汇总一条
    assert len(hist) == 9
    assert all(h.get("steps", 0) >= 0 for h in hist if "steps" in h)


def test_decide_rl_collect_batch_size():
    """每轮只 collect 指定 batch，history 步数 = batch（另加 train_end）。"""
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    insts = generate_bool_lia_dataset(8, seed=4, min_vars=5, max_vars=5)
    tr = DecideRLTrainer(
        BranchingPolicy(),
        DecideRLConfig(
            refocus_every=40,
            workers=1,
            collect_batch_size=3,
            sticky_window=False,
        ),
    )
    hist = tr.train(
        [i.as_tuple() for i in insts],
        iterations=2,
        workers=1,
        collect_seed=4,
        collect_min_vars=5,
        collect_max_vars=5,
        collect_batch_size=3,
    )
    steps = [h for h in hist if "steps" in h and "event" not in h]
    assert len(steps) == 6  # 2 iters × 3
    end = hist[-1]
    assert end.get("event") == "train_end"
    assert end.get("collect_batch_size") == 3
    assert end.get("sticky_window") is False


def test_gpu_infer_pool_queues_slots():
    """空闲槽排队：单设备上两次 infer 均可完成。"""
    from omt_branching.solver.rl_decide import GpuInferPool
    from omt_branching.input.graph_builder import GraphBuilder
    from omt_branching.input.solver_state import (
        BooleanVarInfo,
        ClauseInfo,
        ObjectiveInfo,
        SearchStateInfo,
        SolverSnapshot,
    )

    policy = BranchingPolicy()
    pool = GpuInferPool.from_policy(policy, device="cpu", use_all_gpus=False)
    snap = SolverSnapshot(
        bool_vars=[
            BooleanVarInfo(var_id="a", is_candidate=True),
            BooleanVarInfo(var_id="b", is_candidate=True),
        ],
        clauses=[ClauseInfo(clause_id="c0", literals=[("a", True), ("b", False)])],
        objective=ObjectiveInfo(objective_id="obj"),
        search_state=SearchStateInfo(),
        candidate_bool_ids=["a", "b"],
    )
    g = GraphBuilder().build(snap)
    s1, p1 = pool.infer(g)
    s2, p2 = pool.infer(g)
    assert s1.shape == s2.shape == (2,)
    assert p1.shape == p2.shape == (2,)


def test_gpu_infer_service_remote_client():
    """主进程 GpuInferService + RemoteInferClient（Queue + Pipe）。"""
    import multiprocessing as mp

    from omt_branching.input.graph_builder import GraphBuilder
    from omt_branching.input.solver_state import (
        BooleanVarInfo,
        ClauseInfo,
        ObjectiveInfo,
        SearchStateInfo,
        SolverSnapshot,
    )
    from omt_branching.solver.rl_decide import GpuInferService, RemoteInferClient

    ctx = mp.get_context("spawn")
    req = ctx.Queue()
    policy = BranchingPolicy()
    svc = GpuInferService.from_policy(
        policy, req, device="cpu", use_all_gpus=False
    )
    svc.start()
    try:
        client = RemoteInferClient(req, ctx)
        snap = SolverSnapshot(
            bool_vars=[
                BooleanVarInfo(var_id="a", is_candidate=True),
                BooleanVarInfo(var_id="b", is_candidate=True),
            ],
            clauses=[
                ClauseInfo(clause_id="c0", literals=[("a", True), ("b", False)])
            ],
            objective=ObjectiveInfo(objective_id="obj"),
            search_state=SearchStateInfo(),
            candidate_bool_ids=["a", "b"],
        )
        g = GraphBuilder().build(snap)
        s, p = client.infer(g)
        assert s.shape == (2,)
        assert p.shape == (2,)
    finally:
        svc.stop()

