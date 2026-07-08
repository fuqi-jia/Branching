"""三臂对比：native z3 Optimize / VSIDS-decide / learned-decide（未训练 GNN）。

证明管道正确（learned 臂 == native）并测量 rlimit/conflicts/decisions。**本 Phase 不训练**，
故 learned 未必优于 VSIDS——目的是管道 + 可测量，为 Phase 2（look-ahead imitation + RL）铺路。
"""
from __future__ import annotations

import argparse

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import (
    Z3Backend, generate_bool_lia_dataset, solve_native, solve_omt_with_decider,
)
from omt_branching.solver.policy_decider import PolicyDecider


def _native_rlimit(hard, obj, sense):
    b = Z3Backend()
    b.optimize(b.conjoin(*hard), obj, sense)
    return b.rlimit_count


def main() -> None:
    ap = argparse.ArgumentParser(description="UserPropagator 学习分支三臂对比")
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--refocus", type=int, default=50)
    ap.add_argument("--train", type=int, default=0, help="look-ahead imitation 训练集规模(0=不训练)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rl-iters", type=int, default=0, help="RL 微调轮数(0=不做 RL)")
    ap.add_argument("--hard", action="store_true", help="用更难实例(headroom)")
    args = ap.parse_args()

    from omt_branching.solver import generate_hard_bool_lia_dataset
    gen = generate_hard_bool_lia_dataset if args.hard else generate_bool_lia_dataset

    torch.manual_seed(0)
    insts = gen(args.test, seed=99, min_vars=args.min_vars, max_vars=args.max_vars)

    policy = BranchingPolicy()
    if args.train > 0:
        from omt_branching.model.trainer import ImitationTrainer, TrainConfig
        from omt_branching.solver.training_data import build_lookahead_examples
        train = gen(args.train, seed=1, min_vars=args.min_vars, max_vars=args.max_vars)
        exs = [e for e in build_lookahead_examples(train) if e.bool_target_scores]
        hist = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=args.epochs)
        print(f"look-ahead imitation: {len(exs)} 样本, branch loss "
              f"{hist[0].get('branch', 0):.3f} -> {hist[-1].get('branch', 0):.3f}")
    if args.rl_iters > 0:
        from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig
        rl_train = gen(max(args.train, 40), seed=1, min_vars=args.min_vars, max_vars=args.max_vars)
        rlt = DecideRLTrainer(policy, DecideRLConfig(refocus_every=args.refocus))
        h = rlt.train([i.as_tuple() for i in rl_train], iterations=args.rl_iters, log=False)
        if h:
            print(f"RL 微调: {len(h)} 步, 末条 reward={h[-1]['reward']:.3f} "
                  f"conflicts={h[-1]['conflicts']}, defer_logit={float(rlt.defer_logit):.3f}")
    svc = BranchingPolicyService(policy=policy)

    agg = {"native": {"rlimit": 0.0},
           "vsids": {"rlimit": 0.0, "conflicts": 0.0, "match": 0.0},
           "learned": {"rlimit": 0.0, "conflicts": 0.0, "decisions": 0.0, "match": 0.0}}
    for inst in insts:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)
        agg["native"]["rlimit"] += _native_rlimit(hard, obj, sense)
        v = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
        agg["vsids"]["rlimit"] += v["rlimit"]
        agg["vsids"]["conflicts"] += v["conflicts"]
        agg["vsids"]["match"] += 1.0 if v["value"] == native else 0.0
        ln = solve_omt_with_decider(
            hard, obj, sense,
            decider_factory=lambda a: PolicyDecider(svc, a, args.refocus))
        agg["learned"]["rlimit"] += ln["rlimit"]
        agg["learned"]["conflicts"] += ln["conflicts"]
        agg["learned"]["decisions"] += ln["decisions"]
        agg["learned"]["match"] += 1.0 if ln["value"] == native else 0.0

    n = max(1, len(insts))
    print(f"=== 三臂对比（{len(insts)} 实例，未训练 GNN；rlimit/conflicts 越小越好，match=1 为正确）===")
    print(f"  native(z3 Optimize): rlimit={agg['native']['rlimit']/n:.0f}")
    print(f"  VSIDS-decide       : rlimit={agg['vsids']['rlimit']/n:.0f} "
          f"conflicts={agg['vsids']['conflicts']/n:.1f} match={agg['vsids']['match']/n:.2f}")
    print(f"  learned-decide     : rlimit={agg['learned']['rlimit']/n:.0f} "
          f"conflicts={agg['learned']['conflicts']/n:.1f} decisions={agg['learned']['decisions']/n:.1f} "
          f"match={agg['learned']['match']/n:.2f}")
    print("\nPhase 1 目标：learned 臂 match=1（管道正确）+ 可测量。Phase 2 再训练使其优于 VSIDS。")


if __name__ == "__main__":
    main()
