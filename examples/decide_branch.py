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
    args = ap.parse_args()

    torch.manual_seed(0)
    insts = generate_bool_lia_dataset(args.test, seed=99,
                                      min_vars=args.min_vars, max_vars=args.max_vars)
    svc = BranchingPolicyService(policy=BranchingPolicy())

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
