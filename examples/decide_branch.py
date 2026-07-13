"""三臂对比：native z3 Optimize / VSIDS-decide / learned-decide（未训练 GNN）。

证明管道正确（learned 臂 == native）并测量 rlimit/conflicts/decisions。**本 Phase 不训练**，
故 learned 未必优于 VSIDS——目的是管道 + 可测量，为 Phase 2（look-ahead imitation + RL）铺路。
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import (
    # Z3Backend,
    generate_bool_lia_dataset,
    solve_native,
    solve_omt_with_decider,
)
from omt_branching.solver.policy_decider import PolicyDecider

from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description="UserPropagator 学习分支三臂对比")
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--refocus", type=int, default=50)
    ap.add_argument(
        "--train", type=int, default=0, help="look-ahead imitation 训练集规模(0=不训练)"
    )
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
        hist = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(
            exs, epochs=args.epochs
        )
        print(
            f"look-ahead imitation: {len(exs)} 样本, branch loss "
            f"{hist[0].get('branch', 0):.3f} -> {hist[-1].get('branch', 0):.3f}"
        )
    if args.rl_iters > 0:
        from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

        rl_train = gen(
            max(args.train, 40), seed=1, min_vars=args.min_vars, max_vars=args.max_vars
        )
        rlt = DecideRLTrainer(policy, DecideRLConfig(refocus_every=args.refocus))
        h = rlt.train(
            [i.as_tuple() for i in rl_train], iterations=args.rl_iters, log=False
        )
        if h:
            print(
                f"RL 微调: {len(h)} 步, 末条 reward={h[-1]['reward']:.3f} "
                f"conflicts={h[-1]['conflicts']}, defer_logit={float(rlt.defer_logit):.3f}"
            )
    svc = BranchingPolicyService(policy=policy)

    agg = {
        "native": {"rlimit": 0.0},
        "vsids": {
            "rlimit": 0.0,
            # "solver rlimit": 0.0,
            "decider factory rlimit": 0.0,
            "model base rlimit": 0.0,
            "model cut rlimit": 0.0,
            "check rlimit": 0.0,
            "eval rlimit": 0.0,
            "weighted rlimit": 0.0,
            "conflicts": 0.0,
            "match": 0.0,
        },
        "learned": {
            "rlimit": 0.0,
            # "solver rlimit": 0.0,
            "decider factory rlimit": 0.0,
            "model base rlimit": 0.0,
            "model cut rlimit": 0.0,
            "check rlimit": 0.0,
            "eval rlimit": 0.0,
            "weighted rlimit": 0.0,
            "conflicts": 0.0,
            "decisions": 0.0,
            "match": 0.0,
        },
    }
    skipped = 0
    with tqdm(total=len(insts), desc="test") as pbar:
        for inst in insts:
            hard, obj, sense = inst.as_tuple()
            if h:
                rewards = [col["reward"] for col in h]
                rlimit_max = math.exp(-min(rewards))-1
                rlimit_bound = int(rlimit_max * 100)
            else:
                rlimit_bound = -1
            nat = solve_native(hard, obj, sense, max_rlimit=rlimit_bound)
            if nat["value"] is None:
                skipped += 1
                pbar.update(1)
                continue
            agg["native"]["rlimit"] += nat["rlimit"]
            v = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
            for key in v.keys():
                if key not in agg["vsids"]:
                    continue
                agg["vsids"][key] += v[key]
            agg["vsids"]["match"] += 1.0 if v["value"] == nat["value"] else 0.0
            ln = solve_omt_with_decider(
                hard,
                obj,
                sense,
                decider_factory=lambda a: PolicyDecider(svc, a, args.refocus),
            )
            for key in ln.keys():
                if key not in agg["learned"]:
                    continue
                agg["learned"][key] += ln[key]
            agg["learned"]["match"] += 1.0 if ln["value"] == nat["value"] else 0.0
            pbar.update(1)

    n = max(1, len(insts)) - skipped
    print(
        f"=== 三臂对比（{len(insts)} 实例，未训练 GNN；rlimit/conflicts 越小越好，match=1 为正确）==="
    )
    # print(f"  native(z3 Optimize): rlimit={agg['native']['rlimit']/n:.0f}")
    # print(
    #     f"  VSIDS-decide       : rlimit={agg['vsids']['rlimit']/n:.0f} weighted={agg['vsids']['weighted rlimit']/n:.0f} "
    #     f"model_base={agg['vsids']['model base rlimit']/n:.0f} model_cut={agg['vsids']['model cut rlimit']/n:.0f} "
    #     f"check={100.0 * agg['vsids']['check rlimit'] / agg['vsids']['rlimit']:.2f}% eval={100.0 * agg['vsids']['eval rlimit'] / agg['vsids']['rlimit']:.2f}% "
    #     f"conflicts={agg['vsids']['conflicts']/n:.1f} match={agg['vsids']['match']/n:.2f}"
    # )
    # print(
    #     f"  learned-decide     : rlimit={agg['learned']['rlimit']/n:.0f} weighted={agg['learned']['weighted rlimit']/n:.0f} "
    #     f"model_base={agg['learned']['model base rlimit']/n:.0f} model_cut={agg['learned']['model cut rlimit']/n:.0f} "
    #     f"check={100.0 * agg['learned']['check rlimit'] / agg['learned']['rlimit']:.2f}% eval={100.0 * agg['learned']['eval rlimit'] / agg['learned']['rlimit']:.2f}% "
    #     f"conflicts={agg['learned']['conflicts']/n:.1f} decisions={agg['learned']['decisions']/n:.1f} "
    #     f"match={agg['learned']['match']/n:.2f}"
    # )
    # print(
    #     "\nPhase 1 目标：learned 臂 match=1（管道正确）+ 可测量。Phase 2 再训练使其优于 VSIDS。"
    # )
    agg["native"]["rlimit"] /= n
    for key in agg["vsids"].keys():
        agg["vsids"][key] /= n
    for key in agg["learned"].keys():
        agg["learned"][key] /= n
    with open("examples/artifacts/results.json", "w") as f:
        json.dump(agg, f, indent=4)


if __name__ == "__main__":
    main()
