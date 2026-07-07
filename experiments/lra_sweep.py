"""LRA 多 seed 全规模 sweep：对每个 seed 跑完整 imitation+RL 流程，聚合
native/baseline/neural 的 rlimit/solve_calls/splits/gap/exact 与 bool 准确率的
mean±std，落 JSON（**逐 seed 增量保存**，中途中断也保留已完成结果）。

复用 :mod:`examples.rl_LRA` 的 ``cost_comparison`` / ``branch_accuracy`` 与
``omt_branching.solver`` 的库函数；仅在此参数化 seed 与规模，得到有方差估计的对比。

运行（默认 3 seed、全规模 500/100，耗时数小时）::

    python -m experiments.lra_sweep

快速校验（小规模、多 seed）::

    python -m experiments.lra_sweep --train 60 --test 30 --seeds 2 --max-vars 12
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from examples.rl_LRA import F_SAT_MODE, branch_accuracy, cost_comparison
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver import (
    RLConfig, SolverInLoopRLTrainer, build_imitation_examples, generate_lra_dataset,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "results")
STRATEGIES = ("native", "baseline", "neural")
METRICS = ("rlimit", "solve_calls", "splits", "gap", "exact")


def run_seed(seed: int, args) -> dict:
    """对单个 seed 跑完整流程，返回该 seed 的 bool 准确率与三策略开销/质量聚合。"""
    torch.manual_seed(seed)
    train = generate_lra_dataset(args.train, seed=seed,
                                 min_vars=args.min_vars, max_vars=args.max_vars)
    test = generate_lra_dataset(args.test, seed=seed + 1000,
                                min_vars=args.min_vars, max_vars=args.max_vars)

    policy = BranchingPolicy()
    examples = build_imitation_examples(train)
    ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(examples, epochs=args.epochs)

    rl_config = RLConfig(
        lr=1e-3, gamma=0.98, entropy_coef=5e-3,
        rlimit_penalty_coef=1.0, use_log_cost=True, reward_scale=1.0,
        saturation_bonus=args.saturation_bonus,
        max_split_depth=args.split_depth, max_steps=args.max_steps, f_sat_mode=F_SAT_MODE,
    )
    rl = SolverInLoopRLTrainer(policy, rl_config)
    rl.train([inst.as_tuple() for inst in train], iterations=args.iters, log=False)

    acc, n_valid = branch_accuracy(policy, test)
    agg = cost_comparison(policy, rl_config, test, args.max_steps)
    return {"seed": seed, "bool_acc": acc, "bool_acc_n": n_valid, "agg": agg}


def aggregate(results: list[dict]) -> dict:
    """跨 seed 聚合每策略每指标的 mean/std，以及 bool 准确率 mean/std。"""
    out: dict = {"n_seeds": len(results), "strategies": {}}
    accs = [r["bool_acc"] for r in results]
    out["bool_acc_mean"] = statistics.fmean(accs)
    out["bool_acc_std"] = statistics.pstdev(accs) if len(accs) > 1 else 0.0
    for strat in STRATEGIES:
        out["strategies"][strat] = {}
        for m in METRICS:
            vals = [r["agg"][strat][m] for r in results]
            out["strategies"][strat][m] = {
                "mean": statistics.fmean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
    return out


def _print_summary(summary: dict) -> None:
    print(f"\n=== Sweep 汇总（{summary['n_seeds']} seeds，mean±std）===")
    print(f"bool 准确率(vs strong-branching 专家): "
          f"{summary['bool_acc_mean']:.3f} ± {summary['bool_acc_std']:.3f}")
    print(f"  {'strategy':<12} {'rlimit':>16} {'solve_calls':>14} {'splits':>10} "
          f"{'gap':>14} {'exact':>14}")
    for strat in STRATEGIES:
        s = summary["strategies"][strat]
        print(f"  {strat:<12} "
              f"{s['rlimit']['mean']:>10.0f}±{s['rlimit']['std']:<5.0f} "
              f"{s['solve_calls']['mean']:>7.2f}±{s['solve_calls']['std']:<5.2f} "
              f"{s['splits']['mean']:>5.2f}±{s['splits']['std']:<4.2f} "
              f"{s['gap']['mean']:>7.4f}±{s['gap']['std']:<6.4f} "
              f"{s['exact']['mean']:>6.2f}±{s['exact']['std']:<6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LRA 多 seed 全规模 sweep")
    parser.add_argument("--train", type=int, default=500)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--min-vars", type=int, default=10)
    parser.add_argument("--max-vars", type=int, default=14)
    parser.add_argument("--seeds", type=int, default=3, help="seed 数（从 base-seed 起递增）")
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--split-depth", type=int, default=3)
    parser.add_argument("--saturation-bonus", type=float, default=8.0)
    parser.add_argument("--out", default=os.path.join(ARTIFACTS, "lra_sweep.json"))
    args = parser.parse_args()

    os.makedirs(ARTIFACTS, exist_ok=True)
    seeds = [args.base_seed + i for i in range(args.seeds)]
    print(f"LRA sweep：{args.seeds} seeds × (train {args.train}/test {args.test}, "
          f"vars {args.min_vars}..{args.max_vars}, iters {args.iters}, "
          f"sat_bonus {args.saturation_bonus}) -> {args.out}")

    results: list[dict] = []
    for s in seeds:
        r = run_seed(s, args)
        results.append(r)
        # 增量落盘：每完成一个 seed 就写一次，中断也保留。
        payload = {"config": vars(args), "per_seed": results,
                   "summary": aggregate(results)}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        a = r["agg"]
        print(f"[seed {s}] bool_acc={r['bool_acc']:.3f} (n={r['bool_acc_n']}) | "
              f"neural gap={a['neural']['gap']:.4f} exact={a['neural']['exact']:.2f} "
              f"rlimit={a['neural']['rlimit']:.0f} | "
              f"baseline gap={a['baseline']['gap']:.4f} exact={a['baseline']['exact']:.2f} "
              f"rlimit={a['baseline']['rlimit']:.0f}")

    _print_summary(aggregate(results))
    print(f"\n完整结果已保存 -> {args.out}")


if __name__ == "__main__":
    main()
