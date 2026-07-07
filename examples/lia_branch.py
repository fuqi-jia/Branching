"""LIA B&B 学习分支实验（plain 模式，check-sat 叶子）。

knapsack LIA -> 数值 head strong-branching imitation 冷启动 -> RL 微调 -> 对比各分支策略的
**搜索规模**（solve_calls/splits/steps）；所有策略饱和到 == native 精确最优。

要点（见 docs/superpowers/specs/2026-07-06-lia-branch-experiment-design.md）：

- plain 模式叶子是 ``Solve``（check-sat），故神经 F-Split **就是**优化器；比较的是分支质量。
- native z3 ``Optimize`` 一次求得精确最优，作 ground truth 与 skyline；GOMT 增量回路是外层
  分支框架，节点数远多于 native——本实验比较的是**框架内**学习分支 vs 启发式分支。
- ``strong`` 基线在每个节点做一层前瞻，非常昂贵，默认不参与对比（``--with-strong`` 开启）。

运行::

    python -m examples.lia_branch                 # 默认小规模
    python -m examples.lia_branch --with-strong   # 加入 strong skyline（慢）
"""

from __future__ import annotations

import argparse

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver import (
    NumericHeuristicStrategy, RLConfig, RLRecordingStrategy, SolverInLoopRLTrainer,
    Z3Backend, build_imitation_examples, generate_hard_lia_dataset, solve_and_measure,
    solve_native,
)

F_SAT_MODE = "plain"   # LIA：check-sat 叶子，分支驱动优化


def _measure_native(hard, obj, sense) -> dict:
    backend = Z3Backend()
    backend.optimize(backend.conjoin(*hard), obj, sense)
    return {"solve_calls": backend.solve_calls, "rlimit": backend.rlimit_count}


def compare(policy, rl_config, instances, max_steps, modes):
    """对每个实例跑各策略，返回策略 -> 平均 {solve_calls, splits, steps, match}。"""
    names = ["native", *modes, "neural"]
    agg = {n: {"solve_calls": 0.0, "splits": 0.0, "steps": 0.0, "match": 0.0} for n in names}
    for inst in instances:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)
        nat = _measure_native(hard, obj, sense)
        agg["native"]["solve_calls"] += nat["solve_calls"]
        agg["native"]["match"] += 1.0
        for m in modes:
            r = solve_and_measure(hard, obj, sense,
                                  lambda p, m=m: NumericHeuristicStrategy(p, mode=m, seed=0),
                                  max_steps=max_steps, f_sat_mode=F_SAT_MODE)
            for k in ("solve_calls", "splits", "steps"):
                agg[m][k] += r[k]
            agg[m]["match"] += 1.0 if r["value"] == native else 0.0
        rn = solve_and_measure(hard, obj, sense,
                               lambda p: RLRecordingStrategy(p, policy, rl_config, sample=False),
                               max_steps=max_steps, f_sat_mode=F_SAT_MODE)
        for k in ("solve_calls", "splits", "steps"):
            agg["neural"][k] += rn[k]
        agg["neural"]["match"] += 1.0 if rn["value"] == native else 0.0
    n = max(1, len(instances))
    for name in agg:
        for k in agg[name]:
            agg[name][k] /= n
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="LIA B&B 学习分支实验")
    parser.add_argument("--train", type=int, default=120)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--min-vars", type=int, default=4)
    parser.add_argument("--max-vars", type=int, default=6)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--split-depth", type=int, default=6)
    parser.add_argument("--with-strong", action="store_true", help="加入 strong skyline（慢）")
    args = parser.parse_args()

    torch.manual_seed(0)
    print("=== 1) 生成 knapsack LIA ===")
    train = generate_hard_lia_dataset(args.train, seed=1,
                                      min_vars=args.min_vars, max_vars=args.max_vars)
    test = generate_hard_lia_dataset(args.test, seed=99,
                                     min_vars=args.min_vars, max_vars=args.max_vars)
    print(f"训练 {len(train)} / 测试 {len(test)}")

    print("\n=== 2) 数值 head strong-branching imitation ===")
    policy = BranchingPolicy()
    examples = build_imitation_examples(train, numeric_expert="strong")
    print(f"imitation 样本 {len(examples)}")
    hist = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(examples, epochs=args.epochs)
    print(f"imitation loss: {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f}")

    print("\n=== 3) RL 微调（plain，最小化搜索规模）===")
    rl_config = RLConfig(lr=1e-3, gamma=0.98, entropy_coef=5e-3, rlimit_penalty_coef=1.0,
                         use_log_cost=True, reward_scale=1.0, saturation_bonus=0.0,
                         max_split_depth=args.split_depth, max_steps=args.max_steps,
                         f_sat_mode=F_SAT_MODE)
    rl = SolverInLoopRLTrainer(policy, rl_config)
    rl.train([i.as_tuple() for i in train], iterations=args.iters, log=False)

    print("\n=== 4) 搜索规模对比（solve_calls/splits/steps 越小越好；match 应=1.0）===")
    modes = ["random", "largest_domain", "largest_coeff"]
    if args.with_strong:
        modes.append("strong")
    agg = compare(policy, rl_config, test, args.max_steps, modes)
    print(f"  {'strategy':<16} {'solve_calls':>12} {'splits':>10} {'steps':>10} {'match':>8}")
    for name in ["native", *modes, "neural"]:
        a = agg[name]
        print(f"  {name:<16} {a['solve_calls']:>12.2f} {a['splits']:>10.2f} "
              f"{a['steps']:>10.2f} {a['match']:>8.2f}")
    print("\nplain 模式：所有策略饱和到 == native 精确最优；比较搜索规模。"
          "neural 目标：solve_calls/splits 低于启发式（框架内学习分支的收益）；"
          "native 为 skyline（外层 GOMT 分支不与之比 wall-clock）。")


if __name__ == "__main__":
    main()
