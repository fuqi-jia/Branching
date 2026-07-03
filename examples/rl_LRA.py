"""完整训练流程示例（**LRA 版**）：生成复杂 LRA 实例 -> GNN 监督训练 -> RL 训练 -> 测试对比。

与 :mod:`examples.rl_demo`（LIA 版）的关键差异，源于 LRA 的求解语义：

- **实数不做域二分**：GNN 只在**布尔结构**上分支，连续优化交给 z3 ``Optimize``
  （``f_sat_mode="hybrid"`` 的叶子），因此**必须**用 hybrid 模式，否则 plain 线性搜索
  对实数不有限终止。
- **有界 episode（anytime）**：GOMT 的增量式线性搜索对实数不有限饱和，故训练/评测都用
  ``max_steps`` 预算；``optimal`` 常为 ``False``。真最优由 z3 原生 ``Optimize``
  （:func:`solve_native`）给出，作为**最优性 gap** 的参照真值——不再断言“求得精确最优”。
- **Baseline = 直接 Optimize**：``BaselineStrategy`` 对实数不二分 -> 在 hybrid 下根节点一次
  ``resolve`` 即得全局最优（精确、开销极小）。这也说明：对 LRA，GNN 的价值不在于“比 z3
  更快求全局最优”，而在于**布尔结构分支的取舍**；RL 的目标是学会**少做无益的 split**
  （降低 rlimit 开销），并让数值 head 的排序贴近专家（准确率）。

评测指标：
1. **准确率**：数值 head top-1 与专家（``|目标系数|`` 最大）一致的比例（imitation 学习效果）。
2. **开销/质量**：测试集上 Neural 与 Baseline 的平均 rlimit / solve_calls / splits，
   以及**最优性 gap** ``|incumbent - native| / (|native|+eps)`` 与精确命中率。

运行（默认 500/100，变量数 ≥10；耗时较长）::

    python -m examples.rl_LRA

快速冒烟（小规模，用于验证流程）::

    python -m examples.rl_LRA --train 6 --test 6 --min-vars 10 --max-vars 10 --iters 1
"""

from __future__ import annotations

import argparse
import os

import torch

from omt_branching.model.persistence import load_policy, save_history, save_policy
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver import (
    BaselineStrategy,
    RLConfig,
    RLRecordingStrategy,
    SolverInLoopRLTrainer,
    baseline_numeric_choice,
    build_imitation_examples,
    generate_lra_dataset,
    oracle_numeric_choice,
    policy_numeric_choice,
    solve_and_measure,
    solve_native,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
GNN_CKPT = os.path.join(ARTIFACTS, "gnn_lra.pt")
RL_CKPT = os.path.join(ARTIFACTS, "rl_lra_policy.pt")
HISTORY_JSON = os.path.join(ARTIFACTS, "rl_lra_history.json")

# LRA 用 hybrid（叶子 Optimize），有界 episode 预算保证终止。
F_SAT_MODE = "hybrid"


def branch_accuracy(policy, instances) -> tuple[float, float]:
    """返回 (Neural 数值 head top-1 与专家一致比例, Baseline 与专家一致比例)。"""
    neural_hit = base_hit = total = 0
    for inst in instances:
        oracle = oracle_numeric_choice(inst)
        if oracle is None:
            continue
        total += 1
        if policy_numeric_choice(policy, inst) == oracle:
            neural_hit += 1
        if baseline_numeric_choice(inst) == oracle:
            base_hit += 1
    if total == 0:
        return 0.0, 0.0
    return neural_hit / total, base_hit / total


def _gap(value, native) -> float:
    """相对最优性 gap；value 可能是巨大分母的 Fraction，float() 安全。"""
    try:
        fv, fn = float(value), float(native)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return abs(fv - fn) / (abs(fn) + 1e-9)


def cost_comparison(policy, rl_config, instances, max_steps):
    """在测试集上对比 Neural 与 Baseline 的开销与最优性 gap（hybrid, anytime）。"""
    agg = {"neural": _fresh(), "baseline": _fresh()}
    for inst in instances:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)

        neural = solve_and_measure(
            hard, obj, sense,
            lambda p: RLRecordingStrategy(p, policy, rl_config, sample=False),
            max_steps=max_steps, f_sat_mode=F_SAT_MODE,
        )
        base = solve_and_measure(
            hard, obj, sense, lambda p: BaselineStrategy(p),
            max_steps=max_steps, f_sat_mode=F_SAT_MODE)

        _accumulate(agg["neural"], neural, native)
        _accumulate(agg["baseline"], base, native)
    n = max(1, len(instances))
    for k in agg:
        for m in ("rlimit", "solve_calls", "splits", "gap", "exact"):
            agg[k][m] /= n
    return agg


def _fresh():
    return {"rlimit": 0.0, "solve_calls": 0.0, "splits": 0.0,
            "gap": 0.0, "exact": 0.0}


def _accumulate(acc, res, native):
    acc["rlimit"] += res["rlimit"]
    acc["solve_calls"] += res["solve_calls"]
    acc["splits"] += res["splits"]
    acc["gap"] += _gap(res["value"], native)
    acc["exact"] += 1.0 if (res["value"] == native and res["optimal"]) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LRA 完整训练/评测流程示例")
    parser.add_argument("--train", type=int, default=500, help="训练集规模")
    parser.add_argument("--test", type=int, default=100, help="测试集规模")
    parser.add_argument("--min-vars", type=int, default=10, help="最小变量数(>=10)")
    parser.add_argument("--max-vars", type=int, default=14, help="最大变量数")
    parser.add_argument("--iters", type=int, default=1, help="RL 训练轮数")
    parser.add_argument("--epochs", type=int, default=30, help="imitation 轮数")
    parser.add_argument("--max-steps", type=int, default=80, help="每个 episode 的步数预算")
    parser.add_argument("--split-depth", type=int, default=3, help="每 Δ-round 的 split 预算")
    parser.add_argument("--rl-log", action="store_true", help="逐实例打印 RL 训练日志")
    args = parser.parse_args()

    if args.min_vars < 10:
        parser.error("--min-vars 需 >= 10")

    torch.manual_seed(0)
    os.makedirs(ARTIFACTS, exist_ok=True)

    # ---------------- 1) 生成复杂 LRA 实例 ----------------
    print("=== 1) 生成 OMT(LRA) 实例（布尔结构，变量数 >=10） ===")
    train_set = generate_lra_dataset(args.train, seed=1,
                                     min_vars=args.min_vars, max_vars=args.max_vars)
    test_set = generate_lra_dataset(args.test, seed=99,
                                    min_vars=args.min_vars, max_vars=args.max_vars)
    print(f"训练集 {len(train_set)} 个，测试集 {len(test_set)} 个 "
          f"(vars {args.min_vars}..{args.max_vars})")
    for inst in test_set[:3]:
        print(f"  {inst.instance_id}: family={inst.family} vars={len(inst.variables)} "
              f"hard={len(inst.hard)} sense={inst.sense.value} "
              f"oracle={oracle_numeric_choice(inst)}")
    print("  ...")

    # ---------------- 2) GNN 监督训练 + 保权 ----------------
    print("\n=== 2) GNN 监督训练 (imitation 冷启动) ===")
    policy = BranchingPolicy()
    acc_before, base_acc = branch_accuracy(policy, test_set)
    print(f"训练前 数值分支准确率: Neural={acc_before:.2f}  Baseline={base_acc:.2f}")

    examples = build_imitation_examples(train_set)
    print(f"imitation 样本数: {len(examples)}")
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    history_imit = trainer.fit(examples, epochs=args.epochs)
    print(f"imitation loss: 首轮 {history_imit[0]['loss']:.4f} -> "
          f"末轮 {history_imit[-1]['loss']:.4f}")

    acc_after, _ = branch_accuracy(policy, test_set)
    print(f"训练后 数值分支准确率: Neural={acc_after:.2f}  Baseline={base_acc:.2f}")

    save_policy(policy, GNN_CKPT, meta={"stage": "imitation", "theory": "LRA",
                                        "epochs": args.epochs})
    print(f"GNN 权重已保存 -> {GNN_CKPT}")

    # ---------------- 3) RL 训练 + 结果持久化 ----------------
    print("\n=== 3) Solver-in-the-Loop 强化学习 (REINFORCE, rlimit 代价, hybrid) ===")
    rl_config = RLConfig(
        lr=1e-3, gamma=0.98, entropy_coef=5e-3,
        rlimit_penalty_coef=1.0, use_log_cost=True, reward_scale=0.1,
        max_split_depth=args.split_depth, max_steps=args.max_steps,
        f_sat_mode=F_SAT_MODE,
    )
    rl_trainer = SolverInLoopRLTrainer(policy, rl_config)
    instances = [inst.as_tuple() for inst in train_set]
    history_rl = rl_trainer.train(instances, iterations=args.iters, log=args.rl_log)
    if history_rl:
        print(f"RL 平均 return: 首条 {history_rl[0]['mean_return']:.4f} -> "
              f"末条 {history_rl[-1]['mean_return']:.4f}  (记录 {len(history_rl)} 条)")

    rl_trainer.save(RL_CKPT, history=history_rl)
    save_history(history_rl, HISTORY_JSON)
    print(f"RL 结果已持久化 -> {RL_CKPT}, {HISTORY_JSON}")

    # 演示持久化 round-trip：从磁盘重载权重到全新策略
    reloaded, meta = load_policy(RL_CKPT)
    print(f"重载 checkpoint 成功：kind={meta.get('kind')} "
          f"baseline={meta.get('baseline'):.4f} history_len={len(meta.get('history', []))}")

    # ---------------- 4) 测试：准确率 + 开销/gap 对比 ----------------
    print("\n=== 4) 测试集对比 (Neural vs Baseline, hybrid/anytime) ===")
    acc_neural, acc_base = branch_accuracy(reloaded, test_set)
    print(f"[准确率] 数值 head top-1 与专家一致: Neural={acc_neural:.2f}  Baseline={acc_base:.2f}")

    agg = cost_comparison(reloaded, rl_config, test_set, args.max_steps)
    print("[开销/质量] 测试集平均（rlimit/solve_calls/splits/gap 越小越好，exact 越大越好）:")
    print(f"  {'strategy':<10} {'rlimit':>12} {'solve_calls':>12} {'splits':>8} "
          f"{'gap':>10} {'exact':>8}")
    for name in ("neural", "baseline"):
        a = agg[name]
        print(f"  {name:<10} {a['rlimit']:>12.1f} {a['solve_calls']:>12.2f} "
              f"{a['splits']:>8.2f} {a['gap']:>10.4f} {a['exact']:>8.2f}")

    # LRA 说明：真最优（ground truth）由 z3 原生 Optimize（solve_native）给出。GOMT 的
    # 增量式线性搜索（含 Baseline 的根节点 resolve）对实数用严格 Better 割会触发 z3
    # Optimize 的 epsilon 逼近（分母渐增），在固定步数预算内一般不饱和（anytime），故不
    # 断言精确最优；用最优性 gap 衡量质量。
    gn, gb = agg["neural"]["gap"], agg["baseline"]["gap"]
    rn, rb = agg["neural"]["rlimit"], agg["baseline"]["rlimit"]
    print(f"\n对比（越低越好）: gap Neural={gn:.4f} vs Baseline={gb:.4f}；"
          f"rlimit Neural={rn:.0f} vs Baseline={rb:.0f}")
    verdict = "更优" if (gn <= gb and rn <= rb) else ("互有优劣" if gn <= gb or rn <= rb else "不及")
    print(f"结论：在该步数预算下，学习到的 Neural 策略相对 Baseline {verdict}"
          "（gap 更小且/或 rlimit 更低即更优）。")
    print("\nLRA 完整训练流程验证完成（anytime；ground truth = z3-native Optimize）。")


if __name__ == "__main__":
    main()
