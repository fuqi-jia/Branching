"""完整训练流程示例：生成实例 -> GNN 监督训练 -> RL 训练 -> 测试对比 Baseline。

展示的完整闭环：

1. **生成实例**：用 ``instance_gen`` 合成一批有界 OMT(LIA) 训练/测试实例。
2. **GNN 监督训练**：用启发式专家标签做 imitation 冷启动，并把权重持久化到磁盘。
3. **RL 训练**：在真实 z3 GOMT 回路里 REINFORCE 微调
   —— 单步 reward = incumbent 提升；终局 penalty = z3 **rlimit count** 增长（替代 wall-clock）。
   训练结果（权重 + baseline + 历史）持久化，并演示从磁盘重载。
4. **测试**：在测试集上从 **准确率**（与专家一致的 top-1 分支比例）与 **求解开销**
   （rlimit / solve_calls / splits）两方面对比 Neural 策略与 Baseline 策略，并断言
   所有 optimum 与 z3-native 一致（GOMT soundness）。

运行::

    python -m examples.rl_demo
"""

from __future__ import annotations

import os

import torch

from omt_branching.model.persistence import load_policy, save_history, save_policy
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver import (
    BaselineStrategy,
    RLConfig,
    RLRecordingStrategy,
    Sense,
    SolverInLoopRLTrainer,
    baseline_numeric_choice,
    build_imitation_examples,
    generate_dataset,
    oracle_numeric_choice,
    policy_numeric_choice,
    solve_and_measure,
    solve_native,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
GNN_CKPT = os.path.join(ARTIFACTS, "gnn_imitation.pt")
RL_CKPT = os.path.join(ARTIFACTS, "rl_policy.pt")
HISTORY_JSON = os.path.join(ARTIFACTS, "rl_history.json")

MAX_STEPS = 3000


def branch_accuracy(policy, instances) -> tuple[float, float]:
    """返回 (Neural 与专家一致比例, Baseline 与专家一致比例)。"""
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


def cost_comparison(policy, rl_config, instances):
    """在测试集上对比 Neural 与 Baseline 的求解开销与正确性。"""
    agg = {"neural": _fresh(), "baseline": _fresh()}
    for inst in instances:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)

        neural = solve_and_measure(
            hard, obj, sense,
            lambda p: RLRecordingStrategy(p, policy, rl_config, sample=False),
            max_steps=MAX_STEPS,
        )
        base = solve_and_measure(
            hard, obj, sense, lambda p: BaselineStrategy(p), max_steps=MAX_STEPS)

        _accumulate(agg["neural"], neural, native)
        _accumulate(agg["baseline"], base, native)
    n = max(1, len(instances))
    for k in agg:
        for m in ("rlimit", "solve_calls", "splits"):
            agg[k][m] /= n
        agg[k]["correct"] /= n
    return agg


def _fresh():
    return {"rlimit": 0.0, "solve_calls": 0.0, "splits": 0.0, "correct": 0.0}


def _accumulate(acc, res, native):
    acc["rlimit"] += res["rlimit"]
    acc["solve_calls"] += res["solve_calls"]
    acc["splits"] += res["splits"]
    acc["correct"] += 1.0 if (res["value"] == native and res["optimal"]) else 0.0


def main() -> None:
    torch.manual_seed(0)
    os.makedirs(ARTIFACTS, exist_ok=True)

    # ---------------- 1) 生成实例 ----------------
    print("=== 1) 生成 OMT(LIA) 实例 ===")
    train_set = generate_dataset(6, seed=1, n_vars=3, n_constraints=3, ub=6)
    test_set = generate_dataset(10, seed=99, n_vars=3, n_constraints=3, ub=6)
    print(f"训练集 {len(train_set)} 个，测试集 {len(test_set)} 个")
    for inst in test_set[:4]:
        print(f"  {inst.instance_id}: obj_coeffs={inst.obj_coeffs} "
              f"oracle={oracle_numeric_choice(inst)}")
    print("  ...")

    # ---------------- 2) GNN 监督训练 + 保权 ----------------
    print("\n=== 2) GNN 监督训练 (imitation 冷启动) ===")
    policy = BranchingPolicy()
    acc_before, base_acc = branch_accuracy(policy, test_set)
    print(f"训练前 分支准确率: Neural={acc_before:.2f}  Baseline={base_acc:.2f}")

    examples = build_imitation_examples(train_set)
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    history_imit = trainer.fit(examples, epochs=40)
    print(f"imitation loss: 首轮 {history_imit[0]['loss']:.4f} -> "
          f"末轮 {history_imit[-1]['loss']:.4f}")

    acc_after, _ = branch_accuracy(policy, test_set)
    print(f"训练后 分支准确率: Neural={acc_after:.2f}  Baseline={base_acc:.2f}")

    save_policy(policy, GNN_CKPT, meta={"stage": "imitation", "epochs": 40})
    print(f"GNN 权重已保存 -> {GNN_CKPT}")

    # ---------------- 3) RL 训练 + 结果持久化 ----------------
    print("\n=== 3) Solver-in-the-Loop 强化学习 (REINFORCE, rlimit 代价) ===")
    rl_config = RLConfig(
        lr=1e-3, gamma=0.98, entropy_coef=5e-3,
        rlimit_penalty_coef=1.0, use_log_cost=True, reward_scale=0.1,
        max_split_depth=5, max_steps=MAX_STEPS,
    )
    rl_trainer = SolverInLoopRLTrainer(policy, rl_config)
    instances = [inst.as_tuple() for inst in train_set]
    history_rl = rl_trainer.train(instances, iterations=3, log=True)

    rl_trainer.save(RL_CKPT, history=history_rl)
    save_history(history_rl, HISTORY_JSON)
    print(f"RL 结果已持久化 -> {RL_CKPT}, {HISTORY_JSON}")

    # 演示持久化 round-trip：从磁盘重载权重到全新策略
    reloaded, meta = load_policy(RL_CKPT)
    print(f"重载 checkpoint 成功：stage={meta.get('kind')} "
          f"baseline={meta.get('baseline'):.4f} history_len={len(meta.get('history', []))}")

    # ---------------- 4) 测试：准确率 + 开销 对比 Baseline ----------------
    print("\n=== 4) 测试集对比 (Neural vs Baseline) ===")
    acc_neural, acc_base = branch_accuracy(reloaded, test_set)
    print(f"[准确率] 与专家一致的 top-1 分支: Neural={acc_neural:.2f}  Baseline={acc_base:.2f}")

    agg = cost_comparison(reloaded, rl_config, test_set)
    print("[开销] 测试集平均（越小越好）:")
    print(f"  {'strategy':<10} {'rlimit':>10} {'solve_calls':>12} {'splits':>8} {'correct':>8}")
    for name in ("neural", "baseline"):
        a = agg[name]
        print(f"  {name:<10} {a['rlimit']:>10.1f} {a['solve_calls']:>12.2f} "
              f"{a['splits']:>8.2f} {a['correct']:>8.2f}")

    r_n, r_b = agg["neural"]["rlimit"], agg["baseline"]["rlimit"]
    if r_b > 0:
        print(f"\nNeural 相对 Baseline 的 rlimit 开销比: {r_n / r_b:.2f}x")
    assert agg["neural"]["correct"] == 1.0 == agg["baseline"]["correct"], \
        "存在实例未求得正确 optimum！(soundness)"
    print("soundness 校验通过：Neural 与 Baseline 在测试集上均求得正确 optimum。")
    print("\n完整训练流程验证完成。")


if __name__ == "__main__":
    main()
