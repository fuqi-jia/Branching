"""三臂对比：z3 二进制参考 / VSIDS-decide / learned-decide。

以 ``examples/artifacts/dataset`` 中缓存的 ``solve_binary`` 结果为参考最优值，测量
VSIDS/learned 相对参考的正确性（match）与 rlimit/conflicts/decisions。

数据集须事先由 ``python -m examples.generate_dataset`` 生成；本脚本只检查/重建
``manifest.json``，不生成实例。binary 参考解须由
``python -m examples.solve_dataset_binary`` 写入 ``binary/<split>/<id>.json``。

RL 微调默认在 ``eval/`` 验证集上按 ``mean_reward`` 早停：``--rl-iters N`` 最多 N 轮
（收敛可提前结束）；``--rl-iters -1`` 训到收敛为止。验证集可用::

    python -m examples.generate_dataset --append-eval --eval 10 --min-vars 4 --max-vars 5
    python -m examples.solve_dataset_binary --split eval
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import torch

from omt_branching.model.device import gnn_device
from omt_branching.model.inference import InferenceConfig
from omt_branching.model.persistence import save_history
from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService, ServiceConfig
from omt_branching.solver.rl_decide import (
    DEFAULT_RL_COLLECT_WORKERS,
    DecideRLConfig,
    DecideRLTrainer,
    EarlyStopConfig,
    decide_rl_reward,
    effective_rl_workers,
)
from omt_branching.solver import (
    load_dataset,
    list_split_entries,
    manifest_mismatches,
    rebuild_manifest,
    solve_omt_with_decider,
)
from omt_branching.solver.binary_results import (
    binary_rlimit,
    binary_value,
    load_binary_result,
    missing_binary_ids,
)
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.policy_decider import PolicyDecider

from tqdm import tqdm

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_CKPT_DIR = os.path.join(ARTIFACTS, "rl_checkpoints")
DEFAULT_TEST_WORKERS = 30


def _json_value(v):
    """把 Fraction / 其它非标量转为 JSON 可序列化形式。"""
    if v is None:
        return None
    if isinstance(v, Fraction):
        return str(v)
    return v


def _stats_for_json(stats: dict) -> dict:
    return {k: _json_value(v) for k, v in stats.items()}


def _sync_manifest_if_needed(dataset_dir: str) -> None:
    """若 manifest 与磁盘 .smt2 不一致，按磁盘重建。"""
    issues = manifest_mismatches(dataset_dir)
    if not issues:
        return
    print("检测到 manifest 与磁盘 .smt2 不一致，按磁盘重建 manifest：")
    for msg in issues:
        print(f"  - {msg}")
    rebuild_manifest(dataset_dir, preserve_meta=True)
    print(f"已重建 -> {Path(dataset_dir) / 'manifest.json'}")


def _require_dataset(dataset_dir: str) -> None:
    root = Path(dataset_dir)
    has_smt2 = (
        any(root.glob("test/*.smt2"))
        or any(root.glob("train/*.smt2"))
        or any(root.glob("eval/*.smt2"))
    )
    if not root.is_dir() or not has_smt2:
        raise SystemExit(
            f"未找到数据集: {dataset_dir}\n"
            f"请先运行: python -m examples.generate_dataset"
        )


def _smt2_abs_paths(dataset_dir: str, entries: list[dict]) -> list[str]:
    root = Path(dataset_dir)
    return [str(root / e["smt2"]) for e in entries]


def _require_binary_cache(dataset_dir: str, split: str, entries: list[dict]) -> None:
    missing = missing_binary_ids(dataset_dir, entries, split=split)
    if missing:
        preview = ", ".join(missing[:5])
        more = f" 等共 {len(missing)} 个" if len(missing) > 5 else ""
        raise SystemExit(
            f"划分 {split} 缺少 binary 缓存（如 {preview}{more}）。\n"
            f"请先运行: python -m examples.solve_dataset_binary "
            f"--dataset-dir {dataset_dir}"
        )


def _eval_test_worker(task: tuple) -> dict:
    """ProcessPool worker：从 .smt2 加载实例，binary 用缓存，跑 VSIDS/learned。"""
    (
        smt2_path,
        instance_id,
        binary_result,
        policy_state,
        device,
        refocus,
    ) = task
    from omt_branching.solver.decide_omt import smt2_to_instance

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    hard, obj, sense = inst.as_tuple()
    ref = binary_result
    ref_val = ref.get("value")
    v = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
    policy = BranchingPolicy()
    policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()
    svc = BranchingPolicyService(
        policy=policy,
        config=ServiceConfig(inference=InferenceConfig(device=device)),
    )
    ln = solve_omt_with_decider(
        hard,
        obj,
        sense,
        decider_factory=lambda a: PolicyDecider(svc, a, refocus),
    )
    return {
        "instance_id": inst.instance_id,
        "ref_val": ref_val,
        "binary": ref,
        "vsids": v,
        "learned": ln,
    }


def _eval_val_worker(task: tuple) -> dict:
    """验证集 worker：只跑 learned 臂，返回 reward / weighted rlimit / match。"""
    (
        smt2_path,
        instance_id,
        binary_result,
        policy_state,
        device,
        refocus,
    ) = task
    from omt_branching.solver.decide_omt import smt2_to_instance

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    hard, obj, sense = inst.as_tuple()
    ref = binary_result
    ref_val = ref.get("value")
    ref_rl = ref.get("rlimit")
    policy = BranchingPolicy()
    policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()
    svc = BranchingPolicyService(
        policy=policy,
        config=ServiceConfig(inference=InferenceConfig(device=device)),
    )
    ln = solve_omt_with_decider(
        hard,
        obj,
        sense,
        decider_factory=lambda a: PolicyDecider(svc, a, refocus),
        ref_rlimit=ref_rl,
    )
    reward = decide_rl_reward(ln, ref_val, ref_rl)
    return {
        "instance_id": inst.instance_id,
        "reward": reward,
        "weighted_rlimit": ln.get("weighted rlimit"),
        "rlimit": ln.get("rlimit"),
        "match": 1.0 if ln.get("value") == ref_val else 0.0,
    }


def _policy_state_cpu(policy: BranchingPolicy) -> dict:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def _run_test_parallel(
    entries: list[dict],
    dataset_dir: str,
    policy: BranchingPolicy,
    device: str,
    refocus: int,
    workers: int,
    *,
    split: str = "test",
) -> list[dict]:
    """并发跑测试集（进程池；binary 读缓存；每 worker 从 smt2 加载）。"""
    policy_state = _policy_state_cpu(policy)
    n_workers = max(1, min(workers, len(entries)))
    worker_device = device if n_workers == 1 else "cpu"
    root = Path(dataset_dir)
    tasks = []
    for e in entries:
        iid = e["instance_id"]
        cached = load_binary_result(dataset_dir, iid, split=split)
        if cached is None:
            raise RuntimeError(f"缺少 binary 缓存 ({split}): {iid}")
        tasks.append((
            str(root / e["smt2"]),
            iid,
            cached,
            policy_state,
            worker_device,
            refocus,
        ))
    by_id: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_test_worker, t): t[1] for t in tasks}
        with tqdm(total=len(entries), desc=split) as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                by_id[row["instance_id"]] = row
                pbar.update(1)
    return [by_id[e["instance_id"]] for e in entries]


def _run_val_parallel(
    entries: list[dict],
    dataset_dir: str,
    policy: BranchingPolicy,
    device: str,
    refocus: int,
    workers: int,
    *,
    split: str = "eval",
) -> dict:
    """在验证集上评估 learned 臂，返回聚合指标（供早停）。"""
    if not entries:
        raise ValueError("验证集条目为空")
    policy_state = _policy_state_cpu(policy)
    n_workers = max(1, min(workers, len(entries)))
    worker_device = device if n_workers == 1 else "cpu"
    root = Path(dataset_dir)
    tasks = []
    for e in entries:
        iid = e["instance_id"]
        cached = load_binary_result(dataset_dir, iid, split=split)
        if cached is None:
            raise RuntimeError(f"缺少 binary 缓存 ({split}): {iid}")
        tasks.append((
            str(root / e["smt2"]),
            iid,
            cached,
            policy_state,
            worker_device,
            refocus,
        ))
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_eval_val_worker, t) for t in tasks]
        with tqdm(total=len(entries), desc=f"val/{split}") as pbar:
            for fut in as_completed(futures):
                rows.append(fut.result())
                pbar.update(1)
    n = max(1, len(rows))
    mean_reward = sum(float(r["reward"]) for r in rows) / n
    wr = [
        float(r["weighted_rlimit"])
        for r in rows
        if r.get("weighted_rlimit") is not None
    ]
    mean_weighted = sum(wr) / len(wr) if wr else float("inf")
    match_rate = sum(float(r["match"]) for r in rows) / n
    return {
        "mean_reward": mean_reward,
        "mean_weighted_rlimit": mean_weighted,
        "match_rate": match_rate,
        "n": len(rows),
    }


def _load_train_split(dataset_dir: str) -> tuple[list[OMTInstance], list[dict]]:
    train_entries = list_split_entries(dataset_dir, "train")
    if not train_entries:
        raise SystemExit(
            f"需要 train 划分但数据集中没有: {dataset_dir}\n"
            f"请用 python -m examples.generate_dataset --train N 生成"
        )
    train_insts = load_dataset(dataset_dir, split="train")
    return train_insts, train_entries


def main() -> None:
    ap = argparse.ArgumentParser(
        description="UserPropagator 学习分支三臂对比（使用 artifacts/dataset）"
    )
    ap.add_argument("--refocus", type=int, default=50)
    ap.add_argument(
        "--imitation",
        action="store_true",
        help="在 train 划分上做 look-ahead imitation",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument(
        "--rl-iters",
        type=int,
        default=0,
        help="RL 微调轮数：0=不做；>0=最多该轮（收敛可早停）；-1=训到收敛为止",
    )
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径（默认同 PATH）")
    ap.add_argument(
        "--test-workers",
        type=int,
        default=DEFAULT_TEST_WORKERS,
        help=f"测试 / 验证 / look-ahead 标签构建并发数（默认 {DEFAULT_TEST_WORKERS}）",
    )
    ap.add_argument(
        "--rl-workers",
        type=int,
        default=DEFAULT_RL_COLLECT_WORKERS,
        help="RL collect 进程数（默认 4；实例数<8 时自动串行；与 --test-workers 独立）",
    )
    ap.add_argument(
        "--ckpt-dir",
        default=DEFAULT_CKPT_DIR,
        help="RL 中间 checkpoint 目录",
    )
    ap.add_argument(
        "--ckpt-every",
        type=int,
        default=1,
        help="每隔多少 RL 轮保存一次中间权重（默认每轮）",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="验证集指标连续无相对提升的轮数达到此值则判定收敛（默认 3）",
    )
    ap.add_argument(
        "--early-stop-tol",
        type=float,
        default=0.02,
        help="相对提升阈值（默认 0.02=2%%）；不足则计入 patience",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="每隔多少 RL 轮在 eval 划分上验证一次（默认每轮）",
    )
    ap.add_argument(
        "--early-stop-max-iters",
        type=int,
        default=10_000,
        help="rl-iters=-1 时的安全上限轮数（默认 10000）",
    )
    ap.add_argument(
        "--no-early-stop",
        action="store_true",
        help="禁用验证集早停（rl-iters=-1 时不可用）",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="GNN 设备（默认 cuda 可用则 cuda，否则 cpu）",
    )
    args = ap.parse_args()
    if args.rl_iters < -1:
        raise SystemExit("--rl-iters 须为 -1、0 或正整数")
    if args.rl_iters == -1 and args.no_early_stop:
        raise SystemExit("--rl-iters=-1 必须启用早停（勿加 --no-early-stop）")

    dataset_dir = DEFAULT_DATASET_DIR
    device = args.device or gnn_device()
    print(f"GNN device: {device}")
    print(f"数据集目录: {dataset_dir}")

    z3_path = args.z3_path or shutil.which("z3")
    if not z3_path:
        raise SystemExit("未找到 z3 二进制，请用 --z3-path 指定")

    _require_dataset(dataset_dir)
    _sync_manifest_if_needed(dataset_dir)

    test_entries = list_split_entries(dataset_dir, "test")
    if not test_entries:
        raise SystemExit(f"数据集无 test 划分: {dataset_dir}")
    insts = load_dataset(dataset_dir, split="test")
    print(f"测试集 {len(insts)} 个实例已从磁盘加载")

    train_insts: list[OMTInstance] = []
    train_entries: list[dict] = []
    if args.imitation or args.rl_iters != 0:
        train_insts, train_entries = _load_train_split(dataset_dir)
        print(f"训练集 {len(train_insts)} 个实例已从磁盘加载")

    eval_entries: list[dict] = []
    need_eval = args.rl_iters != 0 and not args.no_early_stop
    if need_eval:
        eval_entries = list_split_entries(dataset_dir, "eval")
        if not eval_entries:
            raise SystemExit(
                f"RL 早停需要 eval 验证集，但数据集中没有: {dataset_dir}\n"
                f"请先运行:\n"
                f"  python -m examples.generate_dataset --append-eval --eval N "
                f"--min-vars ... --max-vars ...\n"
                f"  python -m examples.solve_dataset_binary --split eval"
            )
        print(f"验证集 {len(eval_entries)} 个实例已从磁盘加载")

    torch.manual_seed(0)
    _require_binary_cache(dataset_dir, "test", test_entries)
    if need_eval:
        _require_binary_cache(dataset_dir, "eval", eval_entries)

    policy = BranchingPolicy()
    if args.imitation:
        from omt_branching.model.trainer import ImitationTrainer, TrainConfig
        from omt_branching.solver.training_data import (
            build_lookahead_examples_from_smt2_parallel,
        )

        lookahead_workers = args.test_workers
        paths = _smt2_abs_paths(dataset_dir, train_entries)
        ids = [e["instance_id"] for e in train_entries]
        print(
            f"look-ahead 标签构建: {len(paths)} 实例(from smt2), "
            f"workers={lookahead_workers}（优先读 lookahead/ 缓存）"
        )
        exs = [
            e
            for e in build_lookahead_examples_from_smt2_parallel(
                paths,
                instance_ids=ids,
                workers=lookahead_workers,
                dataset_dir=dataset_dir,
                split="train",
                use_cache=True,
            )
            if e.bool_target_scores
        ]
        hist = ImitationTrainer(policy, TrainConfig(lr=5e-3, device=device)).fit(
            exs, epochs=args.epochs
        )
        print(
            f"look-ahead imitation: {len(exs)} 样本, branch loss "
            f"{hist[0].get('branch', 0):.3f} -> {hist[-1].get('branch', 0):.3f}"
        )

    if args.rl_iters != 0:
        missing = missing_binary_ids(dataset_dir, train_entries, split="train")
        if missing:
            preview = ", ".join(missing[:5])
            more = f" 等共 {len(missing)} 个" if len(missing) > 5 else ""
            raise SystemExit(
                f"RL 需要 train 划分的 binary 缓存（缺 {preview}{more}）。\n"
                f"请先运行: python -m examples.solve_dataset_binary "
                f"--dataset-dir {dataset_dir} --split train"
            )
        rl_train = train_insts
        rl_paths = _smt2_abs_paths(dataset_dir, train_entries)
        rl_ids = [e["instance_id"] for e in train_entries]
        rl_ref_values = [
            binary_value(dataset_dir, iid, split="train") for iid in rl_ids
        ]
        rl_ref_rlimits = [
            binary_rlimit(dataset_dir, iid, split="train") for iid in rl_ids
        ]

        rl_workers = effective_rl_workers(len(rl_train), args.rl_workers)
        rlt = DecideRLTrainer(
            policy,
            DecideRLConfig(
                refocus_every=args.refocus,
                device=device,
                workers=rl_workers,
            ),
        )
        mode = f"并行×{rl_workers}" if rl_workers > 1 else "串行(GPU collect)"
        iters_desc = (
            "直到收敛"
            if args.rl_iters == -1
            else f"最多 {args.rl_iters} 轮"
        )
        print(
            f"RL collect: {len(rl_train)} 实例 × {iters_desc}, {mode} "
            f"(请求 workers={args.rl_workers})；collect 用 CPU，update 用 {device}；"
            f"reward 使用 binary ref_value/ref_rlimit"
        )
        print(f"RL checkpoints -> {args.ckpt_dir}/ (every {args.ckpt_every})")

        early_cfg: EarlyStopConfig | None = None
        eval_cb = None
        if need_eval:
            early_cfg = EarlyStopConfig(
                patience=args.early_stop_patience,
                tol=args.early_stop_tol,
                maximize=True,
                metric_key="mean_reward",
                min_iters=1,
                max_iters=args.early_stop_max_iters,
                eval_every=max(1, args.eval_every),
            )
            print(
                f"早停: eval={len(eval_entries)} 实例, metric=mean_reward↑, "
                f"patience={early_cfg.patience}, tol={early_cfg.tol}, "
                f"eval_every={early_cfg.eval_every}"
                + (
                    f", max_iters={early_cfg.max_iters}"
                    if args.rl_iters == -1
                    else ""
                )
            )

            def eval_cb(finished_iters: int, trainer: DecideRLTrainer) -> dict:
                metrics = _run_val_parallel(
                    eval_entries,
                    dataset_dir,
                    trainer.policy,
                    device,
                    args.refocus,
                    args.test_workers,
                    split="eval",
                )
                print(
                    f"[val it={finished_iters}] mean_reward={metrics['mean_reward']:.4f} "
                    f"weighted_rlimit={metrics['mean_weighted_rlimit']:.1f} "
                    f"match={metrics['match_rate']:.3f}"
                )
                return metrics

        h = rlt.train(
            [i.as_tuple() for i in rl_train],
            iterations=args.rl_iters,
            log=False,
            workers=rl_workers,
            collect_seed=1,
            smt2_paths=rl_paths,
            instance_ids=rl_ids,
            ref_values=rl_ref_values,
            ref_rlimits=rl_ref_rlimits,
            checkpoint_dir=args.ckpt_dir,
            checkpoint_every=args.ckpt_every,
            eval_callback=eval_cb,
            early_stop=early_cfg,
        )
        end_meta = h[-1] if h and h[-1].get("event") == "train_end" else {}
        finished = end_meta.get("finished_iters", args.rl_iters)
        final_path = os.path.join(ARTIFACTS, "rl_decide_policy.pt")
        rlt.save_checkpoint(
            final_path,
            meta={
                "iter": finished,
                "final": True,
                "stop_reason": end_meta.get("stop_reason"),
                "best_metric": end_meta.get("best_metric"),
            },
        )
        hist_path = os.path.join(ARTIFACTS, "rl_decide_history.json")
        save_history(h, hist_path)
        print(f"RL 最终权重 -> {final_path}")
        print(f"RL 历史 -> {hist_path}")
        if end_meta:
            print(
                f"RL 结束: reason={end_meta.get('stop_reason')}, "
                f"iters={finished}, best_mean_reward={end_meta.get('best_metric')}, "
                f"defer_logit={float(rlt.defer_logit):.3f}"
            )
        last_step = next(
            (x for x in reversed(h) if "reward" in x and "event" not in x),
            None,
        )
        if last_step is not None:
            print(
                f"RL 末条 step: reward={last_step['reward']:.3f} "
                f"conflicts={last_step.get('conflicts')}, "
                f"steps={last_step.get('steps')}"
            )

    agg = {
        "binary": {
            "rlimit": 0.0,
            "time_ms": 0.0,
            "conflicts": 0.0,
        },
        "vsids": {
            "rlimit": 0.0,
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
    per_instance: list[dict] = []
    rows = _run_test_parallel(
        test_entries,
        dataset_dir,
        policy,
        device,
        args.refocus,
        args.test_workers,
    )
    for row in rows:
        ref_val = row["ref_val"]
        ref = row["binary"]
        v = row["vsids"]
        ln = row["learned"]
        for key in agg["binary"].keys():
            agg["binary"][key] += ref.get(key) or 0
        for key in v.keys():
            if key not in agg["vsids"]:
                continue
            agg["vsids"][key] += v[key]
        agg["vsids"]["match"] += 1.0 if v["value"] == ref_val else 0.0
        for key in ln.keys():
            if key not in agg["learned"]:
                continue
            agg["learned"][key] += ln[key]
        agg["learned"]["match"] += 1.0 if ln["value"] == ref_val else 0.0
        per_instance.append({
            "instance_id": row["instance_id"],
            "binary": _stats_for_json(ref),
            "vsids": _stats_for_json(v),
            "learned": _stats_for_json(ln),
        })

    n = max(1, len(insts))
    print(
        f"=== 三臂对比（{len(insts)} 实例；binary 为缓存参考；"
        f"match=1 为与 binary 最优值一致）==="
    )
    for key in agg["binary"].keys():
        agg["binary"][key] /= n
    for key in agg["vsids"].keys():
        agg["vsids"][key] /= n
    for key in agg["learned"].keys():
        agg["learned"][key] /= n

    os.makedirs(ARTIFACTS, exist_ok=True)
    results = {
        "reference": "binary_cache",
        "dataset_dir": dataset_dir,
        "summary": agg,
        "n_instances": len(insts),
        "z3_path": z3_path,
        "device": device,
        "test_workers": args.test_workers,
        "per_instance": per_instance,
    }
    results_path = os.path.join(ARTIFACTS, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"实验汇总已保存 -> {results_path}")


if __name__ == "__main__":
    main()
