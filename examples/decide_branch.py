"""三臂对比：z3 二进制参考 / VSIDS-decide / learned-decide（未训练 GNN）。

以 ``solve_binary``（z3 可执行文件）给出参考最优值，测量 VSIDS/learned 相对参考的
正确性（match）与 rlimit/conflicts/decisions。为 Phase 2（look-ahead imitation + RL）铺路。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction

import torch

from omt_branching.model.device import gnn_device
from omt_branching.model.inference import InferenceConfig
from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService, ServiceConfig
from omt_branching.solver import (
    # Z3Backend,
    generate_bool_lia_dataset,
    instance_to_smt2,
    solve_binary,
    solve_omt_with_decider,
)
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.policy_decider import PolicyDecider

from tqdm import tqdm

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "decide_branch_dataset")
DEFAULT_TEST_WORKERS = 30


def _json_value(v):
    """把 Fraction / 其它非标量转为 JSON 可序列化形式。"""
    if v is None:
        return None
    if isinstance(v, Fraction):
        return str(v)
    return v


def _instance_manifest_entry(inst: OMTInstance, *, smt2_relpath: str) -> dict:
    return {
        "instance_id": inst.instance_id,
        "theory": inst.theory,
        "family": inst.family,
        "description": inst.description,
        "sense": inst.sense.value,
        "n_vars": len(inst.variables),
        "n_hard": len(inst.hard),
        "obj_coeffs": inst.obj_coeffs,
        "smt2": smt2_relpath,
    }


def save_dataset(
    instances: list[OMTInstance],
    out_dir: str,
    *,
    split: str,
) -> list[dict]:
    """把实例列表落盘：每个实例一个 .smt2，返回 manifest 条目。"""
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    entries: list[dict] = []
    for inst in instances:
        fname = f"{inst.instance_id}.smt2"
        relpath = os.path.join(split, fname).replace("\\", "/")
        with open(os.path.join(split_dir, fname), "w", encoding="utf-8") as f:
            f.write(instance_to_smt2(inst))
        entries.append(_instance_manifest_entry(inst, smt2_relpath=relpath))
    return entries


def _stats_for_json(stats: dict) -> dict:
    return {k: _json_value(v) for k, v in stats.items()}


def _regenerate_test_instance(
    index: int,
    seed: int,
    *,
    hard: bool,
    min_vars: int,
    max_vars: int,
) -> OMTInstance:
    """复现 ``gen(count=index+1, seed=seed)[index]``（供进程池 worker 独立 z3 上下文）。"""
    from omt_branching.solver.instance_gen import bool_lia_instance_at

    return bool_lia_instance_at(
        index, seed, hard=hard, min_vars=min_vars, max_vars=max_vars
    )


def _eval_test_worker(task: tuple) -> dict:
    """ProcessPool worker：按 index/seed 重建实例并跑三臂评测。"""
    (
        index,
        seed,
        hard,
        min_vars,
        max_vars,
        policy_state,
        device,
        z3_path,
        binary_timeout,
        refocus,
    ) = task
    inst = _regenerate_test_instance(
        index, seed, hard=hard, min_vars=min_vars, max_vars=max_vars
    )
    hard, obj, sense = inst.as_tuple()
    ref = solve_binary(inst, z3_path=z3_path, timeout_s=binary_timeout)
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


def _policy_state_cpu(policy: BranchingPolicy) -> dict:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def _run_test_parallel(
    insts: list[OMTInstance],
    policy: BranchingPolicy,
    device: str,
    z3_path: str,
    binary_timeout: int,
    refocus: int,
    workers: int,
    *,
    test_seed: int,
    hard: bool,
    min_vars: int,
    max_vars: int,
) -> list[dict]:
    """并发跑测试集（进程池；每 worker 独立 z3 上下文）。"""
    policy_state = _policy_state_cpu(policy)
    n_workers = max(1, min(workers, len(insts)))
    # 多进程并发时 GNN 推理走 CPU，避免多进程同时占用同一块 GPU
    worker_device = device if n_workers == 1 else "cpu"
    tasks = [
        (
            i,
            test_seed,
            hard,
            min_vars,
            max_vars,
            policy_state,
            worker_device,
            z3_path,
            binary_timeout,
            refocus,
        )
        for i in range(len(insts))
    ]
    by_id: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_eval_test_worker, t): t[0] for t in tasks}
        with tqdm(total=len(insts), desc="test") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                by_id[row["instance_id"]] = row
                pbar.update(1)
    return [by_id[inst.instance_id] for inst in insts]


def main() -> None:
    ap = argparse.ArgumentParser(description="UserPropagator 学习分支三臂对比（binary 参考）")
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
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="数据集落盘目录（manifest + SMT2）；默认 examples/artifacts/decide_branch_dataset",
    )
    ap.add_argument(
        "--no-save-dataset",
        action="store_true",
        help="不保存数据集（仅跑实验并写 results.json）",
    )
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径（默认同 PATH）")
    ap.add_argument(
        "--binary-timeout",
        type=int,
        default=1200,
        help="单实例 z3 二进制超时（秒）",
    )
    ap.add_argument(
        "--test-workers",
        type=int,
        default=DEFAULT_TEST_WORKERS,
        help=f"测试、look-ahead 与 RL collect 并发数（默认 {DEFAULT_TEST_WORKERS}）",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="GNN 设备（默认 cuda 可用则 cuda，否则 cpu）",
    )
    args = ap.parse_args()

    device = args.device or gnn_device()
    print(f"GNN device: {device}")

    z3_path = args.z3_path or shutil.which("z3")
    if not z3_path:
        raise SystemExit("未找到 z3 二进制，请用 --z3-path 指定")

    from omt_branching.solver import generate_hard_bool_lia_dataset

    gen = generate_hard_bool_lia_dataset if args.hard else generate_bool_lia_dataset

    torch.manual_seed(0)
    insts = gen(args.test, seed=99, min_vars=args.min_vars, max_vars=args.max_vars)
    train_insts: list[OMTInstance] = []

    if not args.no_save_dataset:
        os.makedirs(args.dataset_dir, exist_ok=True)
        gen_name = "hard_bool_lia" if args.hard else "bool_lia"
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": gen_name,
            "params": {
                "test": args.test,
                "train": args.train,
                "min_vars": args.min_vars,
                "max_vars": args.max_vars,
                "hard": args.hard,
                "refocus": args.refocus,
                "epochs": args.epochs,
                "rl_iters": args.rl_iters,
            },
            "seeds": {"test": 99, "train": 1},
            "splits": {},
        }
        manifest["splits"]["test"] = save_dataset(insts, args.dataset_dir, split="test")
        print(f"测试集 {len(insts)} 个实例已保存 -> {args.dataset_dir}/test/")

    policy = BranchingPolicy()
    if args.train > 0:
        from omt_branching.model.trainer import ImitationTrainer, TrainConfig
        from omt_branching.solver.training_data import build_lookahead_examples_parallel

        train_insts = gen(args.train, seed=1, min_vars=args.min_vars, max_vars=args.max_vars)
        if not args.no_save_dataset:
            manifest["splits"]["train"] = save_dataset(
                train_insts, args.dataset_dir, split="train"
            )
            print(f"训练集 {len(train_insts)} 个实例已保存 -> {args.dataset_dir}/train/")
        lookahead_workers = args.test_workers
        print(f"look-ahead 标签构建: {args.train} 实例, workers={lookahead_workers}")
        exs = [
            e
            for e in build_lookahead_examples_parallel(
                args.train,
                seed=1,
                hard=args.hard,
                min_vars=args.min_vars,
                max_vars=args.max_vars,
                workers=lookahead_workers,
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
    if args.rl_iters > 0:
        from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

        rl_count = max(args.train, 40)
        rl_train = gen(
            rl_count, seed=1, min_vars=args.min_vars, max_vars=args.max_vars
        )
        rl_workers = args.test_workers
        rlt = DecideRLTrainer(
            policy,
            DecideRLConfig(
                refocus_every=args.refocus,
                device=device,
                workers=rl_workers,
            ),
        )
        print(f"RL collect: {rl_count} 实例 × {args.rl_iters} 轮, workers={rl_workers}")
        h = rlt.train(
            [i.as_tuple() for i in rl_train],
            iterations=args.rl_iters,
            log=False,
            workers=rl_workers,
            collect_seed=1,
            collect_hard=args.hard,
            collect_min_vars=args.min_vars,
            collect_max_vars=args.max_vars,
        )
        if h:
            print(
                f"RL 微调: {len(h)} 步, 末条 reward={h[-1]['reward']:.3f} "
                f"conflicts={h[-1]['conflicts']}, defer_logit={float(rlt.defer_logit):.3f}"
            )

    agg = {
        "binary": {"rlimit": 0.0, "time_ms": 0.0},
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
    per_instance: list[dict] = []
    rows = _run_test_parallel(
        insts,
        policy,
        device,
        z3_path,
        args.binary_timeout,
        args.refocus,
        args.test_workers,
        test_seed=99,
        hard=args.hard,
        min_vars=args.min_vars,
        max_vars=args.max_vars,
    )
    for row in rows:
        ref_val = row["ref_val"]
        ref = row["binary"]
        v = row["vsids"]
        ln = row["learned"]
        agg["binary"]["rlimit"] += ref.get("rlimit") or 0
        agg["binary"]["time_ms"] += ref.get("time_ms") or 0.0
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
        f"=== 三臂对比（{len(insts)} 实例；binary 为参考；match=1 为与 binary 最优值一致）==="
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
    for key in agg["binary"].keys():
        agg["binary"][key] /= n
    for key in agg["vsids"].keys():
        agg["vsids"][key] /= n
    for key in agg["learned"].keys():
        agg["learned"][key] /= n

    os.makedirs(ARTIFACTS, exist_ok=True)
    results = {
        "reference": "binary",
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

    if not args.no_save_dataset:
        manifest_path = os.path.join(args.dataset_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        print(f"数据集清单已保存 -> {manifest_path}")


if __name__ == "__main__":
    main()
