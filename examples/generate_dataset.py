"""生成 decide_branch 用布尔结构 OMT 数据集并落盘。

默认目录：``examples/artifacts/dataset/``（``train/``、``test/``、``eval/``、``manifest.json``）。

运行::

    python -m examples.generate_dataset
    python -m examples.generate_dataset --test 20 --train 40 --eval 10 --min-vars 4 --max-vars 5
    python -m examples.generate_dataset --hard --force
    # 向已有数据集额外生成 / 覆盖验证集（不改动 train/test）:
    python -m examples.generate_dataset --append-eval --eval 10 --min-vars 4 --max-vars 5

分支敏感实例（CSL≈VSIDS 且 oracle 可显著降 rlimit）请用探针筛选落盘::

    python -m examples.probe_branch_focus --candidates 40 --need 10 --save
    python -m examples.probe_branch_focus --seed 100 --split train --save
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from omt_branching.solver import (
    generate_bool_lia_dataset,
    generate_hard_bool_lia_dataset,
    rebuild_manifest,
    save_dataset,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")


def _append_eval(args) -> None:
    """仅向已有数据集写入 eval 划分并更新 manifest。"""
    root = Path(args.dataset_dir)
    if not root.is_dir():
        raise SystemExit(
            f"数据集目录不存在: {root}\n"
            f"请先运行完整生成，或检查 --dataset-dir"
        )
    has_base = any(root.glob("test/*.smt2")) or any(root.glob("train/*.smt2"))
    if not has_base:
        raise SystemExit(
            f"目录中无 train/test 实例，无法 --append-eval: {root}\n"
            f"请先: python -m examples.generate_dataset --dataset-dir {args.dataset_dir}"
        )
    if args.eval <= 0:
        raise SystemExit("--append-eval 需要 --eval N（N>0）")

    gen = generate_hard_bool_lia_dataset if args.hard else generate_bool_lia_dataset
    gen_name = "hard_bool_lia" if args.hard else "bool_lia"
    eval_insts = gen(
        args.eval,
        seed=args.eval_seed,
        min_vars=args.min_vars,
        max_vars=args.max_vars,
    )
    eval_entries = save_dataset(eval_insts, args.dataset_dir, split="eval")
    print(f"验证集 {len(eval_insts)} -> {root / 'eval'}/")

    # 按磁盘重建，保留 meta，再写入本次 eval 相关 params/seeds
    manifest = rebuild_manifest(args.dataset_dir, preserve_meta=True)
    params = dict(manifest.get("params") or {})
    params["eval"] = len(eval_entries)
    params["eval_min_vars"] = args.min_vars
    params["eval_max_vars"] = args.max_vars
    params["eval_hard"] = args.hard
    seeds = dict(manifest.get("seeds") or {})
    seeds["eval"] = args.eval_seed
    # 若旧 generator 缺失则补上本次生成器名（不覆盖已有）
    if not manifest.get("generator") or manifest.get("generator") == "imported":
        manifest["generator"] = gen_name
    manifest["params"] = params
    manifest["seeds"] = seeds
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print(f"manifest 已更新 -> {manifest_path}")
    print(
        "下一步（ref 缓存 eval）:\n"
        f"  python -m examples.solve_dataset_binary --dataset-dir {args.dataset_dir} "
        f"--split eval"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 decide_branch 数据集（SMT2 + manifest）")
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"落盘目录（默认 {DEFAULT_DATASET_DIR}）",
    )
    ap.add_argument("--test", type=int, default=20, help="测试集实例数")
    ap.add_argument("--train", type=int, default=40, help="训练集实例数（0=不生成 train）")
    ap.add_argument("--eval", type=int, default=0, help="验证集实例数（0=不生成 eval）")
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--hard", action="store_true", help="用更难实例生成器")
    ap.add_argument(
        "--force",
        action="store_true",
        help="目录已有数据时仍覆盖生成（完整生成模式）",
    )
    ap.add_argument(
        "--append-eval",
        action="store_true",
        help="仅向已有数据集追加/覆盖 eval 划分，不改动 train/test",
    )
    ap.add_argument("--test-seed", type=int, default=99)
    ap.add_argument("--train-seed", type=int, default=1)
    ap.add_argument("--eval-seed", type=int, default=42)
    args = ap.parse_args()

    if args.append_eval:
        _append_eval(args)
        return

    root = Path(args.dataset_dir)
    if root.exists() and any(root.rglob("*.smt2")) and not args.force:
        raise SystemExit(
            f"目录已有 .smt2：{root}\n"
            f"如需覆盖请加 --force；若只追加验证集请用:\n"
            f"  python -m examples.generate_dataset --append-eval --eval N "
            f"--min-vars ... --max-vars ..."
        )

    gen = generate_hard_bool_lia_dataset if args.hard else generate_bool_lia_dataset
    gen_name = "hard_bool_lia" if args.hard else "bool_lia"
    os.makedirs(args.dataset_dir, exist_ok=True)

    test_insts = gen(
        args.test,
        seed=args.test_seed,
        min_vars=args.min_vars,
        max_vars=args.max_vars,
    )
    test_entries = save_dataset(test_insts, args.dataset_dir, split="test")
    print(f"测试集 {len(test_insts)} -> {root / 'test'}/")

    train_entries: list[dict] = []
    if args.train > 0:
        train_insts = gen(
            args.train,
            seed=args.train_seed,
            min_vars=args.min_vars,
            max_vars=args.max_vars,
        )
        train_entries = save_dataset(train_insts, args.dataset_dir, split="train")
        print(f"训练集 {len(train_insts)} -> {root / 'train'}/")

    eval_entries: list[dict] = []
    if args.eval > 0:
        eval_insts = gen(
            args.eval,
            seed=args.eval_seed,
            min_vars=args.min_vars,
            max_vars=args.max_vars,
        )
        eval_entries = save_dataset(eval_insts, args.dataset_dir, split="eval")
        print(f"验证集 {len(eval_insts)} -> {root / 'eval'}/")

    seeds = {"test": args.test_seed, "train": args.train_seed}
    if eval_entries:
        seeds["eval"] = args.eval_seed
    splits: dict = {
        "test": test_entries,
        **({"train": train_entries} if train_entries else {}),
        **({"eval": eval_entries} if eval_entries else {}),
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": gen_name,
        "params": {
            "test": args.test,
            "train": args.train,
            "eval": args.eval,
            "min_vars": args.min_vars,
            "max_vars": args.max_vars,
            "hard": args.hard,
        },
        "seeds": seeds,
        "splits": splits,
    }
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print(f"manifest -> {manifest_path}")
    print(
        "下一步（ref 缓存）:\n"
        f"  python -m examples.solve_dataset_binary --dataset-dir {args.dataset_dir}"
    )


if __name__ == "__main__":
    main()
