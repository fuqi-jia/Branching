"""生成 decide_branch 用布尔结构 OMT 数据集并落盘。

默认目录：``examples/artifacts/dataset/``（``train/``、``test/``、``manifest.json``）。

运行::

    python -m examples.generate_dataset
    python -m examples.generate_dataset --test 20 --train 40 --min-vars 4 --max-vars 5
    python -m examples.generate_dataset --hard --force
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
    save_dataset,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 decide_branch 数据集（SMT2 + manifest）")
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"落盘目录（默认 {DEFAULT_DATASET_DIR}）",
    )
    ap.add_argument("--test", type=int, default=20, help="测试集实例数")
    ap.add_argument("--train", type=int, default=40, help="训练集实例数（0=不生成 train）")
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--hard", action="store_true", help="用更难实例生成器")
    ap.add_argument(
        "--force",
        action="store_true",
        help="目录已有数据时仍覆盖生成",
    )
    ap.add_argument("--test-seed", type=int, default=99)
    ap.add_argument("--train-seed", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    if root.exists() and any(root.rglob("*.smt2")) and not args.force:
        raise SystemExit(
            f"目录已有 .smt2：{root}\n"
            f"如需覆盖请加 --force，或另指定 --dataset-dir"
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

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": gen_name,
        "params": {
            "test": args.test,
            "train": args.train,
            "min_vars": args.min_vars,
            "max_vars": args.max_vars,
            "hard": args.hard,
        },
        "seeds": {"test": args.test_seed, "train": args.train_seed},
        "splits": {
            "test": test_entries,
            **({"train": train_entries} if train_entries else {}),
        },
    }
    manifest_path = root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    print(f"manifest -> {manifest_path}")
    print(
        "下一步（binary 缓存）:\n"
        f"  python -m examples.solve_dataset_binary --dataset-dir {args.dataset_dir}"
    )


if __name__ == "__main__":
    main()
