"""加载已有策略权重，在数据集 test 划分上做三臂对比并写 ``results.json``。

评测逻辑与 ``examples/decide_branch.py`` 一致（``PolicyDecider`` + binary 缓存参考），
不训练。默认权重 ``examples/artifacts/rl_decide_policy.pt``，默认数据
``examples/artifacts/dataset``。

运行::

    python -m examples.eval_checkpoint
    python -m examples.eval_checkpoint --checkpoint examples/artifacts/rl_checkpoints/iter_0005.pt
    python -m examples.eval_checkpoint --dataset-dir path/to/dataset --out path/to/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from omt_branching.model.device import gnn_device
from omt_branching.model.persistence import load_policy
from omt_branching.solver import list_split_entries, load_dataset

from examples.decide_branch import (
    ARTIFACTS,
    DEFAULT_DATASET_DIR,
    DEFAULT_TEST_WORKERS,
    _json_value,
    _require_binary_cache,
    _require_dataset,
    _run_test_parallel,
    _stats_for_json,
    _sync_manifest_if_needed,
)

DEFAULT_CKPT = os.path.join(ARTIFACTS, "rl_decide_policy.pt")
DEFAULT_OUT = os.path.join(ARTIFACTS, "results.json")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="加载 checkpoint，在 dataset test 上三臂对比并写 results.json"
    )
    ap.add_argument(
        "--checkpoint",
        "--ckpt",
        dest="checkpoint",
        default=DEFAULT_CKPT,
        help=f"策略权重 .pt（默认 {DEFAULT_CKPT}）",
    )
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"数据集目录（默认 {DEFAULT_DATASET_DIR}）",
    )
    ap.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"结果 JSON 路径（默认 {DEFAULT_OUT}）",
    )
    ap.add_argument("--refocus", type=int, default=50)
    ap.add_argument(
        "--test-workers",
        type=int,
        default=DEFAULT_TEST_WORKERS,
        help=f"测试并发数（默认 {DEFAULT_TEST_WORKERS}）",
    )
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径（默认同 PATH）")
    ap.add_argument(
        "--device",
        default=None,
        help="GNN 设备（默认 cuda 可用则 cuda，否则 cpu）",
    )
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"权重不存在: {ckpt_path}")

    dataset_dir = args.dataset_dir
    device = args.device or gnn_device()
    print(f"GNN device: {device}")
    print(f"checkpoint: {ckpt_path.resolve()}")
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
    _require_binary_cache(dataset_dir, "test", test_entries)

    policy, meta = load_policy(ckpt_path, map_location="cpu")
    policy.to(device)
    policy.eval()
    if meta:
        defer = meta.get("defer_logit")
        extra = f", defer_logit={defer}" if defer is not None else ""
        print(f"已加载权重 meta keys={list(meta.keys())}{extra}")

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

    print(f"summary.binary  = { {k: round(v, 4) if isinstance(v, float) else v for k, v in agg['binary'].items()} }")
    print(f"summary.vsids   match={agg['vsids']['match']:.3f}  "
          f"weighted_rlimit={agg['vsids'].get('weighted rlimit', 0):.1f}")
    print(f"summary.learned match={agg['learned']['match']:.3f}  "
          f"weighted_rlimit={agg['learned'].get('weighted rlimit', 0):.1f}")

    results = {
        "reference": "binary_cache",
        "dataset_dir": dataset_dir,
        "summary": agg,
        "n_instances": len(insts),
        "z3_path": z3_path,
        "device": device,
        "test_workers": args.test_workers,
        "per_instance": per_instance,
        # 评测专用溯源（decide_branch 无此字段）
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_meta": {k: _json_value(v) for k, v in meta.items()},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"实验汇总已保存 -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
