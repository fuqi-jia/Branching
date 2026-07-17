"""并行为数据集构建 look-ahead 标签缓存（``lookahead/<split>/<id>.json``）。

同一 ``.smt2`` + 相同 LookaheadConfig 下结果确定性；供 imitation 训练复用，避免每次
重跑 z3 consequences。

运行::

    python -m examples.build_lookahead_cache
    python -m examples.build_lookahead_cache --dataset-dir examples/artifacts/dataset
    python -m examples.build_lookahead_cache --split train --workers 12 --force
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from omt_branching.solver.decide_omt import list_split_entries, smt2_to_instance
from omt_branching.solver.lookahead import LookaheadConfig, lookahead_scores
from omt_branching.solver.lookahead_cache import (
    has_lookahead_result,
    save_lookahead_result,
)
from omt_branching.solver.propagator_snapshot import build_bool_snapshot

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_WORKERS = max(1, min(12, (os.cpu_count() or 4)))


def _worker(task: tuple) -> dict:
    (
        dataset_dir,
        split,
        instance_id,
        smt2_relpath,
        max_atoms,
        eps,
        force,
    ) = task
    if not force and has_lookahead_result(dataset_dir, instance_id, split=split):
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": True,
            "status": "cached",
        }

    smt2_path = Path(dataset_dir) / smt2_relpath
    if not smt2_path.is_file():
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": False,
            "status": "missing_smt2",
        }

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    cfg = LookaheadConfig(max_atoms=max_atoms, eps=eps)
    hard = list(inst.hard)
    snap, amap = build_bool_snapshot(hard)
    scores, phases = lookahead_scores(hard, atoms=list(amap.values()), config=cfg)
    if not scores:
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": False,
            "status": "empty",
            "n_scores": 0,
        }
    save_lookahead_result(
        dataset_dir,
        instance_id,
        split=split,
        scores=scores,
        phases=phases,
        max_atoms=max_atoms,
        eps=eps,
    )
    return {
        "instance_id": instance_id,
        "split": split,
        "skipped": False,
        "status": "ok",
        "n_scores": len(scores),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="并行构建 look-ahead 标签缓存")
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default=None, help="只处理某一划分（默认全部）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--max-atoms", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存")
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    manifest_path = root / "manifest.json"
    splits: dict[str, list] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            splits = json.load(f).get("splits", {})
    keys = [args.split] if args.split else (list(splits.keys()) or ["test", "train"])

    tasks: list[tuple] = []
    for sp in keys:
        entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
        for e in entries:
            tasks.append((
                args.dataset_dir,
                sp,
                e["instance_id"],
                e["smt2"],
                args.max_atoms,
                args.eps,
                args.force,
            ))

    if not tasks:
        print("无可处理实例")
        return

    n_workers = max(1, min(args.workers, len(tasks)))
    print(
        f"lookahead 缓存: {len(tasks)} 实例, workers={n_workers}, "
        f"max_atoms={args.max_atoms}, force={args.force}"
    )
    print(f"结果目录: {root / 'lookahead'}/")

    stats = {"cached": 0, "ok": 0, "empty": 0, "fail": 0}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="lookahead") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                st = row.get("status")
                if row.get("skipped"):
                    stats["cached"] += 1
                elif st == "ok":
                    stats["ok"] += 1
                elif st == "empty":
                    stats["empty"] += 1
                else:
                    stats["fail"] += 1
                pbar.set_postfix(**stats)
                pbar.update(1)

    print(
        f"完成: cached={stats['cached']} ok={stats['ok']} "
        f"empty={stats['empty']} fail={stats['fail']}"
    )


if __name__ == "__main__":
    main()
