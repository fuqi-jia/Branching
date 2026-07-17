"""在测试集上批量跑 ``solve_omt_with_decider(..., decider_factory=None)``。

``factory=None`` 时**不挂** UserPropagator，走 Solver 线性搜索 + z3 原生 VSIDS
（有预处理），用于测量「始终 defer / 无学习回调」时的回路效率，并可选对照
``binary/`` 缓存。

运行::

    python -m examples.bench_vsids_test
    python -m examples.bench_vsids_test --dataset-dir examples/artifacts/dataset
    python -m examples.bench_vsids_test --workers 16 --limit 20
    python -m examples.bench_vsids_test --ref-rlimit-from-binary
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

from omt_branching.solver.binary_results import load_binary_result
from omt_branching.solver.decide_omt import (
    list_split_entries,
    manifest_mismatches,
    rebuild_manifest,
    smt2_to_instance,
    solve_omt_with_decider,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_WORKERS = max(1, min(30, (os.cpu_count() or 4)))


def _json_value(v):
    if v is None:
        return None
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, float):
        return v
    return v


def _ensure_manifest(dataset_dir: str) -> None:
    path = Path(dataset_dir) / "manifest.json"
    issues = manifest_mismatches(dataset_dir)
    if issues or not path.is_file():
        if issues:
            print("检测到 manifest 与磁盘不一致，按 .smt2 重建：")
            for msg in issues:
                print(f"  - {msg}")
        rebuild_manifest(dataset_dir, preserve_meta=True)
        print(f"已重建 -> {path}")


def _vsids_worker(task: tuple) -> dict:
    """ProcessPool：单实例 ``decider_factory=None``。"""
    dataset_dir, split, instance_id, smt2_rel, use_ref_rlimit = task
    smt2_path = Path(dataset_dir) / smt2_rel
    if not smt2_path.is_file():
        return {
            "instance_id": instance_id,
            "ok": False,
            "error": f"missing_smt2:{smt2_path}",
        }

    ref_rlimit = None
    ref_value = None
    ref_time_ms = None
    if use_ref_rlimit:
        cached = load_binary_result(dataset_dir, instance_id, split=split)
        if cached is not None:
            ref_rlimit = cached.get("rlimit")
            ref_value = cached.get("value")
            ref_time_ms = cached.get("time_ms")

    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    hard, obj, sense = inst.as_tuple()
    t0 = time.perf_counter()
    try:
        res = solve_omt_with_decider(
            hard,
            obj,
            sense,
            decider_factory=None,
            ref_rlimit=ref_rlimit if use_ref_rlimit else None,
        )
    except Exception as exc:  # noqa: BLE001 — worker 汇总错误
        return {
            "instance_id": instance_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "wall_s": time.perf_counter() - t0,
            "ref_rlimit": ref_rlimit,
            "ref_value": _json_value(ref_value),
            "ref_time_ms": ref_time_ms,
        }
    wall_s = time.perf_counter() - t0
    match = None
    if ref_value is not None and res.get("value") is not None:
        match = res["value"] == ref_value
    return {
        "instance_id": instance_id,
        "ok": True,
        "wall_s": wall_s,
        "value": _json_value(res.get("value")),
        "rlimit": res.get("rlimit"),
        "conflicts": res.get("conflicts"),
        "decisions": res.get("decisions"),  # factory=None 时为 None
        "iters": res.get("iters"),
        "ref_rlimit": ref_rlimit,
        "ref_value": _json_value(ref_value),
        "ref_time_ms": ref_time_ms,
        "match_binary": match,
    }


def _summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    fail = [r for r in rows if not r.get("ok")]
    walls = [float(r["wall_s"]) for r in ok if r.get("wall_s") is not None]
    rlimits = [int(r["rlimit"]) for r in ok if r.get("rlimit") is not None]
    matched = [r for r in ok if r.get("match_binary") is True]
    mismatched = [r for r in ok if r.get("match_binary") is False]
    with_ref = [r for r in ok if r.get("ref_time_ms") is not None and r.get("wall_s") is not None]
    ratio_vs_bin = []
    for r in with_ref:
        bt = float(r["ref_time_ms"]) / 1000.0
        if bt > 0:
            ratio_vs_bin.append(float(r["wall_s"]) / bt)

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        xs_s = sorted(xs)
        return {
            "n": len(xs),
            "sum": sum(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "p90": xs_s[max(0, int(0.9 * (len(xs_s) - 1)))],
            "max": xs_s[-1],
            "min": xs_s[0],
        }

    return {
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_fail": len(fail),
        "wall_s": _stats(walls),
        "rlimit": _stats([float(x) for x in rlimits]),
        "match_binary": {
            "n_compared": len(matched) + len(mismatched),
            "n_match": len(matched),
            "n_mismatch": len(mismatched),
        },
        "wall_over_binary_time": _stats(ratio_vs_bin),
        "failures": [
            {"instance_id": r["instance_id"], "error": r.get("error")} for r in fail
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="测试集批量 VSIDS（decider_factory=None）效率评测"
    )
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="含 manifest.json / test/*.smt2 的数据集目录",
    )
    ap.add_argument(
        "--split",
        default="test",
        help="划分名（默认 test）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行进程数（默认 {DEFAULT_WORKERS}）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多评测前 N 个实例（0=全部）",
    )
    ap.add_argument(
        "--ref-rlimit-from-binary",
        action="store_true",
        help="若有 binary 缓存，传入 ref_rlimit 做 OMT 剪枝，并对比 value/墙钟",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="结果 JSON 路径（默认 artifacts/bench_vsids_<split>_<ts>.json）",
    )
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        raise SystemExit(f"数据集目录不存在: {root}")

    _ensure_manifest(args.dataset_dir)
    entries = list_split_entries(args.dataset_dir, args.split)
    if not entries:
        raise SystemExit(f"划分无实例: {args.split} @ {args.dataset_dir}")
    if args.limit and args.limit > 0:
        entries = entries[: args.limit]

    tasks = [
        (
            args.dataset_dir,
            args.split,
            e["instance_id"],
            e["smt2"],
            bool(args.ref_rlimit_from_binary),
        )
        for e in entries
    ]
    n_workers = max(1, min(args.workers, len(tasks)))
    print(
        f"VSIDS bench (factory=None): split={args.split}, "
        f"n={len(tasks)}, workers={n_workers}, "
        f"ref_rlimit_from_binary={args.ref_rlimit_from_binary}"
    )
    print(
        "说明: 不挂 UserPropagator；与「挂 prop 且始终 defer」不同"
        "（后者仍关预处理）。"
    )

    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_vsids_worker, t): t[2] for t in tasks}
        with tqdm(total=len(tasks), desc=f"vsids:{args.split}") as pbar:
            for fut in as_completed(futures):
                rows.append(fut.result())
                pbar.update(1)
    elapsed = time.perf_counter() - t0
    rows.sort(key=lambda r: r.get("instance_id") or "")

    summary = _summarize(rows)
    summary["wall_clock_total_s"] = elapsed
    summary["split"] = args.split
    summary["dataset_dir"] = str(root.resolve())
    summary["decider_factory"] = None
    summary["note"] = (
        "solve_omt_with_decider(decider_factory=None)：无 propagator，原生 VSIDS"
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else Path(ARTIFACTS) / f"bench_vsids_{args.split}_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    w = summary["wall_s"]
    rl = summary["rlimit"]
    print("--- summary ---")
    print(f"ok={summary['n_ok']}/{summary['n_total']}  fail={summary['n_fail']}")
    if w.get("n"):
        print(
            f"wall_s: mean={w['mean']:.3f}  median={w['median']:.3f}  "
            f"p90={w['p90']:.3f}  max={w['max']:.3f}  sum={w['sum']:.3f}"
        )
    if rl.get("n"):
        print(
            f"rlimit: mean={rl['mean']:.0f}  median={rl['median']:.0f}  "
            f"max={rl['max']:.0f}"
        )
    mb = summary["match_binary"]
    if mb["n_compared"]:
        print(
            f"vs binary value: match={mb['n_match']}/{mb['n_compared']}  "
            f"mismatch={mb['n_mismatch']}"
        )
    ratio = summary["wall_over_binary_time"]
    if ratio.get("n"):
        print(
            f"wall/binary_time: mean={ratio['mean']:.2f}x  "
            f"median={ratio['median']:.2f}x  max={ratio['max']:.2f}x"
        )
    print(f"parallel wall_clock_total_s={elapsed:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
