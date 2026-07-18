"""对数据集中每个实例构造 ``ref/<split>/<id>.json`` 参考缓存。

对每个实例：

1. 跑 z3 **binary**（``solve_binary``）得到最优 ``value``；
2. 跑 **公平 VSIDS**（``solve_omt_with_decider``，预处理 + 挂 propagator、decide 恒
   defer）得到同口径资源；
3. 跑 **check-sat-loop**（同样预处理，但不挂 propagator）作为原 VSIDS 基线；
4. 落盘到 ``ref/``：``value`` 始终取 binary；若公平 VSIDS 目标值与 binary 一致，
   则缓存 ``rlimit`` 取 VSIDS 的 ``rlimit``，否则回退 binary ``rlimit``。

可供 ``decide_branch`` 评测臂与 RL ``binary_rlimit`` / ``binary_value`` 复用
（含 ``train`` / ``test`` / ``eval``）。

运行::

    python -m examples.solve_dataset_binary
    python -m examples.solve_dataset_binary --dataset-dir examples/artifacts/dataset
    python -m examples.solve_dataset_binary --workers 16 --timeout 1200 --force
    python -m examples.solve_dataset_binary --split test
    python -m examples.solve_dataset_binary --split eval
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from omt_branching.solver.binary_results import (
    REF_SUBDIR,
    build_ref_payload,
    is_fair_vsids_cache,
    load_binary_result,
    save_binary_result,
)
from omt_branching.solver.decide_omt import (
    list_split_entries,
    manifest_mismatches,
    rebuild_manifest,
    smt2_to_instance,
    solve_binary,
    solve_omt_with_decider,
)

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_WORKERS = max(1, min(30, (os.cpu_count() or 4)))


def _ensure_manifest(dataset_dir: str) -> dict:
    path = Path(dataset_dir) / "manifest.json"
    issues = manifest_mismatches(dataset_dir)
    if issues or not path.is_file():
        if issues:
            print("检测到 manifest 与磁盘不一致，按 .smt2 重建：")
            for msg in issues:
                print(f"  - {msg}")
        rebuild_manifest(dataset_dir, preserve_meta=True)
        print(f"已重建 -> {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _solve_worker(task: tuple) -> dict:
    """ProcessPool worker：binary + 公平 VSIDS + check-sat-loop，立刻写入 ref/。"""
    (
        dataset_dir,
        split,
        instance_id,
        smt2_relpath,
        z3_path,
        timeout_s,
        force,
    ) = task
    if not force:
        cached = load_binary_result(dataset_dir, instance_id, split=split)
        if cached is not None and is_fair_vsids_cache(cached):
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
            "error": str(smt2_path),
        }

    smt2_text = smt2_path.read_text(encoding="utf-8")
    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    hard, obj, sense = inst.as_tuple()

    bin_res = solve_binary(
        inst, z3_path=z3_path, timeout_s=timeout_s, smt2=smt2_text
    )
    vsids_res: dict = {}
    csl_res: dict = {}
    if bin_res.get("status") in ("sat", "unsat") and bin_res.get("value") is not None:
        ref_rl = bin_res.get("rlimit")
        try:
            vsids_res = solve_omt_with_decider(
                hard,
                obj,
                sense,
                decider_factory=None,
                attach_propagator=True,
                ref_rlimit=ref_rl,
            )
        except Exception as exc:  # noqa: BLE001 — worker 内记失败，不拖垮整批
            vsids_res = {"value": None, "error": str(exc)}
        try:
            csl_res = solve_omt_with_decider(
                hard,
                obj,
                sense,
                attach_propagator=False,
                ref_rlimit=ref_rl,
            )
        except Exception as exc:  # noqa: BLE001
            csl_res = {"value": None, "error": str(exc)}

    payload = build_ref_payload(bin_res, vsids_res, csl_res)
    save_binary_result(dataset_dir, instance_id, payload, split=split)
    return {
        "instance_id": instance_id,
        "split": split,
        "skipped": False,
        "status": bin_res.get("status"),
        "rlimit": payload.get("rlimit"),
        "rlimit_source": payload.get("rlimit_source"),
        "vsids_match": payload.get("vsids_match"),
        "value": str(payload["value"]) if payload.get("value") is not None else None,
        "time_ms": bin_res.get("time_ms"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "并行构造数据集 ref 参考缓存"
            "（binary 目标值 + 公平 VSIDS / check-sat-loop rlimit）"
        )
    )
    ap.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="含 manifest.json 的数据集目录",
    )
    ap.add_argument(
        "--split",
        default=None,
        help="只处理某一划分，如 train / test / eval（默认处理全部划分）",
    )
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径")
    ap.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="单实例 binary 超时（秒）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行进程数（默认 {DEFAULT_WORKERS}）",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有 ref 结果并重新求解",
    )
    args = ap.parse_args()

    z3_path = args.z3_path or shutil.which("z3")
    if not z3_path:
        raise SystemExit("未找到 z3 二进制，请用 --z3-path 指定")

    manifest = _ensure_manifest(args.dataset_dir)
    splits = manifest.get("splits", {})
    keys = [args.split] if args.split else (
        list(splits.keys()) or ["test", "train", "eval"]
    )
    tasks: list[tuple] = []
    for sp in keys:
        entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
        if args.split and sp == args.split and not entries:
            raise SystemExit(f"划分无实例: {sp}")
        for entry in entries:
            tasks.append((
                args.dataset_dir,
                sp,
                entry["instance_id"],
                entry["smt2"],
                z3_path,
                args.timeout,
                args.force,
            ))

    if not tasks:
        print("无可求解实例")
        return

    n_workers = max(1, min(args.workers, len(tasks)))
    print(
        f"ref 求解: {len(tasks)} 实例, workers={n_workers}, "
        f"timeout={args.timeout}s, force={args.force}"
    )
    print(f"结果目录: {Path(args.dataset_dir) / REF_SUBDIR}/")
    print(
        "规则: value←binary; rlimit←公平 vsids(若目标值一致) 否则←binary；"
        "另缓存 check_sat_loop（预处理、无 propagator）"
    )

    stats = {
        "cached": 0,
        "ok": 0,
        "fail": 0,
        "timeout": 0,
        "vsids_rlimit": 0,
        "binary_rlimit": 0,
    }
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_solve_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="ref") as pbar:
            for fut in as_completed(futures):
                row = fut.result()
                st = row.get("status")
                if row.get("skipped"):
                    stats["cached"] += 1
                elif st == "timeout":
                    stats["timeout"] += 1
                elif st in ("sat", "unsat"):
                    stats["ok"] += 1
                    if row.get("rlimit_source") == "vsids":
                        stats["vsids_rlimit"] += 1
                    else:
                        stats["binary_rlimit"] += 1
                else:
                    stats["fail"] += 1
                pbar.set_postfix(
                    cached=stats["cached"],
                    ok=stats["ok"],
                    vsids_rl=stats["vsids_rlimit"],
                    bin_rl=stats["binary_rlimit"],
                    fail=stats["fail"],
                    timeout=stats["timeout"],
                )
                pbar.update(1)

    print(
        f"完成: cached={stats['cached']} ok={stats['ok']} "
        f"rlimit_from_vsids={stats['vsids_rlimit']} "
        f"rlimit_from_binary={stats['binary_rlimit']} "
        f"timeout={stats['timeout']} fail={stats['fail']}"
    )


if __name__ == "__main__":
    main()
