"""面向目标值的 look-ahead 教师（imitation）与缓存构建。

算法：
1. 与 ``decide_omt.solve_omt_with_decider`` 相同，经 ``prepare_propagator_formula``
   得到会注册到 UserPropagator 的析取子句原子；
2. 用 z3 二进制求原问题（预处理后硬约束）的目标最优值；
3. 对每个注册原子分别加真/假硬约束，再用 z3 二进制求局部最优；
4. 打分（设注册原子数为 n）：
   - 真/假均不改变全局最优 → 得分 ``-n``（相位任意）；
   - 一侧全局最优、另一侧 unsat → 得分 ``0``（相位取全局最优侧）；
   - 其余：一侧全局最优、另一侧为更差的局部最优，按
     ``|局部最优 - 全局最优|`` 从大到小排序，得分从 ``n`` 递减、步长 1
     （相位取全局最优侧）。

缓存布局（与 split look-ahead 分离）::

    lookahead_objective/<split>/<instance_id>.json

运行::

    python -m examples.build_objective_lookahead
    python -m examples.build_objective_lookahead --split train --workers 8 --force
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.trainer import RankingExample
from omt_branching.solver.decide_omt import (
    list_split_entries,
    smt2_to_instance,
    solve_binary,
)
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.propagator_snapshot import (
    atom_key,
    build_bool_snapshot,
    prepare_propagator_formula,
)

import z3

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_DATASET_DIR = os.path.join(ARTIFACTS, "dataset")
DEFAULT_WORKERS = max(1, min(12, (os.cpu_count() or 4)))
LOOKAHEAD_OBJECTIVE_SUBDIR = "lookahead_objective"


def objective_lookahead_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    return Path(dataset_dir) / LOOKAHEAD_OBJECTIVE_SUBDIR / split / f"{instance_id}.json"


def has_objective_lookahead_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return objective_lookahead_path(dataset_dir, instance_id, split=split).is_file()


def save_objective_lookahead_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
    scores: dict[str, float],
    phases: dict[str, bool],
    opt_value,
    n_atoms: int,
) -> Path:
    path = objective_lookahead_path(dataset_dir, instance_id, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance_id": instance_id,
        "split": split,
        "kind": "objective",
        "n_atoms": int(n_atoms),
        "opt_value": str(opt_value) if opt_value is not None else None,
        "scores": {str(k): float(v) for k, v in scores.items()},
        "phases": {str(k): bool(v) for k, v in phases.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_objective_lookahead_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Optional[dict]:
    path = objective_lookahead_path(dataset_dir, instance_id, split=split)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("kind") not in (None, "objective"):
        return None
    return {
        "scores": {str(k): float(v) for k, v in (payload.get("scores") or {}).items()},
        "phases": {str(k): bool(v) for k, v in (payload.get("phases") or {}).items()},
        "opt_value": payload.get("opt_value"),
        "n_atoms": payload.get("n_atoms"),
    }


def _as_fraction(v) -> Optional[Fraction]:
    if v is None:
        return None
    if isinstance(v, Fraction):
        return v
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(v).limit_denominator()
    return Fraction(str(v))


def _forced_instance(inst: OMTInstance, hard_base: list, lit) -> OMTInstance:
    """在预处理硬约束上追加字面量，供 z3 二进制求解。"""
    return replace(
        inst,
        hard=list(hard_base) + [lit],
        instance_id=f"{inst.instance_id}__objla",
    )


def _solve_opt(
    inst: OMTInstance,
    *,
    z3_path: str | None,
    timeout_s: int,
) -> tuple[Optional[Fraction], str]:
    res = solve_binary(inst, z3_path=z3_path, timeout_s=timeout_s)
    status = str(res.get("status") or "error")
    if status != "sat":
        return None, status
    return _as_fraction(res.get("value")), status


def objective_lookahead_scores(
    inst: OMTInstance,
    *,
    z3_path: str | None = None,
    opt_value=None,
    timeout_s: int = 120,
) -> tuple[dict[str, float], dict[str, bool], Fraction | None, int]:
    """对单实例计算目标值 look-ahead 分数与相位。

    返回 ``(scores, phases, global_opt, n_atoms)``；``scores`` / ``phases`` 以
    ``atom_key`` 为键。
    """
    hard_use, atoms = prepare_propagator_formula(list(inst.hard))
    n = len(atoms)
    if n == 0:
        return {}, {}, None, 0

    base = replace(inst, hard=list(hard_use), instance_id=f"{inst.instance_id}__base")
    opt = _as_fraction(opt_value)
    if opt is None:
        opt, st = _solve_opt(base, z3_path=z3_path, timeout_s=timeout_s)
        if opt is None:
            return {}, {}, None, n

    # 分类：unaffected / failed_lit / impactful
    unaffected: list[str] = []
    failed_lit: list[tuple[str, bool]] = []  # (key, phase_toward_opt)
    impactful: list[tuple[str, Fraction, bool]] = []  # (key, gap, phase)

    for a in atoms:
        k = atom_key(a)
        val_t, st_t = _solve_opt(
            _forced_instance(inst, hard_use, a),
            z3_path=z3_path,
            timeout_s=timeout_s,
        )
        val_f, st_f = _solve_opt(
            _forced_instance(inst, hard_use, z3.Not(a)),
            z3_path=z3_path,
            timeout_s=timeout_s,
        )

        # 任一侧超时/unknown 等非 sat/unsat → 跳过该原子
        if st_t not in ("sat", "unsat") or st_f not in ("sat", "unsat"):
            continue

        t_opt = st_t == "sat" and val_t is not None and val_t == opt
        f_opt = st_f == "sat" and val_f is not None and val_f == opt
        t_unsat = st_t == "unsat"
        f_unsat = st_f == "unsat"

        if t_opt and f_opt:
            unaffected.append(k)
            continue

        if (t_opt and f_unsat) or (f_opt and t_unsat):
            failed_lit.append((k, True if t_opt else False))
            continue

        if t_opt and st_f == "sat" and val_f is not None and val_f != opt:
            impactful.append((k, abs(val_f - opt), True))
            continue
        if f_opt and st_t == "sat" and val_t is not None and val_t != opt:
            impactful.append((k, abs(val_t - opt), False))
            continue

        # 异常：两侧均非全局最优但仍 sat —— 取更接近全局的一侧作相位，按较大偏差打入 impactful
        if st_t == "sat" and st_f == "sat" and val_t is not None and val_f is not None:
            gap_t, gap_f = abs(val_t - opt), abs(val_f - opt)
            if gap_t <= gap_f:
                impactful.append((k, gap_f, True))
            else:
                impactful.append((k, gap_t, False))

    scores: dict[str, float] = {}
    phases: dict[str, bool] = {}

    for k in unaffected:
        scores[k] = float(-n)
        phases[k] = True  # 任意

    for k, ph in failed_lit:
        scores[k] = 0.0
        phases[k] = ph

    impactful.sort(key=lambda row: (-row[1], row[0]))
    rank = n
    for k, _gap, ph in impactful:
        scores[k] = float(rank)
        phases[k] = ph
        rank -= 1

    return scores, phases, opt, n


def _scores_to_example(
    hard_for_graph: list,
    scores: dict[str, float],
    phases: dict[str, bool],
) -> RankingExample | None:
    snap, _amap = build_bool_snapshot(hard_for_graph)
    graph = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
    bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
    bts: dict[int, float] = {}
    pts: dict[int, bool] = {}
    for k, sc in scores.items():
        loc = bmap.get(k)
        if loc is not None:
            bts[loc] = sc
    for k, ph in phases.items():
        loc = bmap.get(k)
        if loc is not None:
            pts[loc] = ph
    if not bts:
        return None
    return RankingExample(graph=graph, bool_target_scores=bts, phase_targets=pts)


def build_objective_lookahead_example(
    inst: OMTInstance,
    *,
    z3_path: str | None = None,
    opt_value=None,
    timeout_s: int = 120,
    scores_phases: tuple[dict, dict] | None = None,
) -> RankingExample | None:
    """单实例 → RankingExample；可用缓存的 ``scores_phases`` 跳过求解。"""
    hard_use, _atoms = prepare_propagator_formula(list(inst.hard))
    if scores_phases is not None:
        scores, phases = scores_phases
    else:
        scores, phases, _opt, _n = objective_lookahead_scores(
            inst, z3_path=z3_path, opt_value=opt_value, timeout_s=timeout_s
        )
    if not scores:
        return None
    # 建图用预处理断言，使 atom_key 与注册原子一致（同 decide 臂）
    return _scores_to_example(hard_use, scores, phases)


def _compute_and_maybe_cache(
    inst: OMTInstance,
    *,
    dataset_dir: str | None = None,
    split: str | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
    z3_path: str | None = None,
    opt_value=None,
    timeout_s: int = 120,
) -> RankingExample | None:
    if use_cache and dataset_dir and split and inst.instance_id:
        cached = load_objective_lookahead_result(
            dataset_dir, inst.instance_id, split=split
        )
        if cached is not None and cached["scores"]:
            return build_objective_lookahead_example(
                inst, scores_phases=(cached["scores"], cached["phases"])
            )
        if cache_only:
            return None

    if cache_only:
        return None

    scores, phases, opt, n = objective_lookahead_scores(
        inst, z3_path=z3_path, opt_value=opt_value, timeout_s=timeout_s
    )
    if use_cache and dataset_dir and split and inst.instance_id and scores:
        save_objective_lookahead_result(
            dataset_dir,
            inst.instance_id,
            split=split,
            scores=scores,
            phases=phases,
            opt_value=opt,
            n_atoms=n,
        )
    if not scores:
        return None
    return build_objective_lookahead_example(inst, scores_phases=(scores, phases))


def _from_smt2_worker(task: tuple) -> tuple[int, RankingExample | None]:
    (
        index,
        smt2_path,
        instance_id,
        dataset_dir,
        split,
        use_cache,
        cache_only,
        z3_path,
        timeout_s,
        opt_value_str,
    ) = task
    inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    opt = _as_fraction(opt_value_str) if opt_value_str is not None else None
    return index, _compute_and_maybe_cache(
        inst,
        dataset_dir=dataset_dir,
        split=split,
        use_cache=use_cache,
        cache_only=cache_only,
        z3_path=z3_path,
        opt_value=opt,
        timeout_s=timeout_s,
    )


def build_objective_lookahead_examples_from_smt2_parallel(
    smt2_paths: list[str],
    *,
    instance_ids: list[str] | None = None,
    workers: int = DEFAULT_WORKERS,
    dataset_dir: str | None = None,
    split: str | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
    z3_path: str | None = None,
    opt_values: list | None = None,
    timeout_s: int = 120,
) -> list[RankingExample]:
    """从已落盘 ``.smt2`` 并行构造目标值 look-ahead imitation 样本。

    ``cache_only=True`` 时只读缓存，缺失则跳过该实例（不现算）。
    """
    if not smt2_paths:
        return []
    ids = instance_ids or [None] * len(smt2_paths)
    if len(ids) != len(smt2_paths):
        raise ValueError("instance_ids 长度必须与 smt2_paths 一致")
    opts = opt_values if opt_values is not None else [None] * len(smt2_paths)
    if len(opts) != len(smt2_paths):
        raise ValueError("opt_values 长度必须与 smt2_paths 一致")

    if workers <= 1:
        out: list[RankingExample] = []
        for path, iid, ov in zip(smt2_paths, ids, opts):
            inst = smt2_to_instance(path, instance_id=iid)
            ex = _compute_and_maybe_cache(
                inst,
                dataset_dir=dataset_dir,
                split=split,
                use_cache=use_cache,
                cache_only=cache_only,
                z3_path=z3_path,
                opt_value=ov,
                timeout_s=timeout_s,
            )
            if ex is not None:
                out.append(ex)
        return out

    n = len(smt2_paths)
    workers = min(workers, n)
    tasks = [
        (
            i,
            smt2_paths[i],
            ids[i],
            dataset_dir,
            split,
            use_cache,
            cache_only,
            z3_path,
            timeout_s,
            None if opts[i] is None else str(opts[i]),
        )
        for i in range(n)
    ]
    slots: list[RankingExample | None] = [None] * n
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_from_smt2_worker, t) for t in tasks]
        with tqdm(total=len(tasks), desc="obj-lookahead") as pbar:
            for fut in as_completed(futures):
                index, ex = fut.result()
                if ex is not None:
                    slots[index] = ex
                pbar.update(1)
    return [ex for ex in slots if ex is not None]


def _cache_worker(task: tuple) -> dict:
    (
        dataset_dir,
        split,
        instance_id,
        smt2_relpath,
        force,
        z3_path,
        timeout_s,
        opt_value_str,
    ) = task
    if not force and has_objective_lookahead_result(
        dataset_dir, instance_id, split=split
    ):
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
    opt = _as_fraction(opt_value_str) if opt_value_str is not None else None
    scores, phases, opt_v, n = objective_lookahead_scores(
        inst, z3_path=z3_path, opt_value=opt, timeout_s=timeout_s
    )
    if not scores:
        return {
            "instance_id": instance_id,
            "split": split,
            "skipped": False,
            "status": "empty",
            "n_scores": 0,
            "n_atoms": n,
        }
    save_objective_lookahead_result(
        dataset_dir,
        instance_id,
        split=split,
        scores=scores,
        phases=phases,
        opt_value=opt_v,
        n_atoms=n,
    )
    return {
        "instance_id": instance_id,
        "split": split,
        "skipped": False,
        "status": "ok",
        "n_scores": len(scores),
        "n_atoms": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="并行构建目标值 look-ahead 标签缓存")
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default=None, help="只处理某一划分（默认全部）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--z3-path", default=None, help="z3 可执行文件路径")
    ap.add_argument("--timeout", type=int, default=120, help="单次 z3binary 超时（秒）")
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存")
    ap.add_argument(
        "--use-ref-value",
        action="store_true",
        help="若存在 ref 缓存则复用其 binary 最优值作为全局最优（仍对真/假侧跑 z3binary）",
    )
    args = ap.parse_args()

    root = Path(args.dataset_dir)
    manifest_path = root / "manifest.json"
    splits: dict[str, list] = {}
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            splits = json.load(f).get("splits", {})
    keys = [args.split] if args.split else (list(splits.keys()) or ["test", "train"])

    ref_values: dict[tuple[str, str], str] = {}
    if args.use_ref_value:
        from omt_branching.solver.binary_results import binary_value, load_binary_result

        for sp in keys:
            entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
            for e in entries:
                iid = e["instance_id"]
                if load_binary_result(args.dataset_dir, iid, split=sp) is None:
                    continue
                try:
                    v = binary_value(args.dataset_dir, iid, split=sp)
                except Exception:
                    continue
                if v is not None:
                    ref_values[(sp, iid)] = str(v)

    tasks: list[tuple] = []
    for sp in keys:
        entries = splits.get(sp) or list_split_entries(args.dataset_dir, sp)
        for e in entries:
            iid = e["instance_id"]
            tasks.append(
                (
                    args.dataset_dir,
                    sp,
                    iid,
                    e["smt2"],
                    args.force,
                    args.z3_path,
                    args.timeout,
                    ref_values.get((sp, iid)),
                )
            )

    if not tasks:
        print("无可处理实例")
        return

    n_workers = max(1, min(args.workers, len(tasks)))
    print(
        f"objective lookahead 缓存: {len(tasks)} 实例, workers={n_workers}, "
        f"force={args.force}, use_ref_value={args.use_ref_value}"
    )
    print(f"结果目录: {root / LOOKAHEAD_OBJECTIVE_SUBDIR}/")

    stats = {"cached": 0, "ok": 0, "empty": 0, "fail": 0}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_cache_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc="obj-lookahead") as pbar:
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                except Exception:
                    stats["fail"] += 1
                    pbar.set_postfix(**stats)
                    pbar.update(1)
                    continue
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
