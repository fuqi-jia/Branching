"""探针：筛选「CSL≈VSIDS、但 oracle 分支可显著降 rlimit」的分支敏感实例。

对候选 ``branch_focus`` 实例测量三臂：

- **check-sat-loop**（``attach_propagator=False``）
- **公平 VSIDS**（挂 prop，decide 恒 defer）
- **oracle**（挂 prop，优先分支种植的模式守卫）

筛选条件（可用 CLI 调）：

1. ``|rlimit_csl - rlimit_vsids| / max(rlimit_csl,1) <= --max-csl-vsids-gap``
2. ``rlimit_oracle / rlimit_vsids <= --max-oracle-ratio``（理论 headroom）
3. 三臂最优 ``value`` 一致

通过后可 ``--save`` 落到数据集目录（含 ``oracle_priority`` 写入 manifest）。

运行::

    python -m examples.probe_branch_focus
    python -m examples.probe_branch_focus --candidates 40 --need 10 --save
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from omt_branching.solver.decide_omt import (
    rebuild_manifest,
    save_dataset,
    solve_omt_with_decider,
)
from omt_branching.solver.instance_gen import generate_branch_focus_lia_instance
from omt_branching.solver.propagator_snapshot import atom_key

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_OUT = os.path.join(ARTIFACTS, "dataset_branch_focus")


def _priority_decider_factory(priority: list[tuple[str, bool]]):
    """按种植序优先分支；未命中守卫时 defer 回 VSIDS。

    在 ``factory(assertions)`` 时按预处理后公式重映射 key，避免 translate/simplify
    后 ``atom_key`` 对不上。
    """

    def factory(assertions):
        prio = _remap_priority(priority, assertions)
        rank = {k: i for i, (k, _) in enumerate(prio)}
        phase_of = dict(prio)

        def decider(undecided, _assignment):
            best_k, best_r = None, 10**9
            for k in undecided:
                r = rank.get(k)
                if r is not None and r < best_r:
                    best_k, best_r = k, r
            if best_k is None:
                return None
            return best_k, phase_of.get(best_k, True)

        return decider

    return factory


def _remap_priority(priority: list[tuple[str, bool]], hard) -> list[tuple[str, bool]]:
    """把生成时的 atom_key 映射到当前公式中同名/同结构原子（预处理后键可能变）。

    策略：若原 key 仍出现在 ``collect_clause_atoms`` 中则保留；否则按变量名子串模糊匹配
    「x0」相关比较原子，供 oracle 仍能优先模式守卫。
    """
    from omt_branching.solver.propagator_snapshot import collect_clause_atoms

    atoms = collect_clause_atoms(hard)
    key_set = {atom_key(a): a for a in atoms}
    out: list[tuple[str, bool]] = []
    for k, ph in priority:
        if k in key_set:
            out.append((k, ph))
    if out:
        return out
    # 回退：含 `_x0` 的比较原子（模式选择变量）
    x0_keys = sorted(k for k in key_set if "_x0" in k or "x0" in k)
    return [(k, True) for k in x0_keys[:4]]


def _probe_one(task: tuple) -> dict:
    """ProcessPool worker：单实例三臂测量。"""
    (
        instance_id,
        seed,
        n_vars,
        n_modes,
        ub,
        n_hard_disj,
        n_distractors,
        k,
        chi,
    ) = task
    rng = random.Random(seed)
    bundle = generate_branch_focus_lia_instance(
        instance_id,
        rng,
        n_vars=n_vars,
        n_modes=n_modes,
        ub=ub,
        chi=chi,
        n_hard_disj=n_hard_disj,
        k=k,
        n_distractors=n_distractors,
    )
    inst = bundle.instance
    hard, obj, sense = inst.as_tuple()

    try:
        csl = solve_omt_with_decider(
            hard, obj, sense, attach_propagator=False
        )
        vsids = solve_omt_with_decider(
            hard, obj, sense, decider_factory=None, attach_propagator=True
        )
        prio = _remap_priority(bundle.oracle_priority, hard)
        oracle = solve_omt_with_decider(
            hard,
            obj,
            sense,
            decider_factory=_priority_decider_factory(prio),
            attach_propagator=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "instance_id": instance_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seed": seed,
            "n_vars": n_vars,
        }

    def _pack(st: dict) -> dict:
        return {
            "value": st.get("value"),
            "rlimit": st.get("rlimit"),
            "weighted rlimit": st.get("weighted rlimit"),
            "check rlimit": st.get("check rlimit"),
            "conflicts": st.get("conflicts"),
            "decisions": st.get("decisions"),
            "iters": st.get("iters"),
        }

    rl_csl = float(csl["rlimit"] or 0)
    rl_v = float(vsids["rlimit"] or 0)
    rl_o = float(oracle["rlimit"] or 0)
    gap = abs(rl_csl - rl_v) / max(rl_csl, 1.0)
    ratio = rl_o / max(rl_v, 1.0)
    values_ok = csl.get("value") == vsids.get("value") == oracle.get("value")

    return {
        "instance_id": instance_id,
        "ok": True,
        "seed": seed,
        "n_vars": n_vars,
        "oracle_priority": bundle.oracle_priority,
        "check_sat_loop": _pack(csl),
        "vsids": _pack(vsids),
        "oracle": _pack(oracle),
        "csl_vsids_gap": gap,
        "oracle_vsids_ratio": ratio,
        "values_ok": values_ok,
        "bundle_params": {
            "n_modes": n_modes,
            "ub": ub,
            "n_hard_disj": n_hard_disj,
            "n_distractors": n_distractors,
            "k": k,
            "chi": chi,
        },
    }


def _pass_filters(row: dict, *, max_gap: float, max_ratio: float, min_vsids_rl: float) -> bool:
    if not row.get("ok") or not row.get("values_ok"):
        return False
    if row["csl_vsids_gap"] > max_gap:
        return False
    if row["oracle_vsids_ratio"] > max_ratio:
        return False
    vsids_rl = float((row.get("vsids") or {}).get("rlimit") or 0)
    if vsids_rl < min_vsids_rl:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="探针筛选分支敏感 OMT 实例")
    ap.add_argument("--candidates", type=int, default=30, help="候选生成数")
    ap.add_argument("--need", type=int, default=8, help="希望筛出的合格数（达标即停）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-vars", type=int, default=7)
    ap.add_argument("--max-vars", type=int, default=9)
    ap.add_argument("--n-modes", type=int, default=4)
    ap.add_argument("--ub", type=int, default=8)
    ap.add_argument("--n-hard-disj", type=int, default=36)
    ap.add_argument("--n-distractors", type=int, default=48)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chi", type=int, default=4)
    ap.add_argument(
        "--max-csl-vsids-gap",
        type=float,
        default=0.25,
        help="|CSL-VSIDS|/CSL 上限（默认 0.25）",
    )
    ap.add_argument(
        "--max-oracle-ratio",
        type=float,
        default=0.75,
        help="oracle/VSIDS rlimit 上限（默认 0.75，即至少省 25%）",
    )
    ap.add_argument(
        "--min-vsids-rlimit",
        type=float,
        default=5_000,
        help="过滤过易实例：VSIDS rlimit 下限",
    )
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    ap.add_argument("--save", action="store_true", help="把合格实例写入数据集目录")
    ap.add_argument("--dataset-dir", default=DEFAULT_OUT)
    ap.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "eval"],
        help="--save 时写入的划分",
    )
    ap.add_argument(
        "--metric",
        default="rlimit",
        choices=["rlimit", "check rlimit", "weighted rlimit"],
        help="用于 gap/ratio 的度量字段（重新从臂统计读取）",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks = []
    for i in range(args.candidates):
        n_vars = rng.randint(args.min_vars, args.max_vars)
        tasks.append(
            (
                f"bfocus{i}",
                args.seed + 10007 * i + n_vars,
                n_vars,
                args.n_modes,
                args.ub,
                args.n_hard_disj,
                args.n_distractors,
                args.k,
                args.chi,
            )
        )

    print(
        f"探针 {len(tasks)} 候选 | workers={args.workers} | "
        f"gap≤{args.max_csl_vsids_gap} oracle/vsids≤{args.max_oracle_ratio} "
        f"vsids_rl≥{args.min_vsids_rlimit:.0f}"
    )

    rows: list[dict] = []
    passed: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_probe_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            # 若指定非 rlimit 度量，重算 gap/ratio
            if row.get("ok") and args.metric != "rlimit":
                def _m(arm: str) -> float:
                    return float((row.get(arm) or {}).get(args.metric) or 0)

                csl_m, v_m, o_m = _m("check_sat_loop"), _m("vsids"), _m("oracle")
                row["csl_vsids_gap"] = abs(csl_m - v_m) / max(csl_m, 1.0)
                row["oracle_vsids_ratio"] = o_m / max(v_m, 1.0)
                row["metric_used"] = args.metric
            rows.append(row)
            ok = _pass_filters(
                row,
                max_gap=args.max_csl_vsids_gap,
                max_ratio=args.max_oracle_ratio,
                min_vsids_rl=args.min_vsids_rlimit,
            )
            tag = "PASS" if ok else ("ERR" if not row.get("ok") else "fail")
            if row.get("ok"):
                print(
                    f"  [{tag}] {row['instance_id']}: "
                    f"csl={row['check_sat_loop']['rlimit']:.0f} "
                    f"vsids={row['vsids']['rlimit']:.0f} "
                    f"oracle={row['oracle']['rlimit']:.0f} "
                    f"gap={row['csl_vsids_gap']:.3f} "
                    f"ratio={row['oracle_vsids_ratio']:.3f} "
                    f"val_ok={row['values_ok']}"
                )
            else:
                print(f"  [{tag}] {row['instance_id']}: {row.get('error')}")
            if ok:
                passed.append(row)
                if len(passed) >= args.need:
                    # 取消剩余（尽力而为）
                    for f in futs:
                        f.cancel()
                    break

    print(f"\n合格 {len(passed)} / 已测 {len(rows)}")
    if passed:
        gaps = [r["csl_vsids_gap"] for r in passed]
        ratios = [r["oracle_vsids_ratio"] for r in passed]
        print(
            f"  gap mean={sum(gaps)/len(gaps):.3f}  "
            f"oracle/vsids mean={sum(ratios)/len(ratios):.3f}"
        )

    out_json = Path(args.dataset_dir) / "probe_branch_focus.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": vars(args),
        "n_measured": len(rows),
        "n_passed": len(passed),
        "passed_ids": [r["instance_id"] for r in passed],
        "rows": rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"明细 -> {out_json}")

    if args.save and passed:
        # 按合格 seed 重放生成并落盘
        insts = []
        oracle_meta = {}
        for r in passed:
            bp = r["bundle_params"]
            rng = random.Random(r["seed"])
            bundle = generate_branch_focus_lia_instance(
                r["instance_id"],
                rng,
                n_vars=r["n_vars"],
                **bp,
            )
            insts.append(bundle.instance)
            oracle_meta[r["instance_id"]] = {
                "oracle_priority": r["oracle_priority"],
                "probe": {
                    "csl_vsids_gap": r["csl_vsids_gap"],
                    "oracle_vsids_ratio": r["oracle_vsids_ratio"],
                    "check_sat_loop": r["check_sat_loop"],
                    "vsids": r["vsids"],
                    "oracle": r["oracle"],
                },
            }
        # 保留其它划分里已有的 oracle / probe 元数据
        mp = Path(args.dataset_dir) / "manifest.json"
        old_meta: dict[tuple[str, str], dict] = {}
        if mp.is_file():
            old_man = json.loads(mp.read_text(encoding="utf-8"))
            for sp, ents in (old_man.get("splits") or {}).items():
                for e in ents:
                    iid = e.get("instance_id")
                    if iid and (e.get("oracle_priority") or e.get("branch_focus_probe")):
                        old_meta[(sp, iid)] = {
                            "oracle_priority": e.get("oracle_priority"),
                            "branch_focus_probe": e.get("branch_focus_probe"),
                        }
        entries = save_dataset(insts, args.dataset_dir, split=args.split)
        manifest = rebuild_manifest(args.dataset_dir, preserve_meta=True)
        for sp, ents in manifest.get("splits", {}).items():
            for e in ents:
                iid = e.get("instance_id")
                if sp == args.split and iid in oracle_meta:
                    e["oracle_priority"] = oracle_meta[iid]["oracle_priority"]
                    e["branch_focus_probe"] = oracle_meta[iid]["probe"]
                elif (sp, iid) in old_meta:
                    prev = old_meta[(sp, iid)]
                    if prev.get("oracle_priority"):
                        e["oracle_priority"] = prev["oracle_priority"]
                    if prev.get("branch_focus_probe"):
                        e["branch_focus_probe"] = prev["branch_focus_probe"]
        manifest["generator"] = "branch_focus_lia"
        manifest["params"] = {
            **(manifest.get("params") or {}),
            "branch_focus": True,
            "probe_filters": {
                "max_csl_vsids_gap": args.max_csl_vsids_gap,
                "max_oracle_ratio": args.max_oracle_ratio,
                "min_vsids_rlimit": args.min_vsids_rlimit,
            },
        }
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4, ensure_ascii=False)
        print(f"已保存 {len(entries)} 实例 -> {args.dataset_dir}/{args.split}/")
        print(f"manifest -> {mp}")
    elif args.save:
        print("无合格实例，跳过 --save")


if __name__ == "__main__":
    main()
