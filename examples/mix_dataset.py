"""拼合多个数据集，供跨规模/跨分布泛化研究。

从若干源数据集中按指定数量随机抽取 train/test 实例，复制 ``.smt2``，并在有缓存时
一并复制 ``ref/`` 与 ``lookahead/``，写入默认目录 ``examples/artifacts/dataset``，
生成新的 ``manifest.json``。

实例 id 默认：``v{n_vars}__{原id}``，其中 ``n_vars`` **以源数据集 manifest 条目为准**
（避免跨规模拼合冲突）。若仍冲突则追加源目录标签消歧。

运行::

    python -m examples.mix_dataset \\
      --source path/to/ds_small 10 5 \\
      --source path/to/ds_large 20 10 \\
      --seed 0 --force

每个 ``--source`` 后接三个参数：``目录  train抽取数  test抽取数``（0 表示该划分不抽）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from omt_branching.solver.binary_results import BINARY_SUBDIR, binary_result_path
from omt_branching.solver.decide_omt import (
    instance_manifest_entry,
    list_split_entries,
    smt2_to_instance,
)
from omt_branching.solver.lookahead_cache import LOOKAHEAD_SUBDIR, lookahead_result_path

ARTIFACTS = os.path.join(os.path.dirname(__file__), "artifacts")
DEFAULT_OUT_DIR = os.path.join(ARTIFACTS, "dataset")


def _source_tag(src_dir: Path, index: int) -> str:
    """从目录名生成合法、尽量短的源标签（仅作 id 冲突消歧）。"""
    raw = src_dir.resolve().name or f"src{index}"
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-") or f"src{index}"
    return tag


def _split_entries_prefer_manifest(src_dir: Path, split: str) -> list[dict]:
    """优先使用源 ``manifest.json`` 中的条目（含权威 ``n_vars``）；否则按磁盘现算。"""
    manifest_path = src_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = manifest.get("splits", {}).get(split, [])
        entries: list[dict] = []
        for e in raw:
            rel = e.get("smt2")
            iid = e.get("instance_id") or (Path(rel).stem if rel else None)
            if not iid:
                continue
            smt2_path = src_dir / rel if rel else src_dir / split / f"{iid}.smt2"
            if not smt2_path.is_file():
                continue
            item = dict(e)
            item["instance_id"] = iid
            item["smt2"] = rel.replace("\\", "/") if rel else f"{split}/{iid}.smt2"
            entries.append(item)
        if entries:
            return entries
    return list_split_entries(src_dir, split)


def _entry_n_vars(entry: dict, src_dir: Path) -> int:
    """变量数：优先 manifest 字段 ``n_vars``，缺失时再从 .smt2 推断。"""
    if entry.get("n_vars") is not None:
        return int(entry["n_vars"])
    rel = entry.get("smt2")
    iid = entry.get("instance_id")
    smt2_path = src_dir / rel if rel else src_dir / f"{iid}.smt2"
    if smt2_path.is_file():
        inst = smt2_to_instance(smt2_path, instance_id=iid or smt2_path.stem)
        return len(inst.variables)
    raise SystemExit(
        f"无法确定 n_vars（manifest 无字段且 smt2 不可读）: "
        f"{src_dir} id={iid}"
    )


def _new_instance_id(
    n_vars: int, old_id: str, *, disambiguator: str | None = None
) -> str:
    """默认 ``v{n_vars}__{old_id}``；冲突时追加 ``__{disambiguator}``。"""
    base = f"v{n_vars}__{old_id}"
    if disambiguator:
        return f"{base}__{disambiguator}"
    return base


def _sample_entries(
    entries: list[dict],
    n: int,
    rng: random.Random,
    *,
    split: str,
    src_dir: Path,
) -> list[dict]:
    if n < 0:
        raise ValueError(f"抽取数不能为负: {split} from {src_dir}")
    if n == 0:
        return []
    if n > len(entries):
        raise SystemExit(
            f"{src_dir}: 划分 {split} 仅有 {len(entries)} 个实例，无法抽取 {n}"
        )
    return rng.sample(list(entries), n)


def _copy_json_update_id(src: Path, dst: Path, new_id: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "instance_id" in payload:
        payload["instance_id"] = new_id
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _prune_split_files(split_dir: Path, keep_names: set[str], pattern: str) -> None:
    if not split_dir.is_dir():
        return
    for p in split_dir.glob(pattern):
        if p.name not in keep_names:
            p.unlink(missing_ok=True)


def _assign_unique_ids(
    planned: list[tuple[Path, dict, str, int]],
) -> list[tuple[Path, dict, str, int]]:
    """为 ``(src_dir, entry, tag, n_vars)`` 分配唯一 ``v{n}__{id}``，冲突则加源标签。"""
    tentative: list[str] = [
        _new_instance_id(n_vars, e["instance_id"])
        for _src, e, _tag, n_vars in planned
    ]
    counts: dict[str, int] = {}
    for nid in tentative:
        counts[nid] = counts.get(nid, 0) + 1
    collided = {nid for nid, c in counts.items() if c > 1}

    out: list[tuple[Path, dict, str, int]] = []
    used: set[str] = set()
    for (src_dir, e, tag, n_vars), cand in zip(planned, tentative):
        if cand in collided:
            nid = _new_instance_id(n_vars, e["instance_id"], disambiguator=tag)
        else:
            nid = cand
        if nid in used:
            k = 2
            while f"{nid}_{k}" in used:
                k += 1
            nid = f"{nid}_{k}"
        used.add(nid)
        out.append((src_dir, e, nid, n_vars))
    return out


def mix_datasets(
    sources: list[tuple[Path, int, int]],
    out_dir: Path,
    *,
    seed: int = 0,
    force: bool = False,
) -> dict:
    """拼合多个源数据集到 ``out_dir``，返回新 manifest。

    ``sources`` 每项为 ``(源目录, train抽取数, test抽取数)``。
    新 ``instance_id`` 默认为 ``v{n_vars}__{原id}``（``n_vars`` 取自源 manifest）。
    """
    if not sources:
        raise ValueError("至少需要一个 --source")

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.rglob("*.smt2")) and not force:
        raise SystemExit(
            f"输出目录已有 .smt2：{out_dir}\n"
            f"如需覆盖请加 --force，或用 --out-dir 指定其它目录"
        )

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_raw: dict[str, list[tuple[Path, dict, str, int]]] = {
        "train": [],
        "test": [],
    }
    source_meta: list[dict] = []

    for i, (src_dir, n_train, n_test) in enumerate(sources):
        src_dir = Path(src_dir)
        if not src_dir.is_dir():
            raise SystemExit(f"源数据集不存在: {src_dir}")
        tag = _source_tag(src_dir, i)
        existing_tags = {m["tag"] for m in source_meta}
        base_tag = tag
        k = 2
        while tag in existing_tags:
            tag = f"{base_tag}_{k}"
            k += 1

        train_ents = _split_entries_prefer_manifest(src_dir, "train")
        test_ents = _split_entries_prefer_manifest(src_dir, "test")
        picked_train = _sample_entries(
            train_ents, n_train, rng, split="train", src_dir=src_dir
        )
        picked_test = _sample_entries(
            test_ents, n_test, rng, split="test", src_dir=src_dir
        )
        source_meta.append({
            "path": str(src_dir.resolve()),
            "tag": tag,
            "available_train": len(train_ents),
            "available_test": len(test_ents),
            "take_train": n_train,
            "take_test": n_test,
            "picked_train_ids": [e["instance_id"] for e in picked_train],
            "picked_test_ids": [e["instance_id"] for e in picked_test],
        })
        for e in picked_train:
            plan_raw["train"].append(
                (src_dir, e, tag, _entry_n_vars(e, src_dir))
            )
        for e in picked_test:
            plan_raw["test"].append(
                (src_dir, e, tag, _entry_n_vars(e, src_dir))
            )

    # 全局按 v{n}__{id} 命名并消歧（train/test 共用命名空间）
    flat: list[tuple[str, Path, dict, str, int]] = []
    for split, rows in plan_raw.items():
        for src_dir, e, tag, n_vars in rows:
            flat.append((split, src_dir, e, tag, n_vars))
    assigned = _assign_unique_ids([(s, e, t, n) for _, s, e, t, n in flat])
    plan: dict[str, list[tuple[Path, dict, str, int]]] = {
        "train": [],
        "test": [],
    }
    for (split, _, _, _, _), (src_dir, e, new_id, n_vars) in zip(flat, assigned):
        plan[split].append((src_dir, e, new_id, n_vars))

    all_new = [nid for sp in plan.values() for _, _, nid, _ in sp]
    if len(all_new) != len(set(all_new)):
        raise SystemExit("拼合后 instance_id 仍冲突")

    splits_manifest: dict[str, list] = {}
    stats = {
        "smt2": 0,
        "binary": 0,
        "lookahead": 0,
        "binary_missing": 0,
        "lookahead_missing": 0,
    }

    for split, items in plan.items():
        if not items:
            continue
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        keep_smt2: set[str] = set()
        keep_bin: set[str] = set()
        keep_la: set[str] = set()
        entries: list[dict] = []

        for src_dir, old_e, new_id, n_vars in items:
            old_id = old_e["instance_id"]
            src_smt2 = src_dir / old_e["smt2"]
            if not src_smt2.is_file():
                src_smt2 = src_dir / split / f"{old_id}.smt2"
            if not src_smt2.is_file():
                raise SystemExit(f"缺少 smt2: {src_smt2}")

            dst_name = f"{new_id}.smt2"
            dst_smt2 = split_dir / dst_name
            shutil.copy2(src_smt2, dst_smt2)
            keep_smt2.add(dst_name)
            stats["smt2"] += 1

            inst = smt2_to_instance(
                dst_smt2,
                instance_id=new_id,
                family=old_e.get("family", "imported"),
                description=old_e.get("description")
                or f"mixed from {old_id} (n_vars={n_vars})",
            )
            rel = f"{split}/{dst_name}"
            entry = instance_manifest_entry(inst, smt2_relpath=rel)
            entry["n_vars"] = n_vars
            entry["source_path"] = str(src_dir.resolve())
            entry["source_instance_id"] = old_id
            entry["source_n_vars"] = n_vars
            entries.append(entry)

            src_bin = binary_result_path(src_dir, old_id, split=split)
            if src_bin.is_file():
                dst_bin = binary_result_path(out_dir, new_id, split=split)
                _copy_json_update_id(src_bin, dst_bin, new_id)
                keep_bin.add(dst_bin.name)
                stats["binary"] += 1
            else:
                stats["binary_missing"] += 1

            src_la = lookahead_result_path(src_dir, old_id, split=split)
            if src_la.is_file():
                dst_la = lookahead_result_path(out_dir, new_id, split=split)
                _copy_json_update_id(src_la, dst_la, new_id)
                keep_la.add(dst_la.name)
                stats["lookahead"] += 1
            else:
                stats["lookahead_missing"] += 1

        _prune_split_files(split_dir, keep_smt2, "*.smt2")
        _prune_split_files(out_dir / BINARY_SUBDIR / split, keep_bin, "*.json")
        _prune_split_files(out_dir / LOOKAHEAD_SUBDIR / split, keep_la, "*.json")
        splits_manifest[split] = entries

    for split in ("train", "test"):
        if split in splits_manifest:
            continue
        for sub in (
            out_dir / split,
            out_dir / BINARY_SUBDIR / split,
            out_dir / LOOKAHEAD_SUBDIR / split,
        ):
            if sub.is_dir():
                for p in sub.glob("*"):
                    if p.is_file():
                        p.unlink(missing_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": "mix",
        "params": {
            "seed": seed,
            "train": len(splits_manifest.get("train", [])),
            "test": len(splits_manifest.get("test", [])),
            "id_scheme": "v{n_vars}__{source_instance_id}",
            "sources": source_meta,
        },
        "seeds": {"mix": seed},
        "splits": splits_manifest,
        "mix_stats": stats,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(
        description="拼合多数据集（smt2 + ref/lookahead 缓存）到默认 dataset 目录"
    )
    ap.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("DIR", "TRAIN_N", "TEST_N"),
        required=True,
        help="源目录及从中抽取的 train/test 数量，可重复多次",
    )
    ap.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUT_DIR}）",
    )
    ap.add_argument("--seed", type=int, default=0, help="随机抽样种子")
    ap.add_argument(
        "--force",
        action="store_true",
        help="输出目录已有 .smt2 时仍覆盖",
    )
    args = ap.parse_args()

    sources: list[tuple[Path, int, int]] = []
    for dir_s, train_s, test_s in args.source:
        try:
            n_train = int(train_s)
            n_test = int(test_s)
        except ValueError as exc:
            raise SystemExit(
                f"--source 的 TRAIN_N/TEST_N 须为整数: {dir_s} {train_s} {test_s}"
            ) from exc
        sources.append((Path(dir_s), n_train, n_test))

    manifest = mix_datasets(
        sources,
        Path(args.out_dir),
        seed=args.seed,
        force=args.force,
    )
    stats = manifest.get("mix_stats", {})
    print(f"拼合完成 -> {Path(args.out_dir).resolve()}")
    print(
        f"train={manifest['params']['train']}  test={manifest['params']['test']}  "
        f"id_scheme={manifest['params']['id_scheme']}  "
        f"smt2={stats.get('smt2')}  ref={stats.get('binary')} "
        f"(缺 {stats.get('binary_missing')})  "
        f"lookahead={stats.get('lookahead')} (缺 {stats.get('lookahead_missing')})"
    )
    print(f"manifest -> {Path(args.out_dir) / 'manifest.json'}")
    if stats.get("binary_missing"):
        print(
            "提示: 部分实例无 ref 缓存，评测/RL 前请运行:\n"
            f"  python -m examples.solve_dataset_binary --dataset-dir {args.out_dir}"
        )


if __name__ == "__main__":
    main()
