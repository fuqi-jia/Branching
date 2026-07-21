"""参考求解结果缓存（目录名 ``ref/``）。

同一 ``.smt2`` 上参考结果可反复复用（评测 / RL reward）。布局（相对数据集根目录）::

    ref/<split>/<instance_id>.json

缓存约定：

- ``value``：始终来自 z3 **binary** 最优目标值；
- ``rlimit``：RL / 剪枝用的参考资源——当公平 VSIDS 目标值与 binary 一致时取
  VSIDS 的 ``rlimit``（否则回退 binary 的 ``rlimit``）；
- ``binary``：z3 二进制统计；
- ``vsids``：公平 VSIDS（预处理 + 挂 propagator、decide 恒 defer）；
- ``check_sat_loop``：原无 propagator 的 Solver 线性搜索臂（同样预处理）。

并行求解时每实例独立文件，完成后立即写入，无共享锁。
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

REF_SUBDIR = "ref"
# 旧名兼容（曾用 binary/）
BINARY_SUBDIR = REF_SUBDIR

_NESTED_ARMS = ("binary", "vsids", "check_sat_loop")


def ref_result_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    """返回某实例 ref 结果 JSON 路径。"""
    return Path(dataset_dir) / REF_SUBDIR / split / f"{instance_id}.json"


def binary_result_path(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Path:
    """``ref_result_path`` 的旧名别名。"""
    return ref_result_path(dataset_dir, instance_id, split=split)


def has_ref_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return ref_result_path(dataset_dir, instance_id, split=split).is_file()


def has_binary_result(dataset_dir, instance_id: str, *, split: str) -> bool:
    return has_ref_result(dataset_dir, instance_id, split=split)


def _json_default(v: Any):
    if isinstance(v, Fraction):
        return str(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def serialize_binary_result(result: dict) -> dict:
    """把求解结果 dict 转为 JSON 可序列化形式（保留全字段）。"""
    out: dict = {}
    for k, v in result.items():
        if k in _NESTED_ARMS and isinstance(v, dict):
            out[k] = serialize_binary_result(v)
        elif k == "z3_stats" and isinstance(v, dict):
            out[k] = {sk: _json_default(sv) for sk, sv in v.items()}
        else:
            out[k] = _json_default(v)
    return out


def _parse_value(raw) -> Any:
    if raw is None or isinstance(raw, (int, float, Fraction)):
        return raw
    if isinstance(raw, str):
        try:
            return Fraction(raw)
        except (ValueError, ZeroDivisionError):
            return raw
    return raw


def deserialize_binary_result(payload: dict) -> dict:
    """读回落盘结果：把 ``value``（及嵌套臂的 value）尽量还原为 ``Fraction``。"""
    out = dict(payload)
    if "value" in out:
        out["value"] = _parse_value(out["value"])
    for nest in _NESTED_ARMS:
        nested = out.get(nest)
        if isinstance(nested, dict) and "value" in nested:
            nested = dict(nested)
            nested["value"] = _parse_value(nested["value"])
            out[nest] = nested
    return out


def is_fair_vsids_cache(ref: dict | None) -> bool:
    """判断 ref 是否含公平 VSIDS + check-sat-loop 缓存。

    公平 VSIDS 挂 prop 后 ``decisions`` 为 int（通常 0）；旧缓存无 prop 时为
    ``None``。同时要求 ``check_sat_loop`` 非空。
    """
    if not isinstance(ref, dict):
        return False
    vsids = ref.get("vsids")
    csl = ref.get("check_sat_loop")
    if not isinstance(vsids, dict) or not vsids:
        return False
    if not isinstance(csl, dict) or not csl:
        return False
    return vsids.get("decisions") is not None


def build_ref_payload(
    binary_result: dict,
    vsids_result: dict | None = None,
    check_sat_loop_result: dict | None = None,
) -> dict:
    """由 binary / 公平 VSIDS / check-sat-loop 结果构造缓存 payload。

    - ``value`` ← binary
    - ``rlimit`` ← 公平 VSIDS 的 ``rlimit``（目标值与 binary 一致时）否则 binary
    """
    bin_val = binary_result.get("value")
    vsids = vsids_result or {}
    csl = check_sat_loop_result or {}
    vsids_val = vsids.get("value")
    match = (
        bin_val is not None
        and vsids_val is not None
        and vsids_val == bin_val
    )
    if match:
        ref_rl = vsids.get("rlimit")
        source = "vsids"
    else:
        ref_rl = binary_result.get("rlimit")
        source = "binary"

    payload = {
        "value": bin_val,
        "rlimit": ref_rl,
        "rlimit_source": source,
        "vsids_match": bool(match),
        "binary_rlimit": binary_result.get("rlimit"),
        "vsids_rlimit": vsids.get("rlimit"),
        "check_sat_loop_rlimit": csl.get("rlimit"),
        "status": binary_result.get("status"),
        "time_ms": binary_result.get("time_ms"),
        "conflicts": binary_result.get("conflicts"),
        "decisions": binary_result.get("decisions"),
        "binary": dict(binary_result),
        "vsids": dict(vsids) if vsids else {},
        "check_sat_loop": dict(csl) if csl else {},
    }
    return payload


def vsids_stats_from_ref(ref: dict) -> dict:
    """取出缓存中的公平 VSIDS 统计。"""
    nested = ref.get("vsids")
    if isinstance(nested, dict) and nested:
        return nested
    return {}


def check_sat_loop_stats_from_ref(ref: dict) -> dict:
    """取出缓存中的 check-sat-loop 统计。"""
    nested = ref.get("check_sat_loop")
    if isinstance(nested, dict) and nested:
        return nested
    return {}


def save_binary_result(
    dataset_dir,
    instance_id: str,
    result: dict,
    *,
    split: str,
) -> Path:
    """立刻把结果写入 ``ref/``（先写临时文件再 rename，避免半截 JSON）。"""
    path = ref_result_path(dataset_dir, instance_id, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_binary_result(result)
    payload.setdefault("instance_id", instance_id)
    payload.setdefault("split", split)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def load_binary_result(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Optional[dict]:
    """加载单实例 ref 结果；不存在返回 ``None``。"""
    path = ref_result_path(dataset_dir, instance_id, split=split)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return deserialize_binary_result(json.load(f))


def binary_stats_from_ref(ref: dict) -> dict:
    """三臂对比用：取出缓存中的 binary 原始统计。"""
    nested = ref.get("binary")
    if isinstance(nested, dict) and nested:
        return nested
    return ref


def binary_rlimit(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
) -> Optional[int]:
    """RL reward 用：返回缓存的参考 ``rlimit``（公平 VSIDS 优先，见模块文档）。"""
    res = load_binary_result(dataset_dir, instance_id, split=split)
    if res is None:
        return None
    rl = res.get("rlimit")
    return int(rl) if rl is not None else None


def binary_value(
    dataset_dir,
    instance_id: str,
    *,
    split: str,
):
    """RL / match 用：返回缓存的最优 ``value``（始终来自 binary）。"""
    res = load_binary_result(dataset_dir, instance_id, split=split)
    if res is None:
        return None
    return res.get("value")


def load_binary_results(
    dataset_dir,
    *,
    split: str | None = None,
) -> dict[str, dict]:
    """批量加载 ``ref/`` 下结果，键为 ``instance_id``。

    ``split`` 为 ``None`` 时扫描所有划分；同 id 后写覆盖先写（通常各 split id 不冲突）。
    """
    root = Path(dataset_dir) / REF_SUBDIR
    if not root.is_dir():
        return {}
    splits = [split] if split is not None else sorted(
        p.name for p in root.iterdir() if p.is_dir()
    )
    out: dict[str, dict] = {}
    for sp in splits:
        sp_dir = root / sp
        if not sp_dir.is_dir():
            continue
        for path in sorted(sp_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                payload = deserialize_binary_result(json.load(f))
            iid = payload.get("instance_id") or path.stem
            out[iid] = payload
    return out


def missing_binary_ids(
    dataset_dir,
    entries: list[dict],
    *,
    split: str,
) -> list[str]:
    """给出 manifest 条目中尚未有 ref 结果的 ``instance_id`` 列表。"""
    missing: list[str] = []
    for e in entries:
        iid = e["instance_id"]
        if not has_ref_result(dataset_dir, iid, split=split):
            missing.append(iid)
    return missing


__all__ = [
    "REF_SUBDIR",
    "BINARY_SUBDIR",
    "ref_result_path",
    "binary_result_path",
    "has_ref_result",
    "has_binary_result",
    "serialize_binary_result",
    "deserialize_binary_result",
    "is_fair_vsids_cache",
    "build_ref_payload",
    "save_binary_result",
    "load_binary_result",
    "binary_stats_from_ref",
    "vsids_stats_from_ref",
    "check_sat_loop_stats_from_ref",
    "binary_rlimit",
    "binary_value",
    "load_binary_results",
    "missing_binary_ids",
]
