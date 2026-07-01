"""训练结果持久化接口。

统一封装策略权重与训练历史的保存/加载，供 imitation 训练与 RL 训练共用：

- :func:`save_policy` / :func:`load_policy` / :func:`load_policy_into`：策略权重
  （连同 :class:`PolicyConfig` 与任意 ``meta``）的 checkpoint 读写。
- :func:`save_history` / :func:`load_history`：训练历史（指标列表）以 JSON 落盘。

权重文件用 ``torch.save``（pickle）；只应加载可信来源的 checkpoint。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Optional

import torch

from omt_branching.model.policy import BranchingPolicy, PolicyConfig


def save_policy(policy: BranchingPolicy, path, meta: Optional[dict] = None) -> None:
    """保存策略权重 + 结构配置 + 附加 ``meta`` 到 ``path``。"""
    config = getattr(policy, "config", None)
    payload = {
        "format": "omt_branching.policy",
        "version": 1,
        "state_dict": policy.state_dict(),
        "policy_config": asdict(config) if is_dataclass(config) else None,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_policy(path, map_location: str = "cpu") -> tuple[BranchingPolicy, dict]:
    """从 ``path`` 重建并返回 ``(policy, meta)``（按保存的 PolicyConfig 构造）。"""
    payload = _load_payload(path, map_location)
    cfg_dict = payload.get("policy_config")
    config = PolicyConfig(**cfg_dict) if cfg_dict else PolicyConfig()
    policy = BranchingPolicy(config=config)
    policy.load_state_dict(payload["state_dict"])
    policy.to(map_location)
    return policy, dict(payload.get("meta", {}))


def load_policy_into(policy: BranchingPolicy, path, map_location: str = "cpu") -> dict:
    """把 ``path`` 的权重加载进已有 ``policy``，返回附带的 ``meta``。"""
    payload = _load_payload(path, map_location)
    policy.load_state_dict(payload["state_dict"])
    policy.to(map_location)
    return dict(payload.get("meta", {}))


def save_history(history: list[dict], path) -> None:
    """把训练历史（指标 dict 列表）写成 JSON。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=_json_default)


def load_history(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_payload(path, map_location: str) -> dict:
    try:  # PyTorch 2.0+：优先安全加载
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # 老版本无 weights_only 参数
        return torch.load(path, map_location=map_location)


def _json_default(o: Any):
    # 兜底把不可 JSON 序列化的对象（如 Fraction）转成字符串/浮点。
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


__all__ = [
    "save_policy",
    "load_policy",
    "load_policy_into",
    "save_history",
    "load_history",
]
