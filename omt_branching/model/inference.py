"""部署期推理 (plan 6.2 / 8.3)。

包装 :class:`BranchingPolicy`，加入工程化控制:

- **规模门限**：图节点数超过 ``max_total_nodes`` 时跳过 GNN，建议回退（OOD/开销）。
- **耗时统计**：记录单次推理耗时，超过 ``time_budget_ms`` 标记 fallback。
- **置信度门限**：候选 top-1 概率低于 ``min_confidence`` 时标记 fallback。

诊断信息写入 ``graph.meta['inference']``，供输出解码器决定是否启用 GNN 建议。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from omt_branching.model.device import gnn_device

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy, PolicyOutput


@dataclass
class InferenceConfig:
    device: str = field(default_factory=gnn_device)
    max_total_nodes: int = 200_000     # 超过则跳过推理直接回退
    time_budget_ms: Optional[float] = None  # 超时标记 fallback；None=不限
    min_confidence: float = 0.0        # top-1 候选概率门限
    refocus_interval: int = 0          # >0 时配合服务层做周期性 refocus


class InferenceEngine:
    """对单张图做一次（无梯度）推理并产出诊断。"""

    def __init__(self, policy: BranchingPolicy, config: InferenceConfig = InferenceConfig()):
        self.policy = policy.to(config.device)
        self.config = config

    def run(self, g: HeteroGraph) -> Optional[PolicyOutput]:
        """返回 PolicyOutput；规模超限则返回 None（求解器应回退）。

        诊断（耗时 / 置信度 / fallback / reason）写入 ``g.meta['inference']``。
        """
        total_nodes = sum(g.num_nodes(nt) for nt in NodeType.all())
        if total_nodes > self.config.max_total_nodes:
            g.meta["inference"] = {
                "time_ms": 0.0, "confidence": 0.0,
                "fallback": True, "reason": "graph_too_large", "total_nodes": total_nodes,
            }
            return None

        g = g.to(self.config.device)
        t0 = time.perf_counter()
        out = self.policy.infer(g)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        probs = out.masked_bool_probs()
        confidence = float(probs.max()) if probs.numel() > 0 else 0.0

        fallback, reason = False, "ok"
        if self.config.time_budget_ms is not None and dt_ms > self.config.time_budget_ms:
            fallback, reason = True, "time_budget_exceeded"
        elif confidence < self.config.min_confidence:
            fallback, reason = True, "low_confidence"

        g.meta["inference"] = {
            "time_ms": dt_ms, "confidence": confidence,
            "fallback": fallback, "reason": reason, "total_nodes": total_nodes,
        }
        return out
