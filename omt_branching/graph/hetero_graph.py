"""轻量异构图容器。

刻意不依赖 PyTorch Geometric，只用 ``torch.Tensor`` 保存：

- ``node_features[NodeType] -> FloatTensor[num_nodes, feat_dim]``
- ``edge_index[EdgeType]   -> LongTensor[2, num_edges]`` (行 0 = src, 行 1 = dst)
- ``edge_features[EdgeType]-> FloatTensor[num_edges, edge_dim]`` (可选)

同时维护一份 ``id_maps``，把求解器原始 ID（任意可哈希对象）映射到图内连续索引，
方便输出部分把节点分数解码回求解器变量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import torch

from omt_branching.interfaces import EDGE_SCHEMA, EdgeType, NodeType


@dataclass
class HeteroGraph:
    """异构图，节点/边按类型分桶存储。"""

    node_features: dict[NodeType, torch.Tensor] = field(default_factory=dict)
    edge_index: dict[EdgeType, torch.Tensor] = field(default_factory=dict)
    edge_features: dict[EdgeType, torch.Tensor] = field(default_factory=dict)

    #: NodeType -> (求解器原始 id -> 图内连续索引)
    id_maps: dict[NodeType, dict[Hashable, int]] = field(default_factory=dict)
    #: NodeType -> (图内索引 -> 求解器原始 id)，由 ``finalize`` 反推得到
    rev_id_maps: dict[NodeType, dict[int, Hashable]] = field(default_factory=dict)

    #: 任意附加元数据（如候选变量列表、快照时间戳）
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # 基本查询
    # ------------------------------------------------------------------ #
    def num_nodes(self, ntype: NodeType) -> int:
        feats = self.node_features.get(ntype)
        return 0 if feats is None else int(feats.shape[0])

    def num_edges(self, etype: EdgeType) -> int:
        idx = self.edge_index.get(etype)
        return 0 if idx is None else int(idx.shape[1])

    def feature_dim(self, ntype: NodeType) -> int:
        feats = self.node_features.get(ntype)
        return 0 if feats is None else int(feats.shape[1])

    def node_types(self) -> list[NodeType]:
        return [nt for nt in NodeType.all() if self.num_nodes(nt) > 0]

    def edge_types(self) -> list[EdgeType]:
        return [et for et in self.edge_index if self.num_edges(et) > 0]

    def local_index(self, ntype: NodeType, solver_id: Hashable) -> int | None:
        """求解器原始 id -> 图内索引；不存在返回 None。"""
        return self.id_maps.get(ntype, {}).get(solver_id)

    def solver_id(self, ntype: NodeType, local_idx: int) -> Hashable | None:
        """图内索引 -> 求解器原始 id。"""
        return self.rev_id_maps.get(ntype, {}).get(local_idx)

    # ------------------------------------------------------------------ #
    # 设备 / 校验
    # ------------------------------------------------------------------ #
    def to(self, device: torch.device | str) -> "HeteroGraph":
        self.node_features = {k: v.to(device) for k, v in self.node_features.items()}
        self.edge_index = {k: v.to(device) for k, v in self.edge_index.items()}
        self.edge_features = {k: v.to(device) for k, v in self.edge_features.items()}
        return self

    def copy_to(self, device: torch.device | str) -> "HeteroGraph":
        """非原地拷贝到 ``device``（共享 id_maps / meta 引用，供只读推理）。"""
        return HeteroGraph(
            node_features={k: v.to(device) for k, v in self.node_features.items()},
            edge_index={k: v.to(device) for k, v in self.edge_index.items()},
            edge_features={k: v.to(device) for k, v in self.edge_features.items()},
            id_maps=self.id_maps,
            rev_id_maps=self.rev_id_maps,
            meta=self.meta,
        )

    def finalize(self) -> "HeteroGraph":
        """构图结束后调用：生成反向 id 映射并做一致性校验。"""
        self.rev_id_maps = {
            nt: {idx: sid for sid, idx in m.items()} for nt, m in self.id_maps.items()
        }
        for etype, idx in self.edge_index.items():
            if idx.numel() == 0:
                continue
            src_t, dst_t = EDGE_SCHEMA[etype]
            n_src, n_dst = self.num_nodes(src_t), self.num_nodes(dst_t)
            if int(idx[0].max()) >= n_src or int(idx[1].max()) >= n_dst:
                raise ValueError(
                    f"边 {etype} 索引越界: src<{n_src}, dst<{n_dst}, got {idx.max(dim=1).values.tolist()}"
                )
        return self

    def summary(self) -> str:
        nodes = ", ".join(f"{nt.value}:{self.num_nodes(nt)}" for nt in self.node_types())
        edges = ", ".join(f"{et.value}:{self.num_edges(et)}" for et in self.edge_types())
        return f"HeteroGraph(nodes=[{nodes}], edges=[{edges}])"
