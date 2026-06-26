"""异构关系消息传递 GNN (R-GCN 风格)。

不依赖 PyTorch Geometric：节点按类型存成 ``dict[NodeType, Tensor]``，每条关系一个
消息 MLP，按目标节点类型用 ``index_add_`` 做 mean 聚合，再做残差 + LayerNorm 更新。
支持边特征（与源节点表示拼接后送入消息 MLP）。

对应 ``plan.md`` 第 6.1 节：异构 message passing，全局 search_state 通过
``state_to_*`` 关系注入每个候选节点。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.input.graph_builder import FeatureSpec
from omt_branching.interfaces import EDGE_SCHEMA, EdgeType, NodeType


class RelationalLayer(nn.Module):
    """一层异构消息传递。"""

    def __init__(self, hidden: int, edge_dims: dict[EdgeType, int]):
        super().__init__()
        self.hidden = hidden
        # 每个目标节点类型的自环变换
        self.self_lin = nn.ModuleDict(
            {nt.value: nn.Linear(hidden, hidden) for nt in NodeType.all()}
        )
        # 每个关系的消息变换：输入 = 源表示 (+ 边特征)
        self.msg_lin = nn.ModuleDict(
            {
                et.value: nn.Linear(hidden + edge_dims.get(et, 0), hidden)
                for et in EdgeType
            }
        )
        self.norm = nn.ModuleDict(
            {nt.value: nn.LayerNorm(hidden) for nt in NodeType.all()}
        )
        self.act = nn.ReLU()

    def forward(self, h: dict[NodeType, torch.Tensor], g: HeteroGraph) -> dict[NodeType, torch.Tensor]:
        # 按目标类型累积消息
        agg = {nt: torch.zeros_like(x) for nt, x in h.items()}
        deg = {nt: torch.zeros(x.shape[0], 1, device=x.device) for nt, x in h.items()}

        for et in g.edge_types():
            src_t, dst_t = EDGE_SCHEMA[et]
            if src_t not in h or dst_t not in h:
                continue
            idx = g.edge_index[et]
            src_idx, dst_idx = idx[0], idx[1]
            src_h = h[src_t][src_idx]
            if et in g.edge_features and g.edge_features[et].numel() > 0:
                src_h = torch.cat([src_h, g.edge_features[et]], dim=1)
            msg = self.msg_lin[et.value](src_h)
            agg[dst_t].index_add_(0, dst_idx, msg)
            deg[dst_t].index_add_(0, dst_idx, torch.ones_like(dst_idx, dtype=msg.dtype).unsqueeze(1))

        out = {}
        for nt, x in h.items():
            mean_msg = agg[nt] / deg[nt].clamp_min(1.0)
            upd = self.self_lin[nt.value](x) + mean_msg
            out[nt] = self.norm[nt.value](x + self.act(upd))
        return out


class HeteroEncoder(nn.Module):
    """输入投影 + L 层关系消息传递，输出每个节点的 embedding。"""

    def __init__(self, feature_spec: FeatureSpec, hidden: int = 64, num_layers: int = 3):
        super().__init__()
        self.spec = feature_spec
        self.hidden = hidden
        self.input_proj = nn.ModuleDict(
            {
                nt.value: nn.Linear(max(1, feature_spec.node_dim(nt)), hidden)
                for nt in NodeType.all()
            }
        )
        self.layers = nn.ModuleList(
            [RelationalLayer(hidden, feature_spec.edge_dims) for _ in range(num_layers)]
        )

    def forward(self, g: HeteroGraph) -> dict[NodeType, torch.Tensor]:
        h: dict[NodeType, torch.Tensor] = {}
        for nt in g.node_types():
            x = g.node_features[nt]
            if x.shape[1] == 0:  # 理论上不会发生，留作健壮性
                x = torch.zeros(x.shape[0], 1, device=x.device)
            h[nt] = self.input_proj[nt.value](x)
        for layer in self.layers:
            h = layer(h, g)
        return h
