"""分支策略网络：编码器 + 多头，产出 :class:`PolicyOutput`。

``PolicyOutput`` 是模型部分与输出部分之间的中间契约：保存所有原始分数 / logit、
候选 mask（图内局部索引）以及对源 :class:`HeteroGraph` 的引用，使输出解码器能把
分数映射回求解器原始 id。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, FeatureSpec
from omt_branching.interfaces import NodeType
from omt_branching.model.gnn import HeteroEncoder
from omt_branching.model.heads import (
    AuxiliaryHeads,
    BranchingHead,
    IntegerBranchHead,
    PhaseHead,
)


@dataclass
class PolicyConfig:
    hidden: int = 64
    num_layers: int = 3
    use_auxiliary: bool = True


@dataclass
class PolicyOutput:
    """模型一次前向的全部输出。"""

    bool_branch_scores: torch.Tensor       # [num_bool] 未归一化分数
    phase_logits: torch.Tensor             # [num_bool] 取真 logit
    int_branch_scores: torch.Tensor        # [num_numeric]
    int_dir_logits: torch.Tensor           # [num_numeric] 向上取整 logit
    aux: dict[str, torch.Tensor] = field(default_factory=dict)

    candidate_bool_local: list[int] = field(default_factory=list)
    candidate_numeric_local: list[int] = field(default_factory=list)
    graph: Optional[HeteroGraph] = None

    # --------- 便捷方法：在候选集合上做 masked softmax ---------
    def masked_bool_probs(self) -> torch.Tensor:
        """返回长度 = num_bool 的概率向量，非候选位置为 0。"""
        return _masked_softmax(self.bool_branch_scores, self.candidate_bool_local)

    def masked_numeric_probs(self) -> torch.Tensor:
        return _masked_softmax(self.int_branch_scores, self.candidate_numeric_local)


def _masked_softmax(scores: torch.Tensor, candidate_local: list[int]) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    mask = torch.full_like(scores, float("-inf"))
    if candidate_local:
        idx = torch.tensor(candidate_local, dtype=torch.long, device=scores.device)
        mask[idx] = scores[idx]
    else:
        mask = scores
    probs = F.softmax(mask, dim=0)
    return torch.nan_to_num(probs, nan=0.0)


class BranchingPolicy(nn.Module):
    """完整策略网络。"""

    def __init__(self, feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
                 config: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.config = config
        self.encoder = HeteroEncoder(feature_spec, config.hidden, config.num_layers)
        self.branch_head = BranchingHead(config.hidden)
        self.phase_head = PhaseHead(config.hidden)
        self.int_head = IntegerBranchHead(config.hidden)
        self.aux_heads = AuxiliaryHeads(config.hidden) if config.use_auxiliary else None

    def forward(self, g: HeteroGraph) -> PolicyOutput:
        emb = self.encoder(g)
        int_scores, int_dirs = self.int_head(emb)
        out = PolicyOutput(
            bool_branch_scores=self.branch_head(emb),
            phase_logits=self.phase_head(emb),
            int_branch_scores=int_scores,
            int_dir_logits=int_dirs,
            aux=self.aux_heads(emb) if self.aux_heads is not None else {},
            candidate_bool_local=list(g.meta.get("candidate_bool_local", [])),
            candidate_numeric_local=list(g.meta.get("candidate_numeric_local", [])),
            graph=g,
        )
        return out

    # 推理用：不建图梯度
    @torch.no_grad()
    def infer(self, g: HeteroGraph) -> PolicyOutput:
        self.eval()
        return self.forward(g)
