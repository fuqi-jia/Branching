"""各预测头 (plan 6.1)。

- :class:`BranchingHead`     : 布尔候选 branching 分数（ranking / pointer）。
- :class:`PhaseHead`         : 布尔变量 polarity 分数 (取真 logit)。
- :class:`IntegerBranchHead` : 整数变量 B&B split 分数 + 方向 logit。
- :class:`AuxiliaryHeads`    : 多任务辅助头（冲突概率、目标改善、unsat core、子树规模）。

所有头都接收 :class:`HeteroEncoder` 输出的节点 embedding，输出未归一化分数 / logit；
softmax / mask 在策略网络层完成。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from omt_branching.interfaces import NodeType


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class BranchingHead(nn.Module):
    """布尔候选 branching 分数。输入 bool_var embedding -> 标量分数。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = _mlp(hidden, hidden, 1)

    def forward(self, emb: dict[NodeType, torch.Tensor]) -> torch.Tensor:
        h = emb.get(NodeType.BOOL_VAR)
        if h is None or h.shape[0] == 0:
            return torch.zeros(0, device=_dev(emb))
        return self.net(h).squeeze(-1)


class PhaseHead(nn.Module):
    """布尔变量 polarity：输出取真的 logit。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = _mlp(hidden, hidden, 1)

    def forward(self, emb: dict[NodeType, torch.Tensor]) -> torch.Tensor:
        h = emb.get(NodeType.BOOL_VAR)
        if h is None or h.shape[0] == 0:
            return torch.zeros(0, device=_dev(emb))
        return self.net(h).squeeze(-1)


class IntegerBranchHead(nn.Module):
    """整数变量 B&B：split 分数 + split 方向 (向上取整 logit)。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.score_net = _mlp(hidden, hidden, 1)
        self.dir_net = _mlp(hidden, hidden, 1)

    def forward(self, emb: dict[NodeType, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h = emb.get(NodeType.NUMERIC_VAR)
        if h is None or h.shape[0] == 0:
            z = torch.zeros(0, device=_dev(emb))
            return z, z
        return self.score_net(h).squeeze(-1), self.dir_net(h).squeeze(-1)


class AuxiliaryHeads(nn.Module):
    """多任务辅助头 (plan 5.3)。

    在 bool_var embedding 上预测:
    - ``conflict_logit``  : 选该变量分支后短期产生冲突的概率。
    - ``in_core_logit``   : 是否属于 unsat core。
    - ``obj_improve``     : 预期 objective 改善 (回归)。
    - ``subtree_size``    : 预期子树规模 (log, 回归)。
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.conflict = _mlp(hidden, hidden, 1)
        self.in_core = _mlp(hidden, hidden, 1)
        self.obj_improve = _mlp(hidden, hidden, 1)
        self.subtree = _mlp(hidden, hidden, 1)

    def forward(self, emb: dict[NodeType, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = emb.get(NodeType.BOOL_VAR)
        if h is None or h.shape[0] == 0:
            z = torch.zeros(0, device=_dev(emb))
            return {"conflict_logit": z, "in_core_logit": z, "obj_improve": z, "subtree_size": z}
        return {
            "conflict_logit": self.conflict(h).squeeze(-1),
            "in_core_logit": self.in_core(h).squeeze(-1),
            "obj_improve": self.obj_improve(h).squeeze(-1),
            "subtree_size": self.subtree(h).squeeze(-1),
        }


def _dev(emb: dict[NodeType, torch.Tensor]) -> torch.device:
    for v in emb.values():
        return v.device
    return torch.device("cpu")
