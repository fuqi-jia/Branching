"""OMT 分支选择 GNN 策略框架。

三大部分:
- ``omt_branching.input``  : 求解器信息 -> 异构图。
- ``omt_branching.model``  : 训练 / 微调 / 推理。
- ``omt_branching.output`` : 模型输出 -> 求解器可用的分支建议。

顶层只暴露最常用的入口，细节见各子模块。
"""

from omt_branching.interfaces import (
    NodeType,
    EdgeType,
    AtomKind,
    ClauseKind,
    SearchMode,
)
from omt_branching.service import BranchingPolicyService, ServiceConfig

__all__ = [
    "NodeType",
    "EdgeType",
    "AtomKind",
    "ClauseKind",
    "SearchMode",
    "BranchingPolicyService",
    "ServiceConfig",
]

__version__ = "0.1.0"
