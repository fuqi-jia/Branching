"""输入部分：把求解器信息转化为异构图。

- ``solver_state`` : 定义求解器需要提供的信息类型（接口契约）。
- ``graph_builder``: 把 ``SolverSnapshot`` 编码成 ``HeteroGraph``。
"""

from omt_branching.input.solver_state import (
    BooleanVarInfo,
    ClauseInfo,
    TheoryAtomInfo,
    NumericVarInfo,
    ObjectiveInfo,
    SearchStateInfo,
    SolverSnapshot,
)
from omt_branching.input.graph_builder import GraphBuilder, FeatureSpec

__all__ = [
    "BooleanVarInfo",
    "ClauseInfo",
    "TheoryAtomInfo",
    "NumericVarInfo",
    "ObjectiveInfo",
    "SearchStateInfo",
    "SolverSnapshot",
    "GraphBuilder",
    "FeatureSpec",
]
