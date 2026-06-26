"""跨模块共享的类型定义：节点类型、边类型、原子/子句/搜索枚举。

这些枚举对应 ``plan.md`` 第 4 节的异构图设计，是输入、模型、输出三部分共同遵守的
词表。图构建器用它们给节点/边命名，模型用它们枚举关系，输出用它们解释候选节点。
"""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """异构图节点类型 (plan 4.1)。"""

    BOOL_VAR = "bool_var"          # CNF/抽象布尔变量、soft indicator
    CLAUSE = "clause"              # 原始/学习/blocking/bound 子句
    THEORY_ATOM = "theory_atom"    # a^T x <= b, x = y + c, PB, BV compare ...
    NUMERIC_VAR = "numeric_var"    # LRA/LIA/MILP 实数/整数变量
    OBJECTIVE = "objective"        # 目标函数 / incumbent / bound
    SEARCH_STATE = "search_state"  # 全局搜索状态（单例节点）

    @classmethod
    def all(cls) -> list["NodeType"]:
        return list(cls)


class EdgeType(str, Enum):
    """异构图边类型 (plan 4.2)。

    采用 (src, relation, dst) 三元组语义；此处用关系名作为枚举值，
    具体 src/dst 在 ``EDGE_SCHEMA`` 中给出。所有边在建图时都会补充反向边，
    便于双向消息传递。
    """

    LITERAL_IN_CLAUSE = "literal_in_clause"        # bool_var -> clause
    ATOM_ABSTRACTED_BY = "atom_abstracted_by"      # theory_atom -> bool_var
    VARIABLE_IN_ATOM = "variable_in_atom"          # numeric_var -> theory_atom
    VARIABLE_IN_OBJECTIVE = "variable_in_objective"  # numeric_var -> objective
    SOFT_WEIGHT = "soft_weight"                    # bool_var -> objective
    BOUND_RELATES_VARIABLE = "bound_relates_variable"  # objective -> numeric_var
    STATE_TO_BOOL = "state_to_bool"                # search_state -> bool_var
    STATE_TO_OBJECTIVE = "state_to_objective"      # search_state -> objective


#: 每个关系的 (源节点类型, 目标节点类型)。建图与模型都依赖该 schema。
EDGE_SCHEMA: dict[EdgeType, tuple[NodeType, NodeType]] = {
    EdgeType.LITERAL_IN_CLAUSE: (NodeType.BOOL_VAR, NodeType.CLAUSE),
    EdgeType.ATOM_ABSTRACTED_BY: (NodeType.THEORY_ATOM, NodeType.BOOL_VAR),
    EdgeType.VARIABLE_IN_ATOM: (NodeType.NUMERIC_VAR, NodeType.THEORY_ATOM),
    EdgeType.VARIABLE_IN_OBJECTIVE: (NodeType.NUMERIC_VAR, NodeType.OBJECTIVE),
    EdgeType.SOFT_WEIGHT: (NodeType.BOOL_VAR, NodeType.OBJECTIVE),
    EdgeType.BOUND_RELATES_VARIABLE: (NodeType.OBJECTIVE, NodeType.NUMERIC_VAR),
    EdgeType.STATE_TO_BOOL: (NodeType.SEARCH_STATE, NodeType.BOOL_VAR),
    EdgeType.STATE_TO_OBJECTIVE: (NodeType.SEARCH_STATE, NodeType.OBJECTIVE),
}


class AtomKind(str, Enum):
    """理论原子类型 (plan 4.3)。"""

    LE = "le"        # a^T x <= b
    GE = "ge"        # a^T x >= b
    EQ = "eq"        # a^T x = b
    PB = "pb"        # pseudo-boolean
    BV_CMP = "bv_cmp"
    OTHER = "other"

    @classmethod
    def all(cls) -> list["AtomKind"]:
        return list(cls)


class ClauseKind(str, Enum):
    """子句来源类型 (plan 4.1)。"""

    ORIGINAL = "original"
    LEARNED = "learned"
    BLOCKING = "blocking"
    BOUND = "bound"

    @classmethod
    def all(cls) -> list["ClauseKind"]:
        return list(cls)


class SearchMode(str, Enum):
    """OMT 搜索模式 (plan 4.3 / OptiMathSAT)。"""

    LINEAR = "linear"
    BINARY = "binary"
    ADAPTIVE = "adaptive"

    @classmethod
    def all(cls) -> list["SearchMode"]:
        return list(cls)
