"""【输入接口】求解器需要提供的信息类型。

这是求解器 (Z3/νZ / OptiMathSAT / 自研) 与本框架之间的**输入契约**。求解器在
decision / branch 点按下面的 dataclass 填一个 :class:`SolverSnapshot`，框架据此建图。

设计原则:

- 所有字段都是普通 Python 标量 / 列表，不引入对求解器内部对象的依赖。
- ``var_id`` / ``atom_id`` / ``num_var_id`` / ``clause_id`` 可以是任意可哈希值
  （int、str、tuple 均可），框架只用它做映射，最终原样回传。
- 可选字段允许为 ``None``：求解器拿不到的特征（如未求 LP relaxation 时的
  ``lp_value``）填 ``None`` 即可，建图时会用缺失掩码 + 默认值处理。
- 字段命名与取值含义对应 ``plan.md`` 第 4.3 节的特征清单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional

from omt_branching.interfaces import AtomKind, ClauseKind, SearchMode


@dataclass
class BooleanVarInfo:
    """布尔 / 文字节点（CNF 变量、理论原子抽象变量、soft indicator）。"""

    var_id: Hashable
    # --- 赋值与候选状态 ---
    assignment: Optional[bool] = None       # None=未赋值, True/False=已赋值
    decision_level: Optional[int] = None     # 被赋值时的 decision level
    is_candidate: bool = True                # 当前是否可作为 decision 候选
    is_eliminated: bool = False              # 是否已被预处理/化简消去
    # --- 求解器 heuristic 分数 ---
    vsids_activity: float = 0.0
    lrb_score: float = 0.0
    chb_score: float = 0.0
    phase_saved: Optional[bool] = None       # phase saving 记录的极性
    # --- 结构 / 出现统计 ---
    occurrence_count: int = 0
    pos_count: int = 0                       # 正极性出现次数
    neg_count: int = 0                       # 负极性出现次数
    is_soft: bool = False                    # 是否来自 soft constraint
    in_recent_learned: bool = False          # 是否出现在最近 learned clause


@dataclass
class ClauseInfo:
    """子句节点。``literals`` 给出 (var_id, polarity) 列表用于连边。"""

    clause_id: Hashable
    literals: list[tuple[Hashable, bool]] = field(default_factory=list)  # (var_id, is_positive)
    kind: ClauseKind = ClauseKind.ORIGINAL
    lbd: Optional[int] = None                # learned clause 的 LBD/glue
    activity: float = 0.0
    is_satisfied: Optional[bool] = None      # 当前是否已满足


@dataclass
class TheoryAtomInfo:
    """理论原子节点，如 a^T x <= b。"""

    atom_id: Hashable
    bool_var_id: Hashable                    # 抽象它的布尔变量 (atom_abstracted_by 边)
    kind: AtomKind = AtomKind.OTHER
    # numeric_var_id -> coefficient，用于 variable_in_atom 边
    var_coeffs: dict[Hashable, float] = field(default_factory=dict)
    rhs: float = 0.0
    # --- 理论求解器反馈 ---
    slack: Optional[float] = None
    violation: Optional[float] = None
    lp_value: Optional[float] = None         # 当前 LP relaxation 下原子 LHS 取值
    reduced_cost: Optional[float] = None
    is_basic: Optional[bool] = None          # simplex basis status
    tightens_objective: Optional[bool] = None  # 取真后是否收紧 obj < incumbent


@dataclass
class NumericVarInfo:
    """数值变量节点（LRA/LIA/MILP 实数/整数变量）。"""

    num_var_id: Hashable
    is_integer: bool = False
    lp_value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    is_fractional: bool = False              # LP 取分数值（B&B 候选）
    objective_coeff: float = 0.0             # 在目标函数中的系数
    reduced_cost: Optional[float] = None
    is_basic: Optional[bool] = None
    pseudocost_up: float = 0.0
    pseudocost_down: float = 0.0


@dataclass
class ObjectiveInfo:
    """目标节点（目标函数 / incumbent / bound / lex 优先级）。"""

    objective_id: Hashable = "obj"
    sense_is_min: bool = True                # True=最小化
    incumbent: Optional[float] = None        # 当前最优可行解目标值
    best_bound: Optional[float] = None       # 当前最优界
    gap: Optional[float] = None              # relative gap
    lex_priority: int = 0                    # lexicographic 优先级 (0=最高)
    # numeric_var_id -> objective coefficient (variable_in_objective 边)
    var_coeffs: dict[Hashable, float] = field(default_factory=dict)
    # bool_var_id -> soft weight (soft_weight 边)
    soft_weights: dict[Hashable, float] = field(default_factory=dict)
    # numeric_var_id -> (lower, upper) 当前由 incumbent/propagation 产生的界
    related_bounds: dict[Hashable, tuple[Optional[float], Optional[float]]] = field(
        default_factory=dict
    )


@dataclass
class SearchStateInfo:
    """全局搜索状态节点 (plan 4.3 OMT 全局特征)。"""

    depth: int = 0
    decision_level: int = 0
    trail_length: int = 0
    restart_count: int = 0
    conflict_count: int = 0
    conflict_rate: float = 0.0               # 近期冲突 / 时间
    learned_clause_count: int = 0
    search_mode: SearchMode = SearchMode.LINEAR
    is_unbounded_check: bool = False
    last_theory_conflict_size: int = 0
    last_unsat_core_size: int = 0
    last_bound_improvement: float = 0.0
    time_budget_left: Optional[float] = None  # 剩余时间预算（秒），None=未知


@dataclass
class SolverSnapshot:
    """一次 decision/branch 点的完整快照，是输入部分的唯一入口对象。

    求解器只需构造该对象并交给 :class:`~omt_branching.input.graph_builder.GraphBuilder`。
    ``candidate_bool_ids`` / ``candidate_numeric_ids`` 显式给出本次可选的分支对象，
    用于模型 ranking head 的候选 mask；为空时默认所有 ``is_candidate`` 节点参与。
    """

    bool_vars: list[BooleanVarInfo] = field(default_factory=list)
    clauses: list[ClauseInfo] = field(default_factory=list)
    theory_atoms: list[TheoryAtomInfo] = field(default_factory=list)
    numeric_vars: list[NumericVarInfo] = field(default_factory=list)
    objective: ObjectiveInfo = field(default_factory=ObjectiveInfo)
    search_state: SearchStateInfo = field(default_factory=SearchStateInfo)

    # 本次决策的候选集合（求解器原始 id）。空 -> 由建图器自动推断。
    candidate_bool_ids: list[Hashable] = field(default_factory=list)
    candidate_numeric_ids: list[Hashable] = field(default_factory=list)

    snapshot_id: Optional[Hashable] = None   # 便于数据采集时回溯

    def candidate_bool_set(self) -> set[Hashable]:
        if self.candidate_bool_ids:
            return set(self.candidate_bool_ids)
        return {b.var_id for b in self.bool_vars if b.is_candidate and not b.is_eliminated}

    def candidate_numeric_set(self) -> set[Hashable]:
        if self.candidate_numeric_ids:
            return set(self.candidate_numeric_ids)
        return {n.num_var_id for n in self.numeric_vars if n.is_fractional}
