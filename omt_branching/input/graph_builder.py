"""把 :class:`SolverSnapshot` 编码成 :class:`HeteroGraph`。

特征布局对每种节点类型固定（见各 ``_encode_*`` 函数），缺失的可选值用
``(default, present_flag)`` 编码，使模型能区分“值为 0”和“没有这个值”。
所有连续标量在写入前做温和的有界变换（log1p / 截断），降低数值尺度差异。

:class:`FeatureSpec` 暴露各节点/边的特征维度，供模型构建输入投影层。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable

import torch

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.interfaces import (
    AtomKind,
    ClauseKind,
    EDGE_SCHEMA,
    EdgeType,
    NodeType,
    SearchMode,
)
from omt_branching.input.solver_state import (
    BooleanVarInfo,
    ClauseInfo,
    NumericVarInfo,
    ObjectiveInfo,
    SearchStateInfo,
    SolverSnapshot,
    TheoryAtomInfo,
)


# --------------------------------------------------------------------------- #
# 数值编码小工具
# --------------------------------------------------------------------------- #
def _signed_log(x: float) -> float:
    """对带符号大数值做压缩，保留符号。"""
    return math.copysign(math.log1p(abs(x)), x)


def _opt(x: float | None, default: float = 0.0) -> list[float]:
    """可选标量 -> [value, present_flag]。"""
    if x is None:
        return [default, 0.0]
    return [float(x), 1.0]


def _opt_signed_log(x: float | None) -> list[float]:
    if x is None:
        return [0.0, 0.0]
    return [_signed_log(float(x)), 1.0]


def _tri_bool(x: bool | None) -> list[float]:
    """三态布尔 -> one-hot [none, true, false]。"""
    if x is None:
        return [1.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0] if x else [0.0, 0.0, 1.0]


def _onehot(idx: int, n: int) -> list[float]:
    v = [0.0] * n
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


# --------------------------------------------------------------------------- #
# FeatureSpec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    """节点/边特征维度。模型据此构建输入投影。"""

    node_dims: dict[NodeType, int]
    edge_dims: dict[EdgeType, int]

    def node_dim(self, nt: NodeType) -> int:
        return self.node_dims.get(nt, 0)

    def edge_dim(self, et: EdgeType) -> int:
        return self.edge_dims.get(et, 0)


# 各节点特征维度（与下方 _encode_* 严格一致）
_NODE_DIMS: dict[NodeType, int] = {
    NodeType.BOOL_VAR: 19,
    NodeType.CLAUSE: 11,
    NodeType.THEORY_ATOM: 22,
    NodeType.NUMERIC_VAR: 17,
    NodeType.OBJECTIVE: 10,
    NodeType.SEARCH_STATE: 16,
}

# 各边特征维度（无显式特征的关系为 0）
_EDGE_DIMS: dict[EdgeType, int] = {
    EdgeType.LITERAL_IN_CLAUSE: 1,
    EdgeType.ATOM_ABSTRACTED_BY: 0,
    EdgeType.VARIABLE_IN_ATOM: 3,
    EdgeType.VARIABLE_IN_OBJECTIVE: 3,
    EdgeType.SOFT_WEIGHT: 2,
    EdgeType.BOUND_RELATES_VARIABLE: 4,
    EdgeType.STATE_TO_BOOL: 0,
    EdgeType.STATE_TO_OBJECTIVE: 0,
}

DEFAULT_FEATURE_SPEC = FeatureSpec(node_dims=dict(_NODE_DIMS), edge_dims=dict(_EDGE_DIMS))


class GraphBuilder:
    """无状态构图器：``build(snapshot) -> HeteroGraph``。"""

    def __init__(self, feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
                 dtype: torch.dtype = torch.float32):
        self.spec = feature_spec
        self.dtype = dtype

    @property
    def feature_spec(self) -> FeatureSpec:
        return self.spec

    # ------------------------------------------------------------------ #
    def build(self, snap: SolverSnapshot) -> HeteroGraph:
        g = HeteroGraph()

        cand_bool = snap.candidate_bool_set()
        cand_num = snap.candidate_numeric_set()

        # ---- 节点：布尔变量 ----
        bool_feats, bool_map = [], {}
        for i, b in enumerate(snap.bool_vars):
            bool_map[b.var_id] = i
            bool_feats.append(self._encode_bool(b, b.var_id in cand_bool))
        self._set_nodes(g, NodeType.BOOL_VAR, bool_feats, bool_map)

        # ---- 节点：子句 ----
        clause_feats, clause_map = [], {}
        for i, c in enumerate(snap.clauses):
            clause_map[c.clause_id] = i
            clause_feats.append(self._encode_clause(c))
        self._set_nodes(g, NodeType.CLAUSE, clause_feats, clause_map)

        # ---- 节点：理论原子 ----
        atom_feats, atom_map = [], {}
        for i, a in enumerate(snap.theory_atoms):
            atom_map[a.atom_id] = i
            atom_feats.append(self._encode_atom(a))
        self._set_nodes(g, NodeType.THEORY_ATOM, atom_feats, atom_map)

        # ---- 节点：数值变量 ----
        num_feats, num_map = [], {}
        for i, n in enumerate(snap.numeric_vars):
            num_map[n.num_var_id] = i
            num_feats.append(self._encode_numeric(n, n.num_var_id in cand_num))
        self._set_nodes(g, NodeType.NUMERIC_VAR, num_feats, num_map)

        # ---- 节点：目标（单例） ----
        obj_map = {snap.objective.objective_id: 0}
        self._set_nodes(g, NodeType.OBJECTIVE,
                        [self._encode_objective(snap.objective)], obj_map)

        # ---- 节点：搜索状态（单例） ----
        state_map = {"__state__": 0}
        self._set_nodes(g, NodeType.SEARCH_STATE,
                        [self._encode_state(snap.search_state)], state_map)

        # ---- 边 ----
        self._build_edges(g, snap, bool_map, clause_map, atom_map, num_map, obj_map, state_map)

        # ---- 元数据：候选索引（供模型 mask / 输出解码） ----
        g.meta["candidate_bool_local"] = [bool_map[v] for v in cand_bool if v in bool_map]
        g.meta["candidate_numeric_local"] = [num_map[v] for v in cand_num if v in num_map]
        g.meta["candidate_bool_ids"] = [v for v in cand_bool if v in bool_map]
        g.meta["candidate_numeric_ids"] = [v for v in cand_num if v in num_map]
        g.meta["snapshot_id"] = snap.snapshot_id

        return g.finalize()

    # ------------------------------------------------------------------ #
    # 节点编码
    # ------------------------------------------------------------------ #
    def _encode_bool(self, b: BooleanVarInfo, is_cand: bool) -> list[float]:
        assign = (
            [1.0, 0.0, 0.0] if b.assignment is None
            else ([0.0, 1.0, 0.0] if b.assignment else [0.0, 0.0, 1.0])
        )
        total = b.pos_count + b.neg_count
        pos_ratio = (b.pos_count / total) if total > 0 else 0.5
        feats = (
            assign
            + _opt(None if b.decision_level is None else math.log1p(b.decision_level))
            + [1.0 if is_cand else 0.0, 1.0 if b.is_eliminated else 0.0]
            + [b.vsids_activity, b.lrb_score, b.chb_score]
            + _tri_bool(b.phase_saved)
            + [math.log1p(b.occurrence_count)]
            + [math.log1p(b.pos_count), math.log1p(b.neg_count)]
            + [pos_ratio]
            + [1.0 if b.is_soft else 0.0]
            + [1.0 if b.in_recent_learned else 0.0]
        )
        return self._check(NodeType.BOOL_VAR, feats)

    def _encode_clause(self, c: ClauseInfo) -> list[float]:
        kind_idx = ClauseKind.all().index(c.kind)
        feats = (
            _onehot(kind_idx, len(ClauseKind.all()))
            + _opt(None if c.lbd is None else float(c.lbd))
            + [c.activity]
            + [math.log1p(len(c.literals))]
            + _tri_bool(c.is_satisfied)
        )
        return self._check(NodeType.CLAUSE, feats)

    def _encode_atom(self, a: TheoryAtomInfo) -> list[float]:
        kind_idx = AtomKind.all().index(a.kind)
        feats = (
            _onehot(kind_idx, len(AtomKind.all()))
            + [_signed_log(a.rhs)]
            + _opt_signed_log(a.slack)
            + _opt_signed_log(a.violation)
            + _opt_signed_log(a.lp_value)
            + _opt(a.reduced_cost)
            + _tri_bool(a.is_basic)
            + _tri_bool(a.tightens_objective)
            + [math.log1p(len(a.var_coeffs))]
        )
        return self._check(NodeType.THEORY_ATOM, feats)

    def _encode_numeric(self, n: NumericVarInfo, is_cand: bool) -> list[float]:
        if n.lp_value is None:
            frac = 0.0
        else:
            frac = abs(n.lp_value - round(n.lp_value))
        feats = (
            [1.0 if n.is_integer else 0.0]
            + _opt_signed_log(n.lp_value)
            + _opt_signed_log(n.lower_bound)
            + _opt_signed_log(n.upper_bound)
            + [1.0 if (n.is_fractional or is_cand) else 0.0]
            + [frac]
            + [_signed_log(n.objective_coeff)]
            + _opt(n.reduced_cost)
            + _tri_bool(n.is_basic)
            + [n.pseudocost_up, n.pseudocost_down]
        )
        return self._check(NodeType.NUMERIC_VAR, feats)

    def _encode_objective(self, o: ObjectiveInfo) -> list[float]:
        feats = (
            [1.0 if o.sense_is_min else 0.0]
            + _opt_signed_log(o.incumbent)
            + _opt_signed_log(o.best_bound)
            + _opt(o.gap)
            + [float(o.lex_priority)]
            + [math.log1p(len(o.var_coeffs)), math.log1p(len(o.soft_weights))]
        )
        return self._check(NodeType.OBJECTIVE, feats)

    def _encode_state(self, s: SearchStateInfo) -> list[float]:
        mode_idx = SearchMode.all().index(s.search_mode)
        feats = (
            [math.log1p(s.depth), math.log1p(s.decision_level), math.log1p(s.trail_length)]
            + [math.log1p(s.restart_count), math.log1p(s.conflict_count)]
            + [s.conflict_rate]
            + [math.log1p(s.learned_clause_count)]
            + _onehot(mode_idx, len(SearchMode.all()))
            + [1.0 if s.is_unbounded_check else 0.0]
            + [math.log1p(s.last_theory_conflict_size), math.log1p(s.last_unsat_core_size)]
            + [_signed_log(s.last_bound_improvement)]
            + _opt(None if s.time_budget_left is None else math.log1p(max(0.0, s.time_budget_left)))
        )
        return self._check(NodeType.SEARCH_STATE, feats)

    # ------------------------------------------------------------------ #
    # 边构建
    # ------------------------------------------------------------------ #
    def _build_edges(self, g, snap, bool_map, clause_map, atom_map, num_map, obj_map, state_map):
        obj_idx = obj_map[snap.objective.objective_id]
        state_idx = state_map["__state__"]

        # literal_in_clause: bool_var -> clause, 边特征 [is_positive]
        src, dst, ef = [], [], []
        for c in snap.clauses:
            ci = clause_map[c.clause_id]
            for vid, pos in c.literals:
                if vid in bool_map:
                    src.append(bool_map[vid]); dst.append(ci); ef.append([1.0 if pos else 0.0])
        self._set_edges(g, EdgeType.LITERAL_IN_CLAUSE, src, dst, ef)

        # atom_abstracted_by: theory_atom -> bool_var
        src, dst = [], []
        for a in snap.theory_atoms:
            if a.bool_var_id in bool_map:
                src.append(atom_map[a.atom_id]); dst.append(bool_map[a.bool_var_id])
        self._set_edges(g, EdgeType.ATOM_ABSTRACTED_BY, src, dst, None)

        # variable_in_atom: numeric_var -> theory_atom, 边特征 [coeff, sign, |coeff|]
        src, dst, ef = [], [], []
        for a in snap.theory_atoms:
            ai = atom_map[a.atom_id]
            for nvid, coeff in a.var_coeffs.items():
                if nvid in num_map:
                    src.append(num_map[nvid]); dst.append(ai)
                    ef.append([_signed_log(coeff), math.copysign(1.0, coeff) if coeff else 0.0, math.log1p(abs(coeff))])
        self._set_edges(g, EdgeType.VARIABLE_IN_ATOM, src, dst, ef)

        # variable_in_objective: numeric_var -> objective
        src, dst, ef = [], [], []
        for nvid, coeff in snap.objective.var_coeffs.items():
            if nvid in num_map:
                src.append(num_map[nvid]); dst.append(obj_idx)
                ef.append([_signed_log(coeff), math.copysign(1.0, coeff) if coeff else 0.0, math.log1p(abs(coeff))])
        self._set_edges(g, EdgeType.VARIABLE_IN_OBJECTIVE, src, dst, ef)

        # soft_weight: bool_var -> objective, 边特征 [weight, log weight]
        src, dst, ef = [], [], []
        for vid, w in snap.objective.soft_weights.items():
            if vid in bool_map:
                src.append(bool_map[vid]); dst.append(obj_idx)
                ef.append([w, math.log1p(abs(w))])
        self._set_edges(g, EdgeType.SOFT_WEIGHT, src, dst, ef)

        # bound_relates_variable: objective -> numeric_var, 边特征 [lower v+p, upper v+p]
        src, dst, ef = [], [], []
        for nvid, (lo, hi) in snap.objective.related_bounds.items():
            if nvid in num_map:
                src.append(obj_idx); dst.append(num_map[nvid])
                ef.append(_opt_signed_log(lo) + _opt_signed_log(hi))
        self._set_edges(g, EdgeType.BOUND_RELATES_VARIABLE, src, dst, ef)

        # state_to_bool / state_to_objective: 全局状态广播
        if bool_map:
            self._set_edges(g, EdgeType.STATE_TO_BOOL,
                            [state_idx] * len(bool_map), list(bool_map.values()), None)
        self._set_edges(g, EdgeType.STATE_TO_OBJECTIVE, [state_idx], [obj_idx], None)

    # ------------------------------------------------------------------ #
    # 张量装配
    # ------------------------------------------------------------------ #
    def _set_nodes(self, g: HeteroGraph, nt: NodeType,
                   feats: list[list[float]], id_map: dict[Hashable, int]) -> None:
        dim = self.spec.node_dim(nt)
        if feats:
            g.node_features[nt] = torch.tensor(feats, dtype=self.dtype)
        else:
            g.node_features[nt] = torch.zeros((0, dim), dtype=self.dtype)
        g.id_maps[nt] = id_map

    def _set_edges(self, g: HeteroGraph, et: EdgeType,
                   src: list[int], dst: list[int], ef: list[list[float]] | None) -> None:
        if src:
            g.edge_index[et] = torch.tensor([src, dst], dtype=torch.long)
            if ef is not None and self.spec.edge_dim(et) > 0:
                g.edge_features[et] = torch.tensor(ef, dtype=self.dtype)
        else:
            g.edge_index[et] = torch.zeros((2, 0), dtype=torch.long)
            if self.spec.edge_dim(et) > 0:
                g.edge_features[et] = torch.zeros((0, self.spec.edge_dim(et)), dtype=self.dtype)

    def _check(self, nt: NodeType, feats: list[float]) -> list[float]:
        expected = self.spec.node_dim(nt)
        if len(feats) != expected:
            raise AssertionError(f"{nt} 特征维度应为 {expected}, 实际 {len(feats)}")
        return feats
