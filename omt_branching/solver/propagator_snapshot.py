"""从 z3 布尔公式构造 SolverSnapshot（供 UserPropagator 学习分支）。

抽取原子（比较原子 + 布尔常量）与**子句共现图**（每个顶层 assertion 的原子构成一个 clause），
配合 propagator 提供的动态赋值/统计，喂给现有 GNN 策略。子句图是学习"好决策"的关键结构
（见 spec §5：无子句图极可能学不动）。
"""
from __future__ import annotations

from typing import Optional

import z3

from omt_branching.input.solver_state import (
    BooleanVarInfo, ClauseInfo, SearchStateInfo, SolverSnapshot,
)

_CMP = {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ}


def atom_key(e) -> str:
    """z3 原子的稳定字符串键（同一进程内对同一原子稳定）。"""
    return str(e)


def _is_atom(e) -> bool:
    if not z3.is_bool(e):
        return False
    op = e.decl().kind()
    if op in _CMP and e.num_args() >= 1 and z3.is_arith(e.arg(0)):
        return True
    return z3.is_const(e) and op == z3.Z3_OP_UNINTERPRETED


def _lit(e):
    """返回 (atom_expr, is_positive)；``Not(a)`` -> (a, False)。"""
    if z3.is_not(e):
        return e.arg(0), False
    return e, True


def _walk_atoms(e, out, seen):
    eid = e.get_id()
    if eid in seen:
        return
    seen.add(eid)
    if _is_atom(e):
        out.append(e)
        return
    if not z3.is_bool(e):
        return
    for ch in e.children():
        _walk_atoms(ch, out, seen)


def collect_atoms(assertions) -> list:
    out: list = []
    seen: set = set()
    dedup: dict = {}
    for a in assertions:
        _walk_atoms(a, out, seen)
    # 按 atom_key 去重、保序
    uniq = []
    for t in out:
        k = atom_key(t)
        if k not in dedup:
            dedup[k] = t
            uniq.append(t)
    return uniq


def _clause_literals(assertion):
    """把一个顶层 assertion 拉平成 (atom_key, is_positive) 列表（Or 展开，其余取自身原子）。"""
    lits = []
    seen = set()

    def add(e):
        atom, pos = _lit(e)
        if _is_atom(atom):
            k = atom_key(atom)
            if k not in seen:
                seen.add(k)
                lits.append((k, pos))
        else:
            for ch in atom.children():
                add(ch)

    if z3.is_or(assertion):
        for ch in assertion.children():
            add(ch)
    else:
        add(assertion)
    return lits


def build_bool_snapshot(assertions, assignment: Optional[dict] = None,
                        stats: Optional[dict] = None, snapshot_id: str = "prop"):
    assignment = assignment or {}
    stats = stats or {}
    atoms = collect_atoms(assertions)
    amap = {atom_key(t): t for t in atoms}

    bool_vars = [
        BooleanVarInfo(var_id=k, assignment=assignment.get(k), is_candidate=True)
        for k in amap
    ]
    clauses = []
    for i, a in enumerate(assertions):
        lits = [(k, p) for (k, p) in _clause_literals(a) if k in amap]
        if lits:
            clauses.append(ClauseInfo(clause_id=f"c{i}", literals=lits))

    search_state = SearchStateInfo(
        decision_level=int(stats.get("decisions", 0)),
        conflict_count=int(stats.get("conflicts", 0)),
        trail_length=len(assignment),
    )
    snap = SolverSnapshot(
        bool_vars=bool_vars, clauses=clauses, theory_atoms=[], numeric_vars=[],
        search_state=search_state,
        candidate_bool_ids=list(amap.keys()), candidate_numeric_ids=[],
        snapshot_id=snapshot_id,
    )
    return snap, amap


__all__ = ["atom_key", "collect_atoms", "build_bool_snapshot"]
