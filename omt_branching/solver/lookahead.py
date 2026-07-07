"""SAT look-ahead 教师：假设某原子，用 z3 consequences 计"强制了多少其他原子"（传播强度），
作 imitation 监督标签。传播强度是**子句共现图的函数**——正是 GNN 已见的特征，故可学
（对比 LIA 分离度需缺失的 LP 特征）。
"""
from __future__ import annotations

from dataclasses import dataclass

import z3

from omt_branching.solver.propagator_snapshot import atom_key, collect_atoms


@dataclass(frozen=True)
class LookaheadConfig:
    max_atoms: int = 32
    eps: float = 1e-9


def _strip_not(e):
    while z3.is_not(e):
        e = e.arg(0)
    return e


def _implied_keys(imps) -> set:
    """从 consequences 的 implication 列表提取被蕴含原子的键集合（剥离 Not / 双重否定）。"""
    out = set()
    for imp in imps:
        cons = imp.arg(1) if (z3.is_implies(imp) and imp.num_args() == 2) else imp
        out.add(atom_key(_strip_not(cons)))
    return out


def _count_marginal(imps, self_key: str, base_keys: set) -> int:
    """假设某原子后**边际**蕴含的其他原子数：扣除 hard 单独就蕴含的（``base_keys``）与自身。"""
    return len(_implied_keys(imps) - base_keys - {self_key})


def lookahead_scores(assertions, atoms=None, config: LookaheadConfig = LookaheadConfig()):
    atom_exprs = list(atoms) if atoms is not None else collect_atoms(list(assertions))
    atom_exprs = atom_exprs[: config.max_atoms]
    s = z3.Solver()
    s.add(*assertions)

    # hard 单独(无假设)就蕴含的原子——与任何假设无关，须从每个假设的传播里扣除，否则每个
    # 原子都"蕴含"这批 box/entailed 原子，计数恒等 -> 标签 uniform 无法学。
    try:
        _, base_imp = s.consequences([], atom_exprs)
        base_keys = _implied_keys(base_imp)
    except z3.Z3Exception:
        base_keys = set()

    scores: dict = {}
    phases: dict = {}
    for a in atom_exprs:
        k = atom_key(a)
        try:
            res_t, imp_t = s.consequences([a], atom_exprs)
            res_f, imp_f = s.consequences([z3.Not(a)], atom_exprs)
        except z3.Z3Exception:
            continue
        # 根状态下某侧不可行 = 该原子被 hard 蕴含/矛盾 -> **不是决策点**（z3 会传播而非分支），
        # 跳过。否则大量 box 原子(x>=0/x<=ub 的取反必 unsat)会以相同大分 swamp 真实 look-ahead
        # 信号，使标签近乎 uniform、无法学习。failed-literal 是搜索期(部分赋值下)的概念，非根标签。
        if res_t == z3.unsat or res_f == z3.unsat:
            continue
        pt = _count_marginal(imp_t, k, base_keys)
        pf = _count_marginal(imp_f, k, base_keys)
        scores[k] = (pt + 1.0) * (pf + 1.0)   # march 风格 product：两侧都边际传播多者优
        phases[k] = pt >= pf                  # 先探传播更多的一侧
    return scores, phases


__all__ = ["LookaheadConfig", "lookahead_scores"]
