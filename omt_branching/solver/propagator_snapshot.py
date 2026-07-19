"""从 z3 布尔公式构造 SolverSnapshot（供 UserPropagator 学习分支）。

抽取原子（比较原子 + 布尔常量）与**子句共现图**，配合 propagator 动态赋值/统计喂给 GNN。

注册策略（与建图分离）：
- :func:`collect_atoms` —— 全部原子（建图 / look-ahead）；
- :func:`collect_clause_atoms` —— 仅 CNF 析取子句（|lits|≥2）中的原子（``prop.add``）；
- :func:`preprocess_assertions` / :func:`prepare_propagator_formula` —— 挂 prop 前轻量预处理。

建图动态投影（``assignment`` / ``fixed``）：
- :func:`build_bool_snapshot` 在非空赋值下做布尔单元传播闭包，再投影子句边、
  剪掉已定候选，供 GNN 在当前部分赋值下重建图（不调用 z3 理论引擎）。

性能：``atom_key`` 按 ``id(expr)`` 缓存；静态 snapshot LRU；``_linear`` 仅单次建图局部缓存。
"""
from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import z3

from omt_branching.input.solver_state import (
    BooleanVarInfo, ClauseInfo, NumericVarInfo, SearchStateInfo, SolverSnapshot,
    TheoryAtomInfo,
)
from omt_branching.solver.extractor import _map_kind, _merge

_CMP = {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ}

# Python 对象 id -> str。键为 id(expr)，值旁路钉住 expr 本身，避免 GC 后 id 复用脏读。
_EXPR_STR_CACHE: dict[int, tuple[object, str]] = {}
_EXPR_STR_CACHE_MAX = 100_000

# 静态 snapshot 骨架 LRU
_STATIC_CACHE: OrderedDict[tuple, "_StaticBoolSnapshot"] = OrderedDict()
_STATIC_CACHE_MAX = 64


def atom_key(e) -> str:
    """z3 原子的稳定字符串键。

    对外语义仍为 ``str(e)``；进程内按 ``id(e)`` 缓存并钉住 ``e``，使同一次求解中
    对同一 ExprRef 的重复调用不再反复字符串化。不用 ``get_id()``：Z3 会在 AST
    回收后复用该 id。
    """
    pid = id(e)
    hit = _EXPR_STR_CACHE.get(pid)
    if hit is not None and hit[0] is e:
        return hit[1]
    s = str(e)
    if len(_EXPR_STR_CACHE) >= _EXPR_STR_CACHE_MAX:
        for drop in list(_EXPR_STR_CACHE.keys())[: _EXPR_STR_CACHE_MAX // 2]:
            _EXPR_STR_CACHE.pop(drop, None)
    _EXPR_STR_CACHE[pid] = (e, s)
    return s


def clear_bool_snapshot_cache() -> None:
    """清空静态结构 / 字符串缓存（测试或长跑内存回收用）。"""
    _EXPR_STR_CACHE.clear()
    _STATIC_CACHE.clear()


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
    """收集断言中全部布尔原子（比较原子 + Bool 常量），供建图 / look-ahead。

    注意：UserPropagator **注册**请用 :func:`collect_clause_atoms`（仅析取子句原子）。
    """
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


def _iter_cnf_clauses(assertion):
    """把断言按顶层 And 拆成子句根；非 And 则自身为一子句。"""
    if z3.is_and(assertion):
        for ch in assertion.children():
            yield from _iter_cnf_clauses(ch)
    else:
        yield assertion


def _clause_atom_exprs(clause) -> list:
    """抽取一个子句中的原子 Expr（Or 展开；含单元子句的单原子）。"""
    atoms: list = []
    seen_ids: set[int] = set()

    def add(e):
        atom, _pos = _lit(e)
        if _is_atom(atom):
            eid = atom.get_id()
            if eid not in seen_ids:
                seen_ids.add(eid)
                atoms.append(atom)
            return
        if z3.is_or(atom):
            for ch in atom.children():
                add(ch)
            return
        for ch in atom.children():
            add(ch)

    if z3.is_or(clause):
        for ch in clause.children():
            add(ch)
    else:
        add(clause)
    return atoms


def collect_clause_atoms(assertions) -> list:
    """仅收集 **CNF 析取子句**（字面量数 ≥ 2）中的原子，供 ``prop.add`` 注册。

    单元约束（盒界 ``x≥0``、单文字子句等）不注册——它们通常由理论传播决定，
    不构成布尔分支决策点。嵌套 ``And`` 会先拆成子句再判断。
    """
    dedup: dict = {}
    uniq: list = []
    for a in assertions:
        for clause in _iter_cnf_clauses(a):
            atoms = _clause_atom_exprs(clause)
            if len(atoms) < 2:
                continue
            for t in atoms:
                k = atom_key(t)
                if k not in dedup:
                    dedup[k] = t
                    uniq.append(t)
    return uniq


def preprocess_assertions(assertions) -> list:
    """在**无** UserPropagator 的独立 Goal 上做轻量预处理，缩小后续原子集。

    使用 ``simplify`` + ``propagate-values``（同 Context）。失败或结果为空时回退原断言。
    调用方应在挂 propagator **之前**调用，并对返回列表求解 / 抽原子。
    """
    assertions = list(assertions)
    if not assertions:
        return []
    ctx = assertions[0].ctx
    try:
        goal = z3.Goal(ctx=ctx)
        goal.add(*assertions)
        tactic = z3.Then(
            z3.Tactic("simplify", ctx=ctx),
            z3.Tactic("propagate-values", ctx=ctx),
        )
        result = tactic(goal)
    except Exception as exc:
        warnings.warn(f"preprocess_assertions 失败，回退原断言: {exc}")
        return assertions

    out: list = []
    seen: set[int] = set()
    for i in range(len(result)):
        sub = result[i]
        for j in range(len(sub)):
            f = sub[j]
            fid = f.get_id()
            if fid in seen:
                continue
            seen.add(fid)
            out.append(f)
    return out if out else assertions


def prepare_propagator_formula(assertions) -> tuple[list, list]:
    """预处理断言并抽取应注册的析取子句原子。

    返回 ``(预处理后断言, register_atoms)``；二者同 Context，供 Solver.add / prop.add。
    """
    pp = preprocess_assertions(assertions)
    return pp, collect_clause_atoms(pp)


def _clause_literals(clause):
    """把一个子句拉平成 (atom_key, is_positive) 列表（Or 展开，其余取自身原子）。"""
    lits = []
    seen_ids: set[int] = set()  # 子句内用 get_id 去重（assertion 存活期间安全）

    def add(e):
        atom, pos = _lit(e)
        if _is_atom(atom):
            eid = atom.get_id()
            if eid not in seen_ids:
                seen_ids.add(eid)
                lits.append((atom_key(atom), pos))
        else:
            for ch in atom.children():
                add(ch)

    if z3.is_or(clause):
        for ch in clause.children():
            add(ch)
    else:
        add(clause)
    return lits

def _linear(e, cache: dict | None = None) -> tuple[dict, float]:
    """把线性算术表达式分解为 ``(变量名->系数, 常数)``；变量键用 ``atom_key``。

    ``cache`` 须为**单次建图局部**字典（键 ``get_id``）。Z3 会在 AST 回收后复用
    ``get_id``，故禁止进程级长期缓存。单次 ``_build_static`` 内相关子式均存活，安全。
    未传入 ``cache`` 时使用临时空字典（便于单测；无跨调用复用）。
    """
    if cache is None:
        cache = {}
    eid = e.get_id()
    cached = cache.get(eid)
    if cached is not None:
        return cached

    if z3.is_int_value(e):
        result = ({}, float(e.as_long()))
        cache[eid] = result
        return result
    if z3.is_rational_value(e):
        # 整数真除避免大分子/分母场景下的浮点溢出。
        result = ({}, e.numerator_as_long() / e.denominator_as_long())
        cache[eid] = result
        return result
    if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED and z3.is_arith(e):
        result = ({atom_key(e): 1.0}, 0.0)
        cache[eid] = result
        return result
    op = e.decl().kind()
    children = e.children()
    if op == z3.Z3_OP_ADD:
        coeffs: dict = {}
        const = 0.0
        for ch in children:
            cc, ck = _linear(ch, cache)
            _merge(coeffs, cc)
            const += ck
        result = (coeffs, const)
        cache[eid] = result
        return result
    if op == z3.Z3_OP_SUB:
        coeffs, const = _linear(children[0], cache)
        coeffs = dict(coeffs)
        for ch in children[1:]:
            cc, ck = _linear(ch, cache)
            _merge(coeffs, cc, -1.0)
            const -= ck
        result = (coeffs, const)
        cache[eid] = result
        return result
    if op == z3.Z3_OP_UMINUS:
        cc, ck = _linear(children[0], cache)
        result = ({v: -a for v, a in cc.items()}, -ck)
        cache[eid] = result
        return result
    if op == z3.Z3_OP_MUL:
        scale = 1.0
        rest = []
        for ch in children:
            if z3.is_int_value(ch):
                scale *= float(ch.as_long())
            elif z3.is_rational_value(ch):
                scale *= float(ch.numerator_as_long()) / float(ch.denominator_as_long())
            else:
                rest.append(ch)
        if not rest:
            result = ({}, scale)
            cache[eid] = result
            return result
        if len(rest) == 1:
            cc, ck = _linear(rest[0], cache)
            result = ({v: a * scale for v, a in cc.items()}, ck * scale)
            cache[eid] = result
            return result
        warnings.warn(f"_linear 遇到无法分解的表达式，退化为空系数: {e}")
        result = ({}, 0.0)  # 两个非常量因子相乘：非线性，本任务原子不会触发
        cache[eid] = result
        return result
    warnings.warn(f"_linear 遇到无法分解的表达式，退化为空系数: {e}")
    result = ({}, 0.0)  # 未识别的算术形状：退化为 0，不阻断建图
    cache[eid] = result
    return result


@dataclass
class _StaticBoolSnapshot:
    """与 assignment / search_state 无关的 snapshot 骨架（可跨 refocus 复用）。"""

    # 钉住顶层 assertion，保证 cache key 中的 get_id 不与后续临时式冲突
    pinned_assertions: tuple = field(repr=False)
    amap: dict
    atom_keys: list[str]
    clauses: list
    theory_atoms: list
    numeric_vars: list
    occ: dict
    pos: dict
    neg: dict


def _assertions_cache_key(assertions) -> tuple:
    if not assertions:
        return (0,)
    ctx = id(assertions[0].ctx) if hasattr(assertions[0], "ctx") else 0
    return (ctx, *(a.get_id() for a in assertions))


def _build_static(assertions) -> _StaticBoolSnapshot:
    """从 assertions 抽取静态结构（atoms / clauses / theory / 度统计）。"""
    atoms = collect_atoms(assertions)
    amap = {atom_key(t): t for t in atoms}
    atom_keys = list(amap.keys())

    occ = {k: 0 for k in amap}
    pos = {k: 0 for k in amap}
    neg = {k: 0 for k in amap}
    clauses = []
    ci = 0
    for a in assertions:
        for clause in _iter_cnf_clauses(a):
            lits = [(k, p) for (k, p) in _clause_literals(clause) if k in amap]
            for k, p in lits:
                occ[k] += 1
                if p:
                    pos[k] += 1
                else:
                    neg[k] += 1
            if lits:
                clauses.append(ClauseInfo(clause_id=f"c{ci}", literals=lits))
                ci += 1
    # 理论原子结构特征（LIA）：numeric_var -> theory_atom -> bool_var。
    # 纯 SAT 布尔常量被守卫跳过，theory_atoms / numeric_vars 保持为空。
    # _linear 局部缓存：仅本函数内有效，避免 get_id 复用脏读。
    linear_cache: dict = {}
    theory_atoms: list[TheoryAtomInfo] = []
    seen_vars: dict = {}
    for k, atom in amap.items():
        op = atom.decl().kind()
        if op not in _CMP or atom.num_args() < 2 or not z3.is_arith(atom.arg(0)):
            continue
        lhs_c, lhs_k = _linear(atom.arg(0), linear_cache)
        rhs_c, rhs_k = _linear(atom.arg(1), linear_cache)
        coeffs: dict = {}
        _merge(coeffs, lhs_c)
        _merge(coeffs, rhs_c, -1.0)
        coeffs = {v: c for v, c in coeffs.items() if c != 0.0}
        rhs_val = rhs_k - lhs_k
        theory_atoms.append(TheoryAtomInfo(
            atom_id=k, bool_var_id=k, kind=_map_kind(op),
            var_coeffs=coeffs, rhs=float(rhs_val),
        ))
        for v in coeffs:
            seen_vars.setdefault(v, None)
    numeric_vars = [NumericVarInfo(num_var_id=v, is_integer=True) for v in seen_vars]

    return _StaticBoolSnapshot(
        pinned_assertions=tuple(assertions),
        amap=amap,
        atom_keys=atom_keys,
        clauses=clauses,
        theory_atoms=theory_atoms,
        numeric_vars=numeric_vars,
        occ=occ,
        pos=pos,
        neg=neg,
    )


def _get_static(assertions) -> _StaticBoolSnapshot:
    key = _assertions_cache_key(assertions)
    hit = _STATIC_CACHE.get(key)
    if hit is not None:
        # 二次校验：get_id 复用时 pinned 引用会对不上
        if (
            len(hit.pinned_assertions) == len(assertions)
            and all(a is b for a, b in zip(hit.pinned_assertions, assertions))
        ):
            _STATIC_CACHE.move_to_end(key)
            return hit
        _STATIC_CACHE.pop(key, None)
    static = _build_static(assertions)
    _STATIC_CACHE[key] = static
    while len(_STATIC_CACHE) > _STATIC_CACHE_MAX:
        _STATIC_CACHE.popitem(last=False)
    return static


def _lit_truth(assignment: dict, key: str, pos: bool) -> Optional[bool]:
    """文字 ``(key, pos)`` 在赋值下：True=已满足，False=已假，None=未定。"""
    if key not in assignment:
        return None
    val = bool(assignment[key])
    return val if pos else (not val)


def _boolean_bcp(clauses: list[ClauseInfo], assignment: dict) -> dict:
    """在子句骨架上对 ``assignment`` 做布尔单元传播闭包（不改动入参、不调 z3）。"""
    asg = dict(assignment)
    changed = True
    while changed:
        changed = False
        for c in clauses:
            undecided: list[tuple[str, bool]] = []
            sat = False
            for k, p in c.literals:
                t = _lit_truth(asg, k, p)
                if t is True:
                    sat = True
                    break
                if t is None:
                    undecided.append((k, p))
            if sat or len(undecided) != 1:
                continue
            k, p = undecided[0]
            forced = bool(p)
            if k not in asg:
                asg[k] = forced
                changed = True
    return asg


def _project_clauses(clauses: list[ClauseInfo], assignment: dict) -> list[ClauseInfo]:
    """按赋值投影子句：已满足者清空边；未满足者去掉假文字；全假标冲突。"""
    out: list[ClauseInfo] = []
    for c in clauses:
        sat = False
        remaining: list[tuple] = []
        for k, p in c.literals:
            t = _lit_truth(assignment, k, p)
            if t is True:
                sat = True
                break
            if t is None:
                remaining.append((k, p))
        if sat:
            out.append(ClauseInfo(
                clause_id=c.clause_id,
                literals=[],
                kind=c.kind,
                lbd=c.lbd,
                activity=c.activity,
                is_satisfied=True,
            ))
        else:
            out.append(ClauseInfo(
                clause_id=c.clause_id,
                literals=remaining,
                kind=c.kind,
                lbd=c.lbd,
                activity=c.activity,
                is_satisfied=False if not remaining else None,
            ))
    return out


def _occ_from_clauses(clauses: list[ClauseInfo], atom_keys: list[str]):
    """从投影后的活跃子句重算出现统计（已满足子句不计）。"""
    occ = {k: 0 for k in atom_keys}
    pos = {k: 0 for k in atom_keys}
    neg = {k: 0 for k in atom_keys}
    for c in clauses:
        if c.is_satisfied:
            continue
        for k, p in c.literals:
            if k not in occ:
                continue
            occ[k] += 1
            if p:
                pos[k] += 1
            else:
                neg[k] += 1
    return occ, pos, neg


def build_bool_snapshot(assertions, assignment: Optional[dict] = None,
                        stats: Optional[dict] = None, snapshot_id: str = "prop"):
    """构造 SolverSnapshot。

    静态部分（原子/子句/理论特征）按 ``assertions`` 缓存。
    ``assignment`` 非空时：布尔 BCP 闭包 → 投影子句边 → 剪掉已定候选，重建动态图视图；
    空赋值时直接复用静态子句对象（零拷贝）。
    """
    assignment = assignment or {}
    stats = stats or {}
    static = _get_static(list(assertions))

    if not assignment:
        bool_vars = [
            BooleanVarInfo(
                var_id=k,
                assignment=None,
                is_candidate=True,
                occurrence_count=static.occ[k],
                pos_count=static.pos[k],
                neg_count=static.neg[k],
            )
            for k in static.atom_keys
        ]
        clauses = static.clauses
        candidates = list(static.atom_keys)
        trail_len = 0
    else:
        asg = _boolean_bcp(static.clauses, assignment)
        clauses = _project_clauses(static.clauses, asg)
        occ, pos, neg = _occ_from_clauses(clauses, static.atom_keys)
        bool_vars = []
        candidates = []
        for k in static.atom_keys:
            val = asg.get(k)
            fixed = val is not None
            bool_vars.append(BooleanVarInfo(
                var_id=k,
                assignment=val,
                is_candidate=not fixed,
                is_eliminated=False,
                occurrence_count=occ[k],
                pos_count=pos[k],
                neg_count=neg[k],
            ))
            if not fixed:
                candidates.append(k)
        trail_len = len(asg)

    search_state = SearchStateInfo(
        decision_level=int(stats.get("decisions", 0)),
        conflict_count=int(stats.get("conflicts", 0)),
        trail_length=trail_len,
    )
    snap = SolverSnapshot(
        bool_vars=bool_vars,
        clauses=clauses,
        theory_atoms=static.theory_atoms,
        numeric_vars=static.numeric_vars,
        search_state=search_state,
        candidate_bool_ids=candidates,
        candidate_numeric_ids=[],
        snapshot_id=snapshot_id,
    )
    return snap, static.amap


__all__ = [
    "atom_key",
    "collect_atoms",
    "collect_clause_atoms",
    "preprocess_assertions",
    "prepare_propagator_formula",
    "build_bool_snapshot",
    "clear_bool_snapshot_cache",
]
