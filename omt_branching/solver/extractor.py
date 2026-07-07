"""z3 状态 -> ``SolverSnapshot``（输入半边桥接缝，adapter 层）。

遍历 ``φ`` 的 z3 AST，枚举数值变量、线性原子、目标系数，结合 incumbent ``I`` 的
取值，**如实**填充 ``SolverSnapshot`` 的结构字段；z3 公开 API 无法提供的深层 theory
内部量（per-var VSIDS、reduced_cost、simplex basis、pseudocost）留在契约缺省值，
由 graph builder 的 present-flag 编码处理（详见 ``API.md`` 的能力边界表）。

实现注意：z3 ``ExprRef.__eq__`` 返回表达式而非 bool，故系数表一律以**变量名字符串**
为键，AST 去重用 ``get_id()``。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Hashable, Optional

import z3

from omt_branching.input.solver_state import (
    BooleanVarInfo, NumericVarInfo, ObjectiveInfo, SearchStateInfo,
    SolverSnapshot, TheoryAtomInfo,
)
from omt_branching.interfaces import AtomKind, SearchMode
from omt_branching.solver.interfaces import Sense

_CMP_KINDS = {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ}
_INF = float("inf")


class _NonLinear(Exception):
    """遇到非线性子项（v1 限定线性算术，跳过该原子）。"""


@dataclass
class Handle:
    """图内候选 id 到 z3 句柄的回指，供策略构造 split 公式。"""

    kind: str                       # "atom" | "numeric"
    z3_obj: object
    var_id: Hashable
    current_value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    is_integer: bool = True          # 数值变量是否为整数（决定 split 是否可留间隙）


@dataclass
class Extraction:
    """一次抽取结果：快照 + 候选句柄映射。"""

    snapshot: SolverSnapshot
    atom_handles: dict[Hashable, Handle] = field(default_factory=dict)
    numeric_handles: dict[Hashable, Handle] = field(default_factory=dict)


def _merge(dst: dict, src: dict, sign: float = 1.0) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + sign * v


def _map_kind(op: int) -> AtomKind:
    if op in (z3.Z3_OP_LE, z3.Z3_OP_LT):
        return AtomKind.LE
    if op in (z3.Z3_OP_GE, z3.Z3_OP_GT):
        return AtomKind.GE
    if op == z3.Z3_OP_EQ:
        return AtomKind.EQ
    return AtomKind.OTHER


class Z3SnapshotExtractor:
    """把 GOMT 状态（z3 上下文）抽取成 ``SolverSnapshot`` 与候选句柄。"""

    def __init__(self, problem):
        self.problem = problem
        self.objective = problem.objective
        self.sense = problem.sense

    def extract(self, state, backend) -> Extraction:
        phi = state.hard
        incumbent = state.incumbent

        # 1) 收集 φ 中的比较原子与布尔常量。
        atoms_raw: list = []
        bool_consts: list = []
        self._walk(phi, atoms_raw, bool_consts, set())

        # 2) 解析原子为线性形式，登记数值变量。
        var_exprs: dict[str, object] = {}
        atom_infos: list[TheoryAtomInfo] = []
        atom_handles: dict[Hashable, Handle] = {}
        for i, atom in enumerate(atoms_raw):
            info = self._atom_info(atom, i, var_exprs)
            if info is None:
                continue
            atom_infos.append(info)
            atom_handles[info.bool_var_id] = Handle("atom", atom, info.bool_var_id)

        # 目标系数（同时登记目标涉及的变量）。
        obj_coeffs = self._safe_linear(self.objective, var_exprs)

        # 3) 由单变量原子推断盒约束上下界。
        lowers: dict[str, float] = {}
        uppers: dict[str, float] = {}
        for info in atom_infos:
            if len(info.var_coeffs) == 1:
                (vname, coeff), = info.var_coeffs.items()
                if coeff == 1.0 and info.kind == AtomKind.GE:
                    lowers[vname] = max(lowers.get(vname, -_INF), info.rhs)
                elif coeff == 1.0 and info.kind == AtomKind.LE:
                    uppers[vname] = min(uppers.get(vname, _INF), info.rhs)

        # 4) LP 松弛值（把整数变量松弛为实数解 LP）——预测 strong-branching 的关键特征
        #    （learn-to-branch 标准做法；z3 不暴露 LP，故此处自行求解）。仅当存在整数变量时
        #    计算（LRA 全实数无意义且会增开销），失败/无整数则为空。
        has_int = any(bool(z3.is_int(e)) for e in var_exprs.values())
        lp_vals = self._lp_relaxation(atom_infos, obj_coeffs) if has_int else {}

        # 5) 数值变量信息 + 句柄。
        numeric_infos: list[NumericVarInfo] = []
        numeric_handles: dict[Hashable, Handle] = {}
        for name, expr in var_exprs.items():
            val = float(backend.value(incumbent, expr)) if incumbent is not None else None
            lo = lowers.get(name)
            up = uppers.get(name)
            is_int = bool(z3.is_int(expr))
            lp_v = lp_vals.get(name)
            is_frac = lp_v is not None and abs(lp_v - round(lp_v)) > 1e-6
            numeric_infos.append(NumericVarInfo(
                num_var_id=name, is_integer=is_int,
                lp_value=lp_v if lp_v is not None else val,   # 有 LP 松弛用之，否则回退 incumbent 取值
                lower_bound=lo, upper_bound=up, is_fractional=is_frac,
                objective_coeff=obj_coeffs.get(name, 0.0),
            ))
            numeric_handles[name] = Handle("numeric", expr, name,
                                           current_value=val, lower=lo, upper=up,
                                           is_integer=is_int)

        # 5) 布尔变量信息（原子抽象布尔 + 纯布尔常量）。
        bool_infos: list[BooleanVarInfo] = [
            BooleanVarInfo(var_id=info.bool_var_id, is_candidate=True, occurrence_count=1)
            for info in atom_infos
        ]
        seen_bool = set()
        for bc in bool_consts:
            bname = bc.decl().name()
            if bname in seen_bool:
                continue
            seen_bool.add(bname)
            bool_infos.append(BooleanVarInfo(var_id=bname, is_candidate=True))

        # 6) 目标与搜索状态。
        obj_val = (float(backend.value(incumbent, self.objective))
                   if incumbent is not None else None)
        objective = ObjectiveInfo(
            sense_is_min=(self.sense is Sense.MIN), incumbent=obj_val,
            var_coeffs=dict(obj_coeffs),
        )
        search_state = SearchStateInfo(
            depth=int(state.step), decision_level=len(state.tau),
            conflict_count=int(state.stats.get("closes", 0)),
            search_mode=SearchMode.LINEAR,
        )

        snapshot = SolverSnapshot(
            bool_vars=bool_infos, clauses=[], theory_atoms=atom_infos,
            numeric_vars=numeric_infos, objective=objective, search_state=search_state,
            candidate_bool_ids=[info.bool_var_id for info in atom_infos],
            candidate_numeric_ids=list(var_exprs.keys()),
            snapshot_id=f"gomt-step-{state.step}",
        )
        return Extraction(snapshot, atom_handles, numeric_handles)

    def _lp_relaxation(self, atom_infos, obj_coeffs) -> dict:
        """整数变量松弛为实数解 LP，返回 ``{var_name: lp_value(float)}``；不可行/异常返回 ``{}``。

        用抽取得到的线性原子（``var_coeffs``/``rhs``/``kind``）与目标系数重建纯实数 LP。这给 GNN
        提供预测 strong-branching 分离度的 LP 特征（``objective_coeff`` 单独与之几乎无关）。z3 不
        暴露内部 LP，故此处自行以实数 ``Optimize`` 求解一次（相对整数 check-sat 开销可忽略）。
        """
        names: set = set()
        for a in atom_infos:
            names.update(a.var_coeffs.keys())
        names.update(obj_coeffs.keys())
        if not names:
            return {}
        R = {n: z3.Real(f"_lp_{n}") for n in names}
        o = z3.Optimize()
        for a in atom_infos:
            if not a.var_coeffs:
                continue
            lhs = z3.Sum([z3.RealVal(str(c)) * R[v] for v, c in a.var_coeffs.items() if v in R])
            b = z3.RealVal(str(a.rhs))
            if a.kind == AtomKind.LE:
                o.add(lhs <= b)
            elif a.kind == AtomKind.GE:
                o.add(lhs >= b)
            elif a.kind == AtomKind.EQ:
                o.add(lhs == b)
        if obj_coeffs:
            obj = z3.Sum([z3.RealVal(str(c)) * R[v] for v, c in obj_coeffs.items() if v in R])
            if self.sense is Sense.MIN:
                o.minimize(obj)
            else:
                o.maximize(obj)
        try:
            if o.check() != z3.sat:
                return {}
            m = o.model()
            out: dict = {}
            for n, r in R.items():
                try:
                    out[n] = float(m.eval(r, model_completion=True).as_fraction())
                except Exception:
                    out[n] = None
            return out
        except Exception:  # pragma: no cover - LP 异常不应中断主抽取
            return {}

    # ------------------------------------------------------------------ #
    def _walk(self, e, atoms: list, bools: list, visited: set) -> None:
        """沿布尔结构递归，收集比较原子与布尔常量（用 get_id 去重）。"""
        eid = e.get_id()
        if eid in visited:
            return
        visited.add(eid)
        if not z3.is_bool(e):
            return
        op = e.decl().kind()
        if op in _CMP_KINDS and e.num_args() >= 1 and z3.is_arith(e.arg(0)):
            atoms.append(e)
            return
        if z3.is_const(e) and op == z3.Z3_OP_UNINTERPRETED:
            bools.append(e)
            return
        for ch in e.children():
            self._walk(ch, atoms, bools, visited)

    def _atom_info(self, atom, index: int, var_exprs: dict) -> Optional[TheoryAtomInfo]:
        try:
            lhs_c, lhs_k = self._linear(atom.arg(0), var_exprs)
            rhs_c, rhs_k = self._linear(atom.arg(1), var_exprs)
        except _NonLinear:
            warnings.warn(f"跳过非线性原子: {atom}")
            return None
        coeffs: dict[str, float] = {}
        _merge(coeffs, lhs_c)
        _merge(coeffs, rhs_c, -1.0)
        coeffs = {v: c for v, c in coeffs.items() if c != 0.0}
        rhs_val = -(lhs_k - rhs_k)
        atom_id = f"atom{index}"
        return TheoryAtomInfo(
            atom_id=atom_id, bool_var_id=f"b_{atom_id}", kind=_map_kind(atom.decl().kind()),
            var_coeffs=coeffs, rhs=float(rhs_val),
        )

    def _safe_linear(self, e, var_exprs: dict) -> dict[str, float]:
        try:
            coeffs, _ = self._linear(e, var_exprs)
        except _NonLinear:
            warnings.warn(f"跳过非线性目标项: {e}")
            return {}
        return {v: c for v, c in coeffs.items() if c != 0.0}

    def _linear(self, e, var_exprs: dict) -> tuple[dict[str, float], float]:
        """把线性算术表达式分解为 (变量名->系数, 常数)。非线性抛 ``_NonLinear``。"""
        if z3.is_int_value(e):
            return {}, float(e.as_long())
        if z3.is_rational_value(e):
            # 用整数真除而非 float(num)/float(den)：当 z3 模型给出巨大分子/分母
            # （LRA 可行模型/目标切分常见）时，前者不会溢出（商可表示即安全）。
            return {}, e.numerator_as_long() / e.denominator_as_long()
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            name = e.decl().name()
            var_exprs[name] = e
            return {name: 1.0}, 0.0
        op = e.decl().kind()
        children = e.children()
        if op == z3.Z3_OP_ADD:
            coeffs: dict[str, float] = {}
            const = 0.0
            for ch in children:
                cc, ck = self._linear(ch, var_exprs)
                _merge(coeffs, cc)
                const += ck
            return coeffs, const
        if op == z3.Z3_OP_SUB:
            coeffs, const = self._linear(children[0], var_exprs)
            coeffs = dict(coeffs)
            for ch in children[1:]:
                cc, ck = self._linear(ch, var_exprs)
                _merge(coeffs, cc, -1.0)
                const -= ck
            return coeffs, const
        if op == z3.Z3_OP_UMINUS:
            cc, ck = self._linear(children[0], var_exprs)
            return {v: -a for v, a in cc.items()}, -ck
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
                return {}, scale
            if len(rest) == 1:
                cc, ck = self._linear(rest[0], var_exprs)
                return {v: a * scale for v, a in cc.items()}, ck * scale
            raise _NonLinear(str(e))
        raise _NonLinear(str(e))


__all__ = ["Z3SnapshotExtractor", "Extraction", "Handle"]
