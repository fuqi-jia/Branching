"""OMT = 单 z3.Solver 线性搜索回路（Solve + Better-cut，直到 UNSAT），可挂
LearnedDecidePropagator 接管内部布尔决策。z3.Optimize 不支持 propagator，故必须走此回路。

对比臂：``solve_native``（Python Optimize API）、``solve_binary``（z3 二进制 ``-st``）、
``solve_omt_with_decider``（公平 VSIDS / learned UserPropagator 回路；
``attach_propagator=False`` 为 check-sat-loop 基线）。
"""

from __future__ import annotations

from fractions import Fraction
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Optional

import z3

from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import prepare_propagator_formula


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


def _num(ref):
    if z3.is_int_value(ref):
        return ref.as_long()
    if z3.is_rational_value(ref):
        return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
    return Fraction(str(ref))


def _defer_always(_undecided, _assignment):
    """公平 VSIDS：decide 回调恒返回 None，不调用 next_split，退回 z3 原生 VSIDS。"""
    return None


def solve_omt_with_decider(
    hard,
    objective,
    sense: Sense,
    decider_factory=None,
    max_iters: int = 100000,
    ctx: z3.Context | None = None,
    *,
    ref_rlimit: int | None = None,
    sample: bool = False,
    attach_propagator: bool = True,
) -> dict:
    """OMT 线性搜索；默认在独立 :class:`z3.Context` 内运行，避免跨线程/跨求解共享表达式。

    始终对硬约束做轻量预处理（``prepare_propagator_formula``），并在简化后的断言上
    ``check``。

    - ``attach_propagator=True``（默认）：挂 ``LearnedDecidePropagator``，只注册析取子句
      （字面量数 ≥ 2）中的原子。``decider_factory=None`` 时 decider 恒 defer（公平
      VSIDS 臂）；否则 ``decider_factory(assertions) -> decider``（learned 臂）。
    - ``attach_propagator=False``：不挂 propagator（check-sat-loop 臂）；忽略
      ``decider_factory``。
    - 每次 better-cut 写入 Solver 后，若 decider 实现 ``add_hard``，会把 cut 并入 GNN
      建图断言，并刷新根级 ``consequences`` 强制赋值（跨 cut 简化建图）；不额外
      ``prop.add``：单元 cut 本就不注册。

    若给定 ``ref_rlimit``，当前消耗超出 ``2 * ref_rlimit`` 时提前返回（未达最优时
    reward 侧多为 -1.0）；未给定时不做该剪枝。训练 collect 与验证均应传入参考
    ``ref_rlimit`` 以加速无望实例。
    """
    if ctx is None:
        ctx = z3.Context()
    hard_iso = [h.translate(ctx) for h in hard]
    obj_iso = objective.translate(ctx)

    s = z3.Solver(ctx=ctx)
    solver_rlimit = _stat(s, "rlimit count")
    rlimit = solver_rlimit
    prop = None
    # 两臂均预处理；公平 VSIDS / learned 再挂 prop（仅注册析取子句原子）。
    hard_use, atoms = prepare_propagator_formula(hard_iso)
    if attach_propagator:
        if decider_factory is None:
            decider = _defer_always
        else:
            decider = decider_factory(hard_use)
        prop = LearnedDecidePropagator(s, atoms, decider)
    decider_factory_rlimit = _stat(s, "rlimit count") - rlimit
    rlimit += decider_factory_rlimit

    s.add(*hard_use)
    model_rlimit = [_stat(s, "rlimit count") - rlimit]
    rlimit += model_rlimit[-1]

    if s.check() != z3.sat:
        raise ValueError("硬约束不可满足")
    check_rlimit = [_stat(s, "rlimit count") - rlimit]
    rlimit += check_rlimit[-1]

    m = s.model()
    best_val = m.eval(obj_iso, model_completion=True)
    eval_rlimit = [_stat(s, "rlimit count") - rlimit]
    rlimit += eval_rlimit[-1]

    records = [(_num(best_val), check_rlimit[-1] + eval_rlimit[-1])]

    iters = 0
    for iters in range(1, max_iters + 1):
        if ref_rlimit is not None and rlimit - solver_rlimit > 2 * ref_rlimit:
            break
        cut = obj_iso > best_val if sense is Sense.MAX else obj_iso < best_val
        s.add(cut)
        # 同步进 GNN 建图断言（若 decider 支持）；不注册到 propagator。
        add_hard = getattr(prop.decider, "add_hard", None) if prop is not None else None
        if callable(add_hard):
            add_hard(cut)
        model_rlimit.append(_stat(s, "rlimit count") - rlimit)
        rlimit += model_rlimit[-1]

        if s.check() != z3.sat:
            break
        check_rlimit.append(_stat(s, "rlimit count") - rlimit)
        rlimit += check_rlimit[-1]

        m = s.model()
        best_val = m.eval(obj_iso, model_completion=True)
        eval_rlimit.append(_stat(s, "rlimit count") - rlimit)
        rlimit += eval_rlimit[-1]

        records.append(
            (_num(best_val), model_rlimit[-1] + check_rlimit[-1] + eval_rlimit[-1])
        )

    stats = {
        "value": _num(best_val),
        # "rlimit": _stat(s, "rlimit count"),
        "rlimit": decider_factory_rlimit
        + sum(model_rlimit)
        + sum(check_rlimit)
        + sum(eval_rlimit),
        "conflicts": _stat(s, "conflicts"),
        "decisions": (prop.n_decisions if prop is not None else None),
        "iters": iters,
    }

    weighted_rlimit = stats["rlimit"]
    local, cost = records[0]
    weighted_rlimit += cost
    for i in range(1, len(records)):
        last_local = local
        local, cost = records[i]
        weighted_rlimit += (
            1.0 * cost * (stats["value"] - local) / (stats["value"] - last_local)
        )
    stats["weighted rlimit"] = weighted_rlimit

    # stats["solver rlimit"] = solver_rlimit
    stats["decider factory rlimit"] = decider_factory_rlimit
    stats["model base rlimit"] = model_rlimit[0]
    stats["model cut rlimit"] = sum(model_rlimit) - model_rlimit[0]
    stats["check rlimit"] = sum(check_rlimit)
    stats["eval rlimit"] = sum(eval_rlimit)

    return stats


def solve_native(
    hard,
    obj,
    sense: Sense,
    max_rlimit: int = -1,
):
    ctx = z3.Context()
    o = z3.Optimize(ctx=ctx)
    if max_rlimit > 0:
        o.set("rlimit", max_rlimit)
    hard_iso = [h.translate(ctx) for h in hard]
    obj_iso = obj.translate(ctx)
    o.add(*hard_iso)
    if sense is Sense.MIN:
        o.minimize(obj_iso)
    else:
        o.maximize(obj_iso)
    res = o.check()
    rlimit = _stat(o, "rlimit count")
    if res != z3.sat:
        return {
            "value": None,
            "rlimit": rlimit,
        }
    m = o.model()
    return {
        "value": _num(m.eval(obj_iso, model_completion=True)),
        "rlimit": rlimit,
    }


def _instance_logic(inst: OMTInstance) -> str:
    if inst.theory == "LRA":
        return "QF_LRA"
    if inst.family == "bool":
        return "ALL"
    return "QF_LIA"


def _instance_var_sort(inst: OMTInstance) -> str:
    return "Real" if inst.theory == "LRA" else "Int"


def instance_to_smt2(inst: OMTInstance) -> str:
    """将 ``OMTInstance`` 导出为 z3 二进制可读的 OMT SMT-LIB2（含 ``get-value``）。"""
    obj = inst.objective.sexpr()
    sense_cmd = "maximize" if inst.sense is Sense.MAX else "minimize"
    var_sort = _instance_var_sort(inst)
    lines = [
        f"(set-logic {_instance_logic(inst)})",
        "(set-option :produce-models true)",
    ]
    for v in inst.variables:
        lines.append(f"(declare-fun {v.decl().name()} () {var_sort})")
    for h in inst.hard:
        lines.append(f"(assert {h.sexpr()})")
    lines.append(f"({sense_cmd} {obj})")
    lines.append("(check-sat)")
    lines.append(f"(get-value ({obj}))")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 读接口：SMT2 (+manifest) -> OMTInstance（与 instance_to_smt2 / save_dataset 互逆）
# --------------------------------------------------------------------------- #

_OBJ_RECOVER_VAR = "__omt_obj_recover__"


def _read_smt2_source(source) -> tuple[str, Optional[str]]:
    """把入参统一成 ``(smt2_text, derived_id)``。

    ``source`` 是**已存在的文件路径**则读文件（``derived_id`` 取文件名）；否则当作
    SMT-LIB2 文本（``derived_id`` 为 ``None``）。
    """
    try:
        cand = Path(source)
        if cand.exists() and cand.is_file():
            return cand.read_text(encoding="utf-8"), cand.stem
    except (OSError, ValueError):
        pass
    return str(source), None


def _declared_var_names(text: str) -> list[str]:
    """按声明顺序取所有 ``(declare-fun name () Int|Real)`` 的变量名。"""
    return re.findall(r"^\(declare-fun\s+(\S+)\s*\(\)\s*(?:Int|Real)\)", text, re.M)


def _extract_objective(text: str) -> tuple[Sense, str]:
    """定位唯一的 ``(maximize|minimize <expr>)``，返回 ``(sense, expr_sexpr)``。

    - 无目标 -> ``ValueError``；多目标 / ``assert-soft`` -> ``NotImplementedError``
      （``OMTInstance`` 仅支持单目标）。
    - ``<expr>`` 可跨多行，按括号配平提取。
    """
    hits = [
        (m.group(1), m.start()) for m in re.finditer(r"\((maximize|minimize)\b", text)
    ]
    if not hits:
        raise ValueError("未找到 (maximize|minimize ...)：不是单目标 OMT 的 .smt2")
    if len(hits) > 1 or "assert-soft" in text:
        raise NotImplementedError(
            "检测到多目标 / assert-soft：OMTInstance 仅支持单目标 OMT，暂不支持"
        )
    cmd, start = hits[0]
    depth, end = 0, -1
    for j in range(start, len(text)):
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end == -1:
        raise ValueError("(maximize|minimize ...) 括号不配平")
    inner = text[start + 1 + len(cmd) : end].strip()  # 去掉 "(cmd"
    if inner.endswith(")"):
        inner = inner[:-1].strip()  # 去掉命令收尾 ")"
    sense = Sense.MAX if cmd == "maximize" else Sense.MIN
    return sense, inner


def _parse_objective_expr(decl_lines: list[str], expr_sexpr: str, sort: str):
    """在给定声明下把目标 S 表达式解析成 z3 ``ArithRef``。

    z3 的 SMT-LIB 解析器会丢弃 OMT 的 maximize/minimize，故用一条等式
    ``(= __obj__ <expr>)`` 让解析器重建目标表达式，再取等式的另一侧。
    """
    body = "\n".join(decl_lines)
    smt2 = (
        f"{body}\n(declare-fun {_OBJ_RECOVER_VAR} () {sort})\n"
        f"(assert (= {_OBJ_RECOVER_VAR} {expr_sexpr}))\n"
    )
    eq = z3.parse_smt2_string(smt2)[0]
    lhs, rhs = eq.arg(0), eq.arg(1)
    if z3.is_const(lhs) and lhs.decl().name() == _OBJ_RECOVER_VAR:
        return rhs
    return lhs


def _as_number(e) -> Optional[float]:
    """把数值型 S 表达式（整数/有理/一元负号/除法）转 float，否则 None。"""
    if z3.is_int_value(e):
        return float(e.as_long())
    if z3.is_rational_value(e):
        return float(e.numerator_as_long()) / float(e.denominator_as_long())
    if z3.is_app(e):
        k = e.decl().kind()
        if k == z3.Z3_OP_UMINUS and e.num_args() == 1:
            n = _as_number(e.arg(0))
            return None if n is None else -n
        if k == z3.Z3_OP_DIV and e.num_args() == 2:
            a, b = _as_number(e.arg(0)), _as_number(e.arg(1))
            return a / b if (a is not None and b) else None
    return None


def _is_arith_var(e) -> bool:
    return (
        z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED and z3.is_arith(e)
    )


def _linear_terms(expr, var_names: list[str]) -> dict[str, float]:
    """把线性目标分解为 ``{变量名: 系数}``（常数项丢弃）。

    所有声明变量默认系数 0.0（与生成器约定一致），再据表达式累加；非线性节点尽力而为。
    """
    coeffs: dict[str, float] = {n: 0.0 for n in var_names}

    def rec(e, scale: float) -> None:
        if _as_number(e) is not None:
            return  # 纯常数项，跳过
        k = e.decl().kind() if z3.is_app(e) else None
        if k == z3.Z3_OP_ADD:
            for ch in e.children():
                rec(ch, scale)
            return
        if k == z3.Z3_OP_SUB:
            chs = e.children()
            rec(chs[0], scale)
            for ch in chs[1:]:
                rec(ch, -scale)
            return
        if k == z3.Z3_OP_UMINUS:
            rec(e.arg(0), -scale)
            return
        if k == z3.Z3_OP_MUL:
            factor, var_children = 1.0, []
            for ch in e.children():
                num = _as_number(ch)
                if num is None:
                    var_children.append(ch)
                else:
                    factor *= num
            if len(var_children) == 1 and _is_arith_var(var_children[0]):
                nm = var_children[0].decl().name()
                coeffs[nm] = coeffs.get(nm, 0.0) + scale * factor
            else:
                for ch in var_children:  # 非纯 c*x：尽力递归
                    rec(ch, scale * factor)
            return
        if _is_arith_var(e):
            nm = e.decl().name()
            coeffs[nm] = coeffs.get(nm, 0.0) + scale
            return
        for ch in (e.children() if z3.is_app(e) else []):  # 未知结构：尽力而为
            rec(ch, scale)

    rec(expr, 1.0)
    return coeffs


def smt2_to_instance(
    source,
    *,
    instance_id: str | None = None,
    family: str = "imported",
    description: str = "",
) -> OMTInstance:
    """把单目标 OMT 的 SMT-LIB2（``instance_to_smt2`` 的产物或等价文件）读回 ``OMTInstance``。

    ``source``：已存在的 ``.smt2`` 文件路径，或直接的 SMT-LIB2 文本。硬约束与变量由 z3 解析；
    目标/方向从唯一的 ``(maximize|minimize ...)`` 行恢复（z3 解析器不返回该命令，见
    :func:`_parse_objective_expr`）；``obj_coeffs`` 由线性目标反推。多目标 / assert-soft 报错。
    """
    text, derived_id = _read_smt2_source(source)
    hard = list(z3.parse_smt2_string(text))
    sense, obj_sexpr = _extract_objective(text)
    sort = "Real" if re.search(r"\(\)\s*Real\s*\)", text) else "Int"
    theory = "LRA" if sort == "Real" else "LIA"
    decl_lines = re.findall(r"^\(set-logic[^\n]*\)|^\(declare-fun[^\n]*\)", text, re.M)
    objective = _parse_objective_expr(decl_lines, obj_sexpr, sort)
    var_names = _declared_var_names(text)
    variables = [z3.Int(n) if sort == "Int" else z3.Real(n) for n in var_names]
    obj_coeffs = _linear_terms(objective, var_names)
    return OMTInstance(
        instance_id=instance_id or derived_id or "imported",
        variables=variables,
        hard=hard,
        objective=objective,
        sense=sense,
        obj_coeffs=obj_coeffs,
        theory=theory,
        family=family,
        description=description,
    )


def load_dataset(path, *, split: str | None = None) -> list[OMTInstance]:
    """读回落盘数据集（``.smt2`` 为权威；``manifest.json`` 仅补充标签）。

    ``path``：数据集目录或 ``manifest.json`` 路径。

    **枚举与求解字段以磁盘 ``.smt2`` 为准**（``test/``、``train/`` 等划分目录下的文件）。
    若存在 ``manifest.json``，仅用其 ``instance_id/family/description`` 做 best-effort 覆盖；
    不再信任可能过期的 manifest 文件列表。无划分目录时退化为 ``rglob("*.smt2")``。
    """
    root = Path(path)
    manifest_path = root if root.is_file() else root / "manifest.json"
    if root.is_file():
        root = root.parent

    label_by_key: dict[tuple[str, str], dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for sp, entries in manifest.get("splits", {}).items():
            for entry in entries:
                iid = entry.get("instance_id") or Path(entry.get("smt2", "")).stem
                if iid:
                    label_by_key[(sp, iid)] = entry

    split_dirs = _discover_split_dirs(root)
    if split is not None:
        split_dirs = [split] if (root / split).is_dir() else []

    out: list[OMTInstance] = []
    if split_dirs:
        for sp in split_dirs:
            for smt2_path in sorted((root / sp).glob("*.smt2")):
                labels = label_by_key.get((sp, smt2_path.stem), {})
                out.append(
                    smt2_to_instance(
                        smt2_path,
                        instance_id=labels.get("instance_id") or smt2_path.stem,
                        family=labels.get("family", "imported"),
                        description=labels.get("description", ""),
                    )
                )
        return out

    # 无标准划分目录：扫全部 .smt2（跳过仅作缓存的无关路径）
    return [smt2_to_instance(f) for f in sorted(root.rglob("*.smt2"))]


def _discover_split_dirs(root: Path) -> list[str]:
    """返回含 ``*.smt2`` 的一级子目录名（排除缓存/产物目录）。"""
    skip = {"binary", "ref", "lookahead", "lookahead_objective", "rl_checkpoints"}
    found: list[str] = []
    if not root.is_dir():
        return found
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name not in skip and any(p.glob("*.smt2")):
            found.append(p.name)
    return found


def instance_manifest_entry(inst: OMTInstance, *, smt2_relpath: str) -> dict:
    """由 ``OMTInstance`` 构造一条 manifest 条目（与 ``save_dataset`` 一致）。"""
    return {
        "instance_id": inst.instance_id,
        "theory": inst.theory,
        "family": inst.family,
        "description": inst.description,
        "sense": inst.sense.value,
        "n_vars": len(inst.variables),
        "n_hard": len(inst.hard),
        "obj_coeffs": inst.obj_coeffs,
        "smt2": smt2_relpath.replace("\\", "/"),
    }


def manifest_mismatches(dataset_dir) -> list[str]:
    """检查 manifest 与磁盘 ``.smt2`` 是否一致，返回问题描述列表（空=一致）。"""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    issues: list[str] = []
    if not manifest_path.is_file():
        if _discover_split_dirs(root):
            issues.append("存在 .smt2 划分目录但缺少 manifest.json")
        return issues

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    disk_splits = set(_discover_split_dirs(root))
    listed_splits = {sp for sp, ents in splits.items() if ents}

    for sp in sorted(disk_splits | listed_splits):
        disk_files = (
            {p.name for p in (root / sp).glob("*.smt2")}
            if (root / sp).is_dir()
            else set()
        )
        listed = {Path(e["smt2"]).name for e in splits.get(sp, []) if e.get("smt2")}
        extra = disk_files - listed
        missing = listed - disk_files
        if extra:
            issues.append(f"{sp}/: 磁盘有但 manifest 未列: {sorted(extra)}")
        if missing:
            issues.append(f"{sp}/: manifest 有但磁盘缺失: {sorted(missing)}")
        for e in splits.get(sp, []):
            rel = e.get("smt2")
            if not rel:
                continue
            p = root / rel
            if not p.is_file():
                continue
            inst = smt2_to_instance(p, instance_id=e.get("instance_id"))
            if e.get("sense") and e["sense"] != inst.sense.value:
                issues.append(
                    f"{rel}: sense manifest={e['sense']} smt2={inst.sense.value}"
                )
            if e.get("theory") and e["theory"] != inst.theory:
                issues.append(
                    f"{rel}: theory manifest={e['theory']} smt2={inst.theory}"
                )
            if e.get("n_vars") is not None and int(e["n_vars"]) != len(inst.variables):
                issues.append(
                    f"{rel}: n_vars manifest={e['n_vars']} smt2={len(inst.variables)}"
                )
            if e.get("n_hard") is not None and int(e["n_hard"]) != len(inst.hard):
                issues.append(
                    f"{rel}: n_hard manifest={e['n_hard']} smt2={len(inst.hard)}"
                )
            if e.get("obj_coeffs") is not None:
                mc = {k: float(v) for k, v in e["obj_coeffs"].items()}
                ac = {k: float(v) for k, v in inst.obj_coeffs.items()}
                if mc != ac:
                    issues.append(f"{rel}: obj_coeffs 与 smt2 不一致")
    return issues


def _infer_family_description(instance_id: str) -> tuple[str, str]:
    """无旧标签时，按 id 前缀猜测 family/description。"""
    if instance_id.startswith("bfocus"):
        return "branch_focus", "branch-focus LIA (imported)"
    if instance_id.startswith("hblia"):
        return "bool", "hard bool LIA (imported)"
    if instance_id.startswith("blia"):
        return "bool", "bool LIA (imported)"
    if instance_id.startswith("hlia"):
        return "knapsack", "hard LIA (imported)"
    if instance_id.startswith("lra"):
        return "imported", "LRA (imported)"
    return "imported", ""


def rebuild_manifest(
    dataset_dir,
    *,
    preserve_meta: bool = True,
) -> dict:
    """按磁盘 ``.smt2`` 重建 ``manifest.json``（``.smt2`` 为权威）。

    ``preserve_meta=True`` 时保留旧 manifest 的 ``generator/params/seeds/created_at``，
    以及同 id 的 ``family/description``；``params.test/train`` 按实际文件数更新。
    """
    from datetime import datetime, timezone

    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    old: dict = {}
    manifest_path = root / "manifest.json"
    if preserve_meta and manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))

    old_labels: dict[tuple[str, str], dict] = {}
    for sp, entries in old.get("splits", {}).items():
        for e in entries:
            iid = e.get("instance_id") or Path(e.get("smt2", "")).stem
            if iid:
                old_labels[(sp, iid)] = e

    splits: dict[str, list] = {}
    for sp in _discover_split_dirs(root):
        entries: list[dict] = []
        for smt2_path in sorted((root / sp).glob("*.smt2")):
            labels = old_labels.get((sp, smt2_path.stem), {})
            iid = labels.get("instance_id") or smt2_path.stem
            fam = labels.get("family")
            desc = labels.get("description")
            if not fam or fam == "imported":
                inferred_f, inferred_d = _infer_family_description(iid)
                fam = fam if fam and fam != "imported" else inferred_f
                if not desc:
                    desc = inferred_d
            inst = smt2_to_instance(
                smt2_path,
                instance_id=iid,
                family=fam or "imported",
                description=desc or "",
            )
            rel = f"{sp}/{smt2_path.name}"
            entries.append(instance_manifest_entry(inst, smt2_relpath=rel))
        splits[sp] = entries

    params = dict(old.get("params") or {})
    for sp_name in ("test", "train", "eval"):
        if sp_name in splits:
            params[sp_name] = len(splits[sp_name])

    manifest = {
        "created_at": old.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "generator": old.get("generator", "imported"),
        "params": params,
        "seeds": old.get("seeds", {}),
        "splits": splits,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    return manifest


def save_dataset(
    instances: list[OMTInstance],
    out_dir,
    *,
    split: str,
    prune_orphans: bool = True,
) -> list[dict]:
    """把实例列表落盘为 ``<out_dir>/<split>/<id>.smt2``，返回 manifest 条目。

    ``prune_orphans=True`` 时删除该划分下未出现在本次写入列表中的旧 ``.smt2``，
    避免 manifest 与磁盘残留文件不一致。
    """
    split_dir = Path(out_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    keep: set[str] = set()
    for inst in instances:
        fname = f"{inst.instance_id}.smt2"
        keep.add(fname)
        relpath = f"{split}/{fname}"
        (split_dir / fname).write_text(instance_to_smt2(inst), encoding="utf-8")
        entries.append(instance_manifest_entry(inst, smt2_relpath=relpath))
    if prune_orphans:
        for old in split_dir.glob("*.smt2"):
            if old.name not in keep:
                old.unlink(missing_ok=True)
    return entries


def list_split_entries(dataset_dir, split: str) -> list[dict]:
    """返回某划分的 manifest 风格条目（始终按磁盘 ``.smt2`` 现算）。"""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    labels: dict[str, dict] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for e in manifest.get("splits", {}).get(split, []):
            iid = e.get("instance_id") or Path(e.get("smt2", "")).stem
            if iid:
                labels[iid] = e
    entries: list[dict] = []
    for smt2_path in sorted((root / split).glob("*.smt2")):
        lab = labels.get(smt2_path.stem, {})
        inst = smt2_to_instance(
            smt2_path,
            instance_id=lab.get("instance_id") or smt2_path.stem,
            family=lab.get("family", "imported"),
            description=lab.get("description", ""),
        )
        entries.append(
            instance_manifest_entry(inst, smt2_relpath=f"{split}/{smt2_path.name}")
        )
    return entries


def _parse_smt_num(token: str) -> Optional[Fraction]:
    token = token.strip()
    if not token:
        return None
    if token.startswith("(/"):
        inner = token.strip("()")
        parts = inner.split()
        if len(parts) == 3 and parts[0] == "/":
            return Fraction(int(parts[1]), int(parts[2]))
        return None
    try:
        return Fraction(int(token))
    except ValueError:
        return None


def _parse_smt_value_at(rest: str) -> Optional[Fraction]:
    """从 S 表达式尾部解析数值（整数或有理数 ``(/ n d)``）。"""
    rest = rest.lstrip()
    if not rest:
        return None
    if rest.startswith("(/"):
        depth = 0
        for i, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return _parse_smt_num(rest[: i + 1])
        return None
    m = re.match(r"(-?\d+)", rest)
    return _parse_smt_num(m.group(1)) if m else None


def _parse_get_value(stdout: str, obj_sexpr: str) -> Optional[Fraction]:
    """从 ``(get-value (obj))`` 响应解析最优值：取**目标表达式之后**的数值。"""
    head = stdout.split("(:", 1)[0]
    idx = head.find(obj_sexpr)
    if idx == -1:
        return None
    rest = head[idx + len(obj_sexpr) :].lstrip()
    while rest.startswith(")"):
        rest = rest[1:].lstrip()
    return _parse_smt_value_at(rest)


def _parse_z3_statistics(stdout: str) -> dict[str, int | float]:
    """解析 z3 ``-st`` 统计块中的 ``:keyword value`` 对。"""
    start = stdout.find("(:")
    if start == -1:
        return {}
    section = stdout[start:]
    out: dict[str, int | float] = {}
    for m in re.finditer(r":([-\w]+)\s+([-\d.]+)", section):
        key, raw = m.group(1), m.group(2)
        try:
            out[key] = int(raw) if "." not in raw else float(raw)
        except ValueError:
            continue
    return out


def _parse_rlimit(stdout: str) -> int:
    stats = _parse_z3_statistics(stdout)
    if "rlimit-count" in stats:
        return int(stats["rlimit-count"])
    m = re.search(r":rlimit-count\s+(\d+)", stdout)
    return int(m.group(1)) if m else 0


def _parse_sat(stdout: str) -> str:
    for line in stdout.splitlines():
        s = line.strip()
        if s in ("sat", "unsat", "unknown"):
            return s
    return "error"


def solve_binary(
    inst: OMTInstance,
    *,
    z3_path: str | None = None,
    timeout_s: int = 120,
    smt2: str | None = None,
) -> dict:
    """用 z3 二进制（``z3 -st``）求解 OMT，返回 ``value``/``rlimit``。

    SMT-LIB2 默认由 :func:`instance_to_smt2` 生成，与数据集落盘内容一致；
    可传入 ``smt2`` 覆盖（例如复用已保存的 ``.smt2`` 文件内容）。
    """
    exe = z3_path or shutil.which("z3")
    if not exe:
        raise FileNotFoundError("未找到 z3 二进制，请安装 z3 或通过 z3_path 指定")

    smt2_text = smt2 if smt2 is not None else instance_to_smt2(inst)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".smt2", delete=False, encoding="ascii"
    ) as tmp:
        tmp.write(smt2_text)
        path = tmp.name
    try:
        t0 = perf_counter()
        proc = subprocess.run(
            [exe, "-st", path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed_ms = (perf_counter() - t0) * 1000.0
    except subprocess.TimeoutExpired:
        return {
            "value": None,
            "rlimit": None,
            "conflicts": None,
            "decisions": None,
            "time_ms": timeout_s * 1000.0,
            "status": "timeout",
            "stderr": "timeout",
            "z3_stats": {},
        }
    finally:
        Path(path).unlink(missing_ok=True)

    stdout = proc.stdout or ""
    stderr = (proc.stderr or "").strip()
    obj_sexpr = inst.objective.sexpr()
    status = _parse_sat(stdout)
    z3_stats = _parse_z3_statistics(stdout)
    rlimit = int(z3_stats.get("rlimit-count", 0))
    value = _parse_get_value(stdout, obj_sexpr) if status == "sat" else None
    if status == "sat" and value is None:
        err_lines = [
            ln.strip() for ln in stdout.splitlines() if ln.strip().startswith("(error")
        ]
        if err_lines:
            stderr = (
                "; ".join(err_lines)
                if not stderr
                else f"{stderr}; {'; '.join(err_lines)}"
            )
    return {
        "value": value,
        "rlimit": rlimit,
        "conflicts": z3_stats.get("conflicts"),
        "decisions": z3_stats.get("decisions"),
        "time_ms": elapsed_ms,
        "status": status,
        "stderr": stderr,
        "returncode": proc.returncode,
        "z3_stats": z3_stats,
    }


__all__ = [
    "instance_to_smt2",
    "smt2_to_instance",
    "load_dataset",
    "instance_manifest_entry",
    "manifest_mismatches",
    "rebuild_manifest",
    "save_dataset",
    "list_split_entries",
    "solve_omt_with_decider",
    "solve_native",
    "solve_binary",
]
