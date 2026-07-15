"""OMT = 单 z3.Solver 线性搜索回路（Solve + Better-cut，直到 UNSAT），可挂
LearnedDecidePropagator 接管内部布尔决策。z3.Optimize 不支持 propagator，故必须走此回路。

对比臂：``solve_native``（Python Optimize API）、``solve_binary``（z3 二进制 ``-st``）、
``solve_omt_with_decider``（VSIDS / learned UserPropagator 回路）。
"""

from __future__ import annotations

from fractions import Fraction
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
from omt_branching.solver.propagator_snapshot import collect_atoms


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


def solve_omt_with_decider(
    hard,
    objective,
    sense: Sense,
    decider_factory=None,
    max_iters: int = 100000,
    ctx: z3.Context | None = None,
) -> dict:
    """OMT 线性搜索；默认在独立 :class:`z3.Context` 内运行，避免跨线程/跨求解共享表达式。"""
    if ctx is None:
        ctx = z3.Context()
    hard_iso = [h.translate(ctx) for h in hard]
    obj_iso = objective.translate(ctx)

    s = z3.Solver(ctx=ctx)
    solver_rlimit = _stat(s, "rlimit count")
    rlimit = solver_rlimit
    prop = None
    if decider_factory is not None:
        atoms = collect_atoms(hard_iso)
        decider = decider_factory(hard_iso)
        prop = LearnedDecidePropagator(s, atoms, decider)
    decider_factory_rlimit = _stat(s, "rlimit count") - rlimit
    rlimit += decider_factory_rlimit

    s.add(*hard_iso)
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
        cut = obj_iso > best_val if sense is Sense.MAX else obj_iso < best_val
        s.add(cut)
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

        records.append((_num(best_val), check_rlimit[-1] + eval_rlimit[-1]))

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

    local, cost = records[0]
    weighted_rlimit = len(records) * cost
    for i in range(1, len(records)):
        last_local = local
        local, cost = records[i]
        weighted_rlimit += (
            # max(
            #     (stats["value"] - last_local) / (local - last_local),
            #     len(records) - i,
            # )
            (len(records) - i)
            * cost
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
            ln.strip()
            for ln in stdout.splitlines()
            if ln.strip().startswith("(error")
        ]
        if err_lines:
            stderr = "; ".join(err_lines) if not stderr else f"{stderr}; {'; '.join(err_lines)}"
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
    "solve_omt_with_decider",
    "solve_native",
    "solve_binary",
]
