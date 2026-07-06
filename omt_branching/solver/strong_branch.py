"""strong-branching 布尔专家：一层前瞻，用目标分离度给布尔原子打分。

在根状态对每个候选布尔原子 a 做一层前瞻 optimize(φ∧a) 与 optimize(φ∧¬a)，以两子最优
目标的分离度 |v_a − v_na| 打分——最能把目标分到两侧的原子，一侧出 incumbent 后另一侧会被
GOMT 的 Better 割廉价剪掉。恰有一侧 UNSAT 的原子近乎被 φ 蕴含、不产生真实进展，故记 0 分。
phase 目标为先探保留更优目标的一侧。候选原子取自**原始硬约束 φ**（不含增量 Better 割 δ0），
故不会把"目标上界割"误当结构原子。所有 z3 交互经 Z3Backend（不改系统 z3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

from omt_branching.solver.extractor import Z3SnapshotExtractor
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.z3_backend import Z3Backend


@dataclass(frozen=True)
class StrongBranchConfig:
    """strong-branching 专家配置。``max_atoms`` 限制每实例评测的候选原子数（按目标系数质量预筛）。"""

    max_atoms: int = 24
    eps: float = 1e-9


def _atom_obj_mass(var_coeffs: dict, obj_coeffs: dict) -> float:
    """原子涉及变量的 |目标系数| 质量，用于预筛（越大越可能与目标相关）。"""
    return sum(abs(obj_coeffs.get(v, 0.0)) for v in var_coeffs)


def _prefilter(snapshot, config: StrongBranchConfig) -> list[Hashable]:
    """按目标系数质量降序取前 ``max_atoms`` 个候选原子 bool_var_id。"""
    obj_coeffs = snapshot.objective.var_coeffs
    ranked = sorted(snapshot.theory_atoms,
                    key=lambda a: _atom_obj_mass(a.var_coeffs, obj_coeffs), reverse=True)
    return [a.bool_var_id for a in ranked[: config.max_atoms]]


def _root_extraction(instance: OMTInstance):
    """构造实例根状态并从**原始 φ**（不含 δ0 割）抽取。

    返回 ``(extraction, phi, objective, sense, backend)``；``φ`` 不可满足时抛异常（由调用方兜底）。
    """
    hard, obj, sense = instance.as_tuple()
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=hard, objective=obj, sense=sense)
    state = problem.initial_state(backend)          # 可能抛 Infeasible
    extraction = Z3SnapshotExtractor(problem).extract(state, backend)  # state.hard = 原始 φ
    phi = backend.conjoin(*hard)
    return extraction, phi, obj, sense, backend


def strong_branch_scores(extraction, phi, objective, sense: Sense, backend: Z3Backend,
                         config: StrongBranchConfig = StrongBranchConfig()):
    """给候选布尔原子打分。返回 ``(scores, phases)``，均以原子 bool_var_id 为键。

    - ``scores[bid] = |optimize(φ∧a) − optimize(φ∧¬a)|``；恰有一侧 UNSAT → 0（近乎被蕴含）。
    - ``phases[bid]``：先探保留更优目标的一侧（MAX→更高，MIN→更低）。
    """
    scores: dict[Hashable, float] = {}
    phases: dict[Hashable, bool] = {}
    for bid in _prefilter(extraction.snapshot, config):
        handle = extraction.atom_handles.get(bid)
        if handle is None:
            continue
        a = handle.z3_obj
        ra = backend.optimize(backend.conjoin(phi, a), objective, sense)
        rna = backend.optimize(backend.conjoin(phi, backend.negate(a)), objective, sense)
        if ra is None or rna is None:
            scores[bid] = 0.0
            continue
        try:
            va, vna = float(ra[1]), float(rna[1])
        except (TypeError, ValueError, OverflowError):
            continue
        scores[bid] = abs(va - vna)
        phases[bid] = (va >= vna) if sense is Sense.MAX else (va <= vna)
    return scores, phases


def oracle_bool_choice(instance: OMTInstance,
                       config: StrongBranchConfig = StrongBranchConfig()) -> Optional[Hashable]:
    """strong-branching 专家选择：目标分离度最大的原子 bool_var_id；无有意义分支返回 None。"""
    try:
        extraction, phi, obj, sense, backend = _root_extraction(instance)
    except Exception:
        return None
    scores, _ = strong_branch_scores(extraction, phi, obj, sense, backend, config)
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > config.eps else None


__all__ = ["StrongBranchConfig", "strong_branch_scores", "oracle_bool_choice"]
