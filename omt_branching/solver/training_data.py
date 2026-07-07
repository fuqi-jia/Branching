"""从 OMT 实例构造 GNN 监督（imitation）训练数据，并提供策略 top 选择查询。

- :func:`build_imitation_examples`：对每个实例取**初始状态**的图快照，用启发式专家
  （目标系数越大越该分支，见 :func:`~omt_branching.solver.instance_gen.oracle_numeric_choice`）
  给数值候选打分，产出 :class:`RankingExample` 冷启动标签（plan 5.2 heuristic distillation）。
- :func:`policy_numeric_choice`：在初始状态图上跑策略，返回其 top-1 数值分支变量 id，
  用于评测“分支选择准确率”（与专家一致的比例）。

均经 ``omt_branching.solver`` 的 adapter（Z3Backend / Z3SnapshotExtractor）与 z3 交互。
"""

from __future__ import annotations

from typing import Hashable, Optional

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import RankingExample
from omt_branching.solver.extractor import Z3SnapshotExtractor
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.z3_backend import Z3Backend


def _initial_extraction(instance: OMTInstance):
    """构造实例根状态并从**原始 φ** 抽取 (graph, extraction, backend)；不可行返回 (None, None, None)。

    从 ``state.hard``（原始 φ，不含 δ0 割）抽取，使布尔候选=结构原子，排除目标上界割。
    """
    hard, obj, sense = instance.as_tuple()
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=hard, objective=obj, sense=sense)
    try:
        state = problem.initial_state(backend)
    except Exception:
        return None, None, None
    extractor = Z3SnapshotExtractor(problem)
    extraction = extractor.extract(state, backend)
    builder = GraphBuilder(DEFAULT_FEATURE_SPEC)
    graph = builder.build(extraction.snapshot)
    return graph, extraction, backend


def build_imitation_example(instance: OMTInstance, config: "StrongBranchConfig" = None,
                            numeric_expert: str = "coeff") -> Optional[RankingExample]:
    """把单实例根图打包成 :class:`RankingExample`：数值 head + bool head 标签。

    ``numeric_expert``：``"coeff"``（默认，数值 head 用 |目标系数|，LRA/rl_demo 兼容）或
    ``"strong"``（数值 head 用整数 strong-branching 目标分离度，LIA B&B 实验用）。
    bool head 始终用 strong-branching 专家（LRA 主路径）。
    """
    from omt_branching.solver.strong_branch import (
        StrongBranchConfig, strong_branch_numeric_scores, strong_branch_scores,
    )

    cfg = config or StrongBranchConfig()
    graph, extraction, backend = _initial_extraction(instance)
    if graph is None:
        return None

    # --- 数值 head 标签 ---
    nmap = graph.id_maps.get(NodeType.NUMERIC_VAR, {})
    is_max = instance.sense.value == "max"
    int_scores: dict[int, float] = {}
    int_dirs: dict[int, bool] = {}
    if numeric_expert == "strong":
        hard, obj, sense = instance.as_tuple()
        phi = backend.conjoin(*hard)
        raw_num, raw_dir = strong_branch_numeric_scores(extraction, phi, obj, sense, backend, cfg)
        for vid, sc in raw_num.items():
            local = nmap.get(vid)
            if local is not None:
                int_scores[local] = sc
        for vid, up in raw_dir.items():
            local = nmap.get(vid)
            if local is not None:
                int_dirs[local] = up
    else:  # "coeff"：|目标系数|（LRA/rl_demo 兼容）
        for nv in extraction.snapshot.numeric_vars:
            local = nmap.get(nv.num_var_id)
            if local is None:
                continue
            int_scores[local] = abs(nv.objective_coeff)
            # 朝改善目标的方向 split：MAX 且正系数 -> 向上；MIN 且正系数 -> 向下。
            int_dirs[local] = (nv.objective_coeff >= 0) == is_max

    # --- bool head 标签（strong branching；LRA 主路径）---
    # LIA（numeric_expert="strong"）只用数值 head，bool head 不参与求解，故跳过昂贵且无用的
    # bool strong 标签计算。
    bool_scores: dict[int, float] = {}
    phase_targets: dict[int, bool] = {}
    if numeric_expert != "strong":
        hard, obj, sense = instance.as_tuple()
        phi = backend.conjoin(*hard)
        raw_scores, raw_phases = strong_branch_scores(extraction, phi, obj, sense, backend, cfg)
        bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
        for bid, sc in raw_scores.items():
            local = bmap.get(bid)
            if local is not None:
                bool_scores[local] = sc
        for bid, ph in raw_phases.items():
            local = bmap.get(bid)
            if local is not None:
                phase_targets[local] = ph

    if not int_scores and not bool_scores:
        return None
    return RankingExample(graph=graph, int_target_scores=int_scores, int_dir_targets=int_dirs,
                          bool_target_scores=bool_scores, phase_targets=phase_targets)


def build_imitation_examples(instances, numeric_expert: str = "coeff") -> list[RankingExample]:
    """对一组实例批量构造 imitation 训练样本（跳过无数值候选/不可行者）。"""
    out: list[RankingExample] = []
    for inst in instances:
        ex = build_imitation_example(inst, numeric_expert=numeric_expert)
        if ex is not None:
            out.append(ex)
    return out


def baseline_numeric_choice(instance: OMTInstance) -> Optional[Hashable]:
    """复刻 :class:`BaselineStrategy` 的 root 选择（最大域跨度变量），用于准确率对比。"""
    graph, extraction, _ = _initial_extraction(instance)
    if graph is None:
        return None
    best: Optional[Hashable] = None
    best_span = 0.0
    for handle in extraction.numeric_handles.values():
        if handle.lower is None or handle.upper is None:
            continue
        span = handle.upper - handle.lower
        if span >= 1 and span > best_span:
            best, best_span = handle.var_id, span
    return best


def policy_numeric_choice(policy: BranchingPolicy,
                          instance: OMTInstance) -> Optional[Hashable]:
    """返回策略在初始状态选择的 top-1 数值分支变量 id（用于准确率评测）。"""
    graph, extraction, _ = _initial_extraction(instance)
    if graph is None:
        return None
    out = policy.infer(graph)
    probs = out.masked_numeric_probs()
    if probs.numel() == 0 or not out.candidate_numeric_local:
        return None
    import torch

    local = int(torch.argmax(probs).item())
    return graph.solver_id(NodeType.NUMERIC_VAR, local)


def bool_branch_hit(policy: BranchingPolicy, instance: OMTInstance,
                    config: "StrongBranchConfig" = None) -> Optional[bool]:
    """在**同一根抽取**上比较策略 bool-head top-1 与 strong-branching 专家是否一致。

    无有意义分离原子（专家空/全 0）或无 bool 候选时返回 ``None``（该实例不计入准确率）。
    单次抽取保证专家与策略共享同一套原子 id，规避跨抽取 id 漂移。
    """
    import torch

    from omt_branching.solver.strong_branch import StrongBranchConfig, strong_branch_scores

    cfg = config or StrongBranchConfig()
    graph, extraction, backend = _initial_extraction(instance)
    if graph is None:
        return None
    hard, obj, sense = instance.as_tuple()
    phi = backend.conjoin(*hard)
    scores, _ = strong_branch_scores(extraction, phi, obj, sense, backend, cfg)
    if not scores or max(scores.values()) <= cfg.eps:
        return None
    oracle_bid = max(scores, key=lambda k: scores[k])

    out = policy.infer(graph)
    probs = out.masked_bool_probs()
    if probs.numel() == 0 or not out.candidate_bool_local:
        return None
    local = int(torch.argmax(probs).item())
    return graph.solver_id(NodeType.BOOL_VAR, local) == oracle_bid


__all__ = [
    "build_imitation_example",
    "build_imitation_examples",
    "policy_numeric_choice",
    "baseline_numeric_choice",
    "bool_branch_hit",
]
