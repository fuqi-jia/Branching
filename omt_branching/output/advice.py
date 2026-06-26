"""【输出接口】返回给求解器的信息格式。

这是本框架交还给求解器的**输出契约**。所有 id 都是求解器最初在
:class:`SolverSnapshot` 中提供的原始 id（已由解码器从图内索引映射回来）。

求解器的典型用法（VSIDS refocus，plan 7.1）:

1. 若 ``use_gnn is False``：忽略本建议，使用原生 heuristic。
2. 否则把 ``activity_priors`` 按 plan 6.2 的融合公式混入 SAT activity::

       score(v) = alpha * normalized_solver_activity(v)
                + beta  * activity_priors[v]
                + gamma * theory_score(v)

3. ``phase_suggestions`` 作为 phase saving 的偏置。
4. 整数 B&B 时参考 ``integer_split`` / ``ranked_integer_candidates``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Optional


@dataclass
class IntegerSplitAdvice:
    """单个整数变量的 B&B split 建议。"""

    num_var_id: Hashable
    branch_up: bool          # True: 优先 x >= ceil(v); False: x <= floor(v)
    score: float             # 选择该变量的偏好分数
    direction_confidence: float = 0.0  # 方向 logit 的 sigmoid 置信度


@dataclass
class BranchingAdvice:
    """一次分支咨询的完整返回。

    字段语义对求解器是稳定的接口，新增字段应保持向后兼容（追加而非修改）。
    """

    # --- 布尔层 (阶段 A/B) ---
    #: 归一化的 activity 先验，可直接按权重混入 VSIDS。键为求解器布尔变量 id。
    activity_priors: dict[Hashable, float] = field(default_factory=dict)
    #: 候选布尔变量按偏好降序排列（最该分支的在前）。
    ranked_candidates: list[Hashable] = field(default_factory=list)
    #: 极性建议：var_id -> 优先取真(True)/取假(False)。
    phase_suggestions: dict[Hashable, bool] = field(default_factory=dict)

    # --- 整数 B&B 层 (阶段 C) ---
    #: 最优整数 split 建议（无整数候选时为 None）。
    integer_split: Optional[IntegerSplitAdvice] = None
    #: 整数候选按偏好降序，便于 reliability/pseudo-cost 融合。
    ranked_integer_candidates: list[IntegerSplitAdvice] = field(default_factory=list)

    # --- 控制与可解释信息 ---
    #: 是否建议采用 GNN 建议；False 表示求解器应回退原生 heuristic。
    use_gnn: bool = True
    #: top-1 候选的概率，作为置信度。
    confidence: float = 0.0
    #: 辅助预测：var_id -> {conflict_prob, in_core_prob, obj_improve, subtree_size}。
    aux_predictions: dict[Hashable, dict[str, float]] = field(default_factory=dict)
    #: 诊断信息（推理耗时、回退原因、图规模等）。
    diagnostics: dict[str, float | str | int] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def top_candidate(self) -> Optional[Hashable]:
        """最该分支的布尔变量；无候选返回 None。"""
        return self.ranked_candidates[0] if self.ranked_candidates else None

    def mixed_activity(self, solver_activity: dict[Hashable, float],
                       alpha: float = 1.0, beta: float = 1.0) -> dict[Hashable, float]:
        """便捷融合：``alpha * 原生 activity + beta * GNN 先验`` (plan 6.2)。

        仅在 ``use_gnn`` 时混入先验，否则原样返回求解器 activity。
        """
        if not self.use_gnn:
            return dict(solver_activity)
        keys = set(solver_activity) | set(self.activity_priors)
        return {
            k: alpha * solver_activity.get(k, 0.0) + beta * self.activity_priors.get(k, 0.0)
            for k in keys
        }
