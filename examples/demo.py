"""端到端示例：合成一个 OMT 求解器快照，跑通 输入 -> 模型 -> 输出，
并演示一步 imitation 训练与一步 solver-in-the-loop 微调。

运行::

    python -m examples.demo
"""

from __future__ import annotations

import torch

from omt_branching import BranchingPolicyService
from omt_branching.input import (
    BooleanVarInfo,
    ClauseInfo,
    NumericVarInfo,
    ObjectiveInfo,
    SearchStateInfo,
    SolverSnapshot,
    TheoryAtomInfo,
)
from omt_branching.interfaces import AtomKind, ClauseKind, NodeType, SearchMode
from omt_branching.input.graph_builder import GraphBuilder
from omt_branching.model.finetune import (
    FinetuneConfig,
    SolverInLoopFinetuner,
    Trajectory,
    TrajectoryStep,
)
from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, RankingExample, TrainConfig


def make_snapshot() -> SolverSnapshot:
    """构造一个小型 OMT(LIA) 风格快照。

    场景: 3 个布尔变量（其中 b2 来自 soft constraint），2 个理论原子
    (x + y <= 5, x >= 2)，2 个整数变量 x,y（x 当前 LP 取分数值），
    目标 minimize 2x + 3y。
    """
    bool_vars = [
        BooleanVarInfo(var_id="b0", is_candidate=True, vsids_activity=0.8,
                       occurrence_count=5, pos_count=3, neg_count=2),
        BooleanVarInfo(var_id="b1", is_candidate=True, vsids_activity=0.3,
                       occurrence_count=2, pos_count=1, neg_count=1, phase_saved=True),
        BooleanVarInfo(var_id="b2", is_candidate=True, vsids_activity=0.1,
                       is_soft=True, occurrence_count=1, pos_count=1),
        BooleanVarInfo(var_id="b3", assignment=True, decision_level=1,
                       is_candidate=False, vsids_activity=0.5),
    ]
    clauses = [
        ClauseInfo(clause_id="c0", literals=[("b0", True), ("b1", False)],
                   kind=ClauseKind.ORIGINAL),
        ClauseInfo(clause_id="c1", literals=[("b1", True), ("b2", True)],
                   kind=ClauseKind.LEARNED, lbd=2),
    ]
    atoms = [
        TheoryAtomInfo(atom_id="a0", bool_var_id="b0", kind=AtomKind.LE,
                       var_coeffs={"x": 1.0, "y": 1.0}, rhs=5.0,
                       lp_value=4.5, slack=0.5, tightens_objective=True),
        TheoryAtomInfo(atom_id="a1", bool_var_id="b1", kind=AtomKind.GE,
                       var_coeffs={"x": 1.0}, rhs=2.0, lp_value=2.3),
    ]
    numerics = [
        NumericVarInfo(num_var_id="x", is_integer=True, lp_value=2.3,
                       lower_bound=0.0, upper_bound=10.0, is_fractional=True,
                       objective_coeff=2.0, pseudocost_up=1.2, pseudocost_down=0.8),
        NumericVarInfo(num_var_id="y", is_integer=True, lp_value=1.0,
                       lower_bound=0.0, upper_bound=10.0, is_fractional=False,
                       objective_coeff=3.0),
    ]
    objective = ObjectiveInfo(
        sense_is_min=True, incumbent=12.0, best_bound=9.5, gap=0.26,
        var_coeffs={"x": 2.0, "y": 3.0},
        soft_weights={"b2": 4.0},
        related_bounds={"x": (0.0, 10.0)},
    )
    state = SearchStateInfo(depth=3, decision_level=2, trail_length=6,
                            conflict_count=4, search_mode=SearchMode.LINEAR,
                            time_budget_left=30.0)
    return SolverSnapshot(
        bool_vars=bool_vars, clauses=clauses, theory_atoms=atoms,
        numeric_vars=numerics, objective=objective, search_state=state,
        snapshot_id="demo-0",
    )


def main() -> None:
    torch.manual_seed(0)
    snap = make_snapshot()

    # ---------------- 输入：建图 ----------------
    builder = GraphBuilder()
    graph = builder.build(snap)
    print("=== 输入部分 ===")
    print(graph.summary())
    print("候选布尔(局部索引):", graph.meta["candidate_bool_local"],
          "-> 求解器 id:", graph.meta["candidate_bool_ids"])

    # ---------------- 模型 + 输出：推理建议 ----------------
    service = BranchingPolicyService()
    advice = service.advise(snap)
    print("\n=== 输出部分 (推理建议) ===")
    print("use_gnn:", advice.use_gnn, "| confidence:", round(advice.confidence, 4))
    print("activity_priors:", {k: round(v, 4) for k, v in advice.activity_priors.items()})
    print("ranked_candidates:", advice.ranked_candidates)
    print("phase_suggestions:", advice.phase_suggestions)
    if advice.integer_split:
        s = advice.integer_split
        print(f"integer_split: {s.num_var_id} branch_up={s.branch_up} "
              f"score={s.score:.4f} dir_conf={s.direction_confidence:.4f}")
    print("diagnostics:", advice.diagnostics)

    # 演示与求解器 activity 的融合 (plan 6.2)
    solver_activity = {"b0": 0.8, "b1": 0.3, "b2": 0.1}
    mixed = advice.mixed_activity(solver_activity, alpha=0.5, beta=0.5)
    print("mixed_activity:", {k: round(v, 4) for k, v in mixed.items()})

    # ---------------- 模型：一步 imitation 训练 ----------------
    print("\n=== 模型部分 (imitation 训练) ===")
    policy = service.policy
    trainer = ImitationTrainer(policy, TrainConfig(lr=1e-3))
    # 专家偏好: b0 最该分支; x 该做整数 split 且向上; b1 取真
    bmap = graph.id_maps[NodeType.BOOL_VAR]
    nmap = graph.id_maps[NodeType.NUMERIC_VAR]
    example = RankingExample(
        graph=graph,
        bool_target_scores={bmap["b0"]: 2.0, bmap["b1"]: 1.0, bmap["b2"]: 0.0},
        phase_targets={bmap["b1"]: True, bmap["b0"]: False},
        int_target_scores={nmap["x"]: 2.0, nmap["y"]: 0.0},
        int_dir_targets={nmap["x"]: True},
        conflict_targets={bmap["b0"]: True},
        obj_improve_targets={bmap["b0"]: 1.5},
    )
    for step in range(5):
        parts = trainer.train_step(example)
        print(f"step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in parts.items()))

    # ---------------- 模型：一步 solver-in-the-loop 微调 ----------------
    print("\n=== 模型部分 (REINFORCE 微调) ===")
    finetuner = SolverInLoopFinetuner(policy, FinetuneConfig(lr=3e-4))
    chosen = graph.meta["candidate_bool_local"][0]
    traj = Trajectory(
        steps=[TrajectoryStep(graph=graph, chosen_bool_local=chosen, reward=0.0)],
        terminal_reward=1.0,  # 例如 -log(1+solve_time) 的归一化奖励
    )
    stats = finetuner.reinforce_update(traj)
    print("reinforce:", {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in stats.items()})

    print("\n端到端流程完成。")


if __name__ == "__main__":
    main()
