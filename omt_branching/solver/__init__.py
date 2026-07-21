"""z3 ↔ Neural GOMT 桥接子包。

把外部 OMT 求解器（z3）与现有 Neural 分支策略组件联通起来，理论骨架是
GOMT calculus（``GOMT.pdf`` 第 4 节）。Neural 策略精确地插入到 calculus 唯一的
启发式自由点 —— F-Split 规则。接口说明见 ``omt_branching/solver/API.md``。

典型用法::

    from omt_branching.solver import NeuralGOMTSolver, Sense
    res = NeuralGOMTSolver().solve(hard_list, objective, Sense.MIN)
    print(res.value, res.optimal, res.stats)
"""

from __future__ import annotations

from omt_branching.solver.interfaces import (
    BranchingStrategy, GOMTState, Sense, SolveBackend, SplitDecision,
)
from omt_branching.solver.problem import GOMTProblem, Infeasible
from omt_branching.solver.calculus import GOMTConfig, GOMTResult, GOMTSolver
from omt_branching.solver.z3_backend import Unbounded, Z3Backend
from omt_branching.solver.extractor import Extraction, Handle, Z3SnapshotExtractor
from omt_branching.solver.strategy import (
    BaselineStrategy, NeuralStrategy, NumericHeuristicStrategy, StrategyConfig,
)
from omt_branching.solver.bridge import BridgeConfig, NeuralGOMTSolver
from omt_branching.solver.rl import (
    RLConfig, RLEpisode, RLRecordingStrategy, RLStep, SolverInLoopRLTrainer,
    solve_and_measure,
)
from omt_branching.solver.instance_gen import (
    LRA_FAMILIES, OMTInstance, BranchFocusInstance,
    generate_bool_lia_dataset, generate_bool_lia_instance, generate_hard_bool_lia_dataset,
    generate_branch_focus_lia_dataset, generate_branch_focus_lia_instance,
    generate_dataset, generate_hard_lia_dataset, generate_hard_lia_instance,
    generate_instance, generate_lra_dataset, generate_lra_instance, oracle_numeric_choice,
)
from omt_branching.solver.training_data import (
    build_imitation_example, build_imitation_examples, policy_numeric_choice,
    baseline_numeric_choice, bool_branch_hit,
)
from omt_branching.solver.propagator_snapshot import (
    atom_key,
    build_bool_snapshot,
    clear_bool_snapshot_cache,
    collect_atoms,
    collect_clause_atoms,
    merge_root_assignment,
    prepare_propagator_formula,
    preprocess_assertions,
    root_forced_assignment,
)
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.policy_decider import PolicyDecider, solve_with_learned_policy
from omt_branching.solver.decide_omt import (
    instance_to_smt2, load_dataset, smt2_to_instance,
    solve_binary, solve_omt_with_decider, solve_native,
    instance_manifest_entry, manifest_mismatches, rebuild_manifest,
    save_dataset, list_split_entries,
)
from omt_branching.solver.binary_results import (
    REF_SUBDIR,
    binary_rlimit,
    binary_value,
    build_ref_payload,
    check_sat_loop_stats_from_ref,
    has_binary_result,
    is_fair_vsids_cache,
    load_binary_result,
    load_binary_results,
    missing_binary_ids,
    save_binary_result,
    vsids_stats_from_ref,
)
from omt_branching.solver.lookahead_cache import (
    has_lookahead_result, load_lookahead_result, save_lookahead_result,
)
from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat, generate_hard_smt_lia
from omt_branching.solver.sat_solve import solve_sat_with_decider
from omt_branching.solver.vsids_trace import (
    VSIDSTraceConfig, build_vsids_examples_sat, collect_vsids_trajectory,
)
from omt_branching.solver.strong_branch import (
    StrongBranchConfig, oracle_bool_choice, oracle_numeric_choice_sb,
    strong_branch_numeric_scores, strong_branch_scores,
)
__all__ = [
    # 接口与数据类型
    "Sense",
    "SplitDecision",
    "GOMTState",
    "SolveBackend",
    "BranchingStrategy",
    # 问题与 calculus
    "GOMTProblem",
    "Infeasible",
    "GOMTSolver",
    "GOMTConfig",
    "GOMTResult",
    # z3 adapter
    "Z3Backend",
    "Unbounded",
    "Z3SnapshotExtractor",
    "Extraction",
    "Handle",
    # 策略
    "NeuralStrategy",
    "BaselineStrategy",
    "NumericHeuristicStrategy",
    "StrategyConfig",
    # 门面
    "NeuralGOMTSolver",
    "BridgeConfig",
    # Solver-in-the-Loop 强化学习
    "SolverInLoopRLTrainer",
    "RLConfig",
    "RLEpisode",
    "RLStep",
    "RLRecordingStrategy",
    "solve_and_measure",
    # 实例生成
    "OMTInstance",
    "generate_instance",
    "generate_dataset",
    "generate_lra_instance",
    "generate_lra_dataset",
    "generate_hard_lia_instance",
    "generate_hard_lia_dataset",
    "generate_bool_lia_instance",
    "generate_bool_lia_dataset",
    "generate_hard_bool_lia_dataset",
    "BranchFocusInstance",
    "generate_branch_focus_lia_instance",
    "generate_branch_focus_lia_dataset",
    "LRA_FAMILIES",
    "oracle_numeric_choice",
    # 训练数据
    "build_imitation_example",
    "build_imitation_examples",
    "policy_numeric_choice",
    "baseline_numeric_choice",
    "bool_branch_hit",
    # strong-branching 专家
    "StrongBranchConfig",
    "oracle_bool_choice",
    "strong_branch_scores",
    "oracle_numeric_choice_sb",
    "strong_branch_numeric_scores",
    # UserPropagator 学习分支（Phase 1）
    "atom_key",
    "collect_atoms",
    "collect_clause_atoms",
    "preprocess_assertions",
    "prepare_propagator_formula",
    "root_forced_assignment",
    "merge_root_assignment",
    "build_bool_snapshot",
    "clear_bool_snapshot_cache",
    "LearnedDecidePropagator",
    "PolicyDecider",
    "solve_with_learned_policy",
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
    "REF_SUBDIR",
    "binary_rlimit",
    "binary_value",
    "build_ref_payload",
    "check_sat_loop_stats_from_ref",
    "has_binary_result",
    "is_fair_vsids_cache",
    "load_binary_result",
    "load_binary_results",
    "missing_binary_ids",
    "save_binary_result",
    "vsids_stats_from_ref",
    "has_lookahead_result",
    "load_lookahead_result",
    "save_lookahead_result",
    "generate_php",
    "generate_rand_3sat",
    "generate_hard_smt_lia",
    "solve_sat_with_decider",
    "VSIDSTraceConfig",
    "build_vsids_examples_sat",
    "collect_vsids_trajectory",
]
