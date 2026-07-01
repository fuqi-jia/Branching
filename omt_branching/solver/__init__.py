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
    BaselineStrategy, NeuralStrategy, StrategyConfig,
)
from omt_branching.solver.bridge import BridgeConfig, NeuralGOMTSolver, solve_native
from omt_branching.solver.rl import (
    RLConfig, RLEpisode, RLRecordingStrategy, RLStep, SolverInLoopRLTrainer,
    solve_and_measure,
)
from omt_branching.solver.instance_gen import (
    OMTInstance, generate_dataset, generate_instance, oracle_numeric_choice,
)
from omt_branching.solver.training_data import (
    build_imitation_example, build_imitation_examples, policy_numeric_choice,
    baseline_numeric_choice,
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
    "StrategyConfig",
    # 门面
    "NeuralGOMTSolver",
    "BridgeConfig",
    "solve_native",
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
    "oracle_numeric_choice",
    # 训练数据
    "build_imitation_example",
    "build_imitation_examples",
    "policy_numeric_choice",
    "baseline_numeric_choice",
]
