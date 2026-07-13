"""桥接门面：把 problem + backend + extractor + strategy + service 组装成单一入口。

求解器侧只需持有一个 :class:`NeuralGOMTSolver`，调用 ``solve(hard, objective, sense)``
即可用 Neural 策略驱动的 GOMT calculus（z3 为理论后端）求解单目标 OMT，并拿到
最优解、最优值与统计。``solve_native`` 用 z3 原生 ``Optimize`` 作参照 oracle。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omt_branching.service import BranchingPolicyService
from omt_branching.solver.calculus import GOMTConfig, GOMTResult, GOMTSolver
from omt_branching.solver.interfaces import Sense, Value
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.strategy import BaselineStrategy, NeuralStrategy
from omt_branching.solver.z3_backend import Z3Backend


@dataclass(frozen=True)
class BridgeConfig:
    """门面配置。

    - ``strategy``：``"neural"`` 用 Neural F-Split 驱动；``"baseline"`` 用确定性二分基线。
    - ``f_sat_mode``：``"plain"``（默认，Neural 驱动）或 ``"hybrid"``（z3 Optimize 加速）。
    - ``max_steps``：派生步数上限。
    """

    strategy: str = "neural"
    f_sat_mode: str = "plain"
    max_steps: int = 10_000


class NeuralGOMTSolver:
    """Neural 策略驱动的 GOMT OMT 求解器门面。"""

    def __init__(
        self,
        service: Optional[BranchingPolicyService] = None,
        config: BridgeConfig = BridgeConfig(),
    ):
        self.service = service
        self.config = config

    def solve(self, hard_list, objective, sense: Sense) -> GOMTResult:
        """求解单目标 OMT 实例。

        - ``hard_list``：硬约束 ``φ`` 的合取项（后端表达式序列）。
        - ``objective``：目标项 ``t``。
        - ``sense``：``Sense.MIN`` / ``Sense.MAX``。
        """
        backend = Z3Backend()
        problem = GOMTProblem(
            hard_list=tuple(hard_list), objective=objective, sense=sense
        )
        if self.config.strategy == "baseline":
            strategy = BaselineStrategy(problem)
        else:
            strategy = NeuralStrategy(problem, self.service or BranchingPolicyService())
        solver = GOMTSolver(
            problem,
            backend,
            strategy,
            GOMTConfig(
                max_steps=self.config.max_steps, f_sat_mode=self.config.f_sat_mode
            ),
        )
        return solver.run()


'''
def solve_native(hard_list, objective, sense: Sense) -> Value:
    """参照 oracle：z3 原生 ``Optimize`` 的最优值。"""
    backend = Z3Backend()
    result = backend.optimize(backend.conjoin(*hard_list), objective, sense)
    if result is None:
        raise ValueError("native optimize: 硬约束不可满足")
    return result[1]
'''


__all__ = [
    "NeuralGOMTSolver",
    "BridgeConfig",
    # "solve_native",
]
