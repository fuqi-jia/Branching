from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import (
    GOMTConfig, GOMTProblem, GOMTSolver, Z3Backend, solve_native,
)
from omt_branching.solver.instance_gen import generate_hard_lia_dataset
from omt_branching.solver.strategy import NumericHeuristicStrategy


@pytest.mark.parametrize("mode", ["largest_domain", "largest_coeff", "random", "strong"])
def test_heuristic_strategy_reaches_native_optimum(mode):
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=4, max_vars=5)[0]
    hard, obj, sense = inst.as_tuple()
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=hard, objective=obj, sense=sense)
    strat = NumericHeuristicStrategy(problem, mode=mode, seed=0)
    solver = GOMTSolver(problem, backend, strat, GOMTConfig(max_steps=5000, f_sat_mode="plain"))
    res = solver.run()
    assert res.value == solve_native(hard, obj, sense)     # plain 模式饱和到精确最优
