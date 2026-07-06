from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import Sense, Z3Backend


def _box():
    x, y = z3.Ints("x y")
    base = z3.And(x >= 0, x <= 10, y >= 0, y <= 10)
    return x, y, base


def test_solve_branch_matches_nonincremental():
    x, y, base = _box()
    be = Z3Backend()
    m = be.solve_branch(base, x >= 7)
    assert m is not None and be.value(m, x) >= 7
    # 不可行分支返回 None
    assert be.solve_branch(base, x >= 100) is None


def test_optimize_branch_matches_and_reuses():
    x, y, base = _box()
    be = Z3Backend()
    r1 = be.optimize_branch(base, x <= 4, x, Sense.MAX)
    r2 = be.optimize_branch(base, x >= 6, x, Sense.MAX)   # 复用同一 optimizer
    assert r1 is not None and r1[1] == 4
    assert r2 is not None and r2[1] == 10
    # 与非增量 optimize 一致
    one = be.optimize(z3.And(base, x >= 6), x, Sense.MAX)
    assert one[1] == r2[1]


def test_incumbent_model_survives_further_incremental_calls():
    x, y, base = _box()
    be = Z3Backend()
    m = be.solve_branch(base, x >= 7)          # 取得 incumbent 模型
    _ = be.solve_branch(base, x <= 2)          # 后续增量调用不应使旧模型失效
    assert be.value(m, x) >= 7                  # 旧模型仍可正确取值


def test_incremental_rlimit_accumulates_as_deltas():
    x, y, base = _box()
    be = Z3Backend()
    be.optimize_branch(base, x <= 4, x, Sense.MAX)
    r1 = be.rlimit_count
    be.optimize_branch(base, x >= 6, x, Sense.MAX)
    r2 = be.rlimit_count
    # 单调增，且每步增量有界（不是把累计总量重复相加）
    assert r2 > r1 >= 0
    assert (r2 - r1) <= r1 + 1000        # 第二步增量不应爆炸式等于整个累计
    assert be.solve_calls == 2
