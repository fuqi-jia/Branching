from __future__ import annotations

import random

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver.instance_gen import (
    generate_hard_lia_instance, generate_hard_lia_dataset, _validate,
)


def test_hard_lia_instance_feasible_bounded_integer():
    inst = generate_hard_lia_instance("h0", random.Random(0), n_vars=10, n_constraints=8)
    assert _validate(inst)                      # witness 驱动 -> 必 SAT
    assert inst.theory == "LIA" and len(inst.variables) == 10
    assert all(z3.is_int(v) for v in inst.variables)
    assert inst.obj_coeffs and len(inst.obj_coeffs) == 10


def test_hard_lia_dataset_reproducible():
    a = generate_hard_lia_dataset(5, seed=1, min_vars=8, max_vars=10)
    b = generate_hard_lia_dataset(5, seed=1, min_vars=8, max_vars=10)
    assert [i.instance_id for i in a] == [i.instance_id for i in b]
    assert len(a) == 5 and all(i.theory == "LIA" for i in a)
