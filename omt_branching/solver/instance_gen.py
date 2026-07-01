"""合成 OMT(LIA) 实例生成器（训练/评测数据源）。

生成有界整数线性优化实例：每个变量 ``0 <= x_j <= ub``，若干形如
``Σ a_i x_i <= b`` 的非负系数约束（右端取可行且有约束力的值），目标为
``Σ c_i x_i``（正系数）。由于变量有界且约束右端非负，实例**必然可行**
（``x=0`` 满足）且目标**有界**，适合作 GOMT 求解的训练集。

也可作脚本运行，把生成的数据集摘要打印出来::

    python -m omt_branching.solver.instance_gen --count 8 --seed 0
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Hashable

import z3

from omt_branching.solver.interfaces import Sense


@dataclass
class OMTInstance:
    """一个单目标 OMT(LIA) 实例（z3 表达式载体）。"""

    instance_id: str
    variables: list = field(default_factory=list)      # list[z3.ArithRef]
    hard: list = field(default_factory=list)           # 硬约束表达式列表
    objective: object = None                            # 目标项
    sense: Sense = Sense.MAX
    obj_coeffs: dict[str, float] = field(default_factory=dict)  # 变量名 -> 目标系数

    def as_tuple(self):
        """返回 ``(hard_tuple, objective, sense)``，可直接喂给求解器/训练器。"""
        return tuple(self.hard), self.objective, self.sense


def generate_instance(instance_id: str, rng: random.Random, *,
                      n_vars: int = 3, n_constraints: int = 3, ub: int = 8,
                      max_coeff: int = 4, sense: Sense = Sense.MAX) -> OMTInstance:
    """生成单个实例。变量名以 ``instance_id`` 前缀保证跨实例不冲突。"""
    xs = [z3.Int(f"{instance_id}_x{j}") for j in range(n_vars)]
    names = [f"{instance_id}_x{j}" for j in range(n_vars)]

    hard: list = []
    for x in xs:
        hard.append(x >= 0)
        hard.append(x <= ub)

    for _ in range(n_constraints):
        coeffs = [rng.randint(0, max_coeff) for _ in range(n_vars)]
        if all(c == 0 for c in coeffs):
            coeffs[rng.randrange(n_vars)] = 1
        lhs = z3.Sum([c * x for c, x in zip(coeffs, xs)])
        max_lhs = sum(c * ub for c in coeffs)
        lo = max(1, int(max_lhs * 0.3))
        hi = max(lo, int(max_lhs * 0.8))
        b = rng.randint(lo, hi)
        hard.append(lhs <= b)

    obj_c = [rng.randint(1, max_coeff) for _ in range(n_vars)]
    objective = z3.Sum([c * x for c, x in zip(obj_c, xs)])

    return OMTInstance(
        instance_id=instance_id, variables=xs, hard=hard,
        objective=objective, sense=sense,
        obj_coeffs={n: float(c) for n, c in zip(names, obj_c)},
    )


def generate_dataset(count: int, seed: int = 0, *, id_prefix: str = "inst",
                     **kwargs) -> list[OMTInstance]:
    """生成 ``count`` 个实例的数据集（同一 ``seed`` 可复现）。"""
    rng = random.Random(seed)
    return [generate_instance(f"{id_prefix}{i}", rng, **kwargs) for i in range(count)]


def oracle_numeric_choice(instance: OMTInstance) -> Hashable:
    """启发式专家标签：选目标系数绝对值最大的数值变量（heuristic distillation）。

    用作 imitation 冷启动标签与评测“准确率”的参照专家。
    """
    if not instance.obj_coeffs:
        return None
    return max(instance.obj_coeffs, key=lambda n: abs(instance.obj_coeffs[n]))


def _main() -> None:
    parser = argparse.ArgumentParser(description="生成合成 OMT(LIA) 实例并打印摘要")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-vars", type=int, default=3)
    parser.add_argument("--n-constraints", type=int, default=3)
    parser.add_argument("--ub", type=int, default=8)
    args = parser.parse_args()

    ds = generate_dataset(args.count, seed=args.seed, n_vars=args.n_vars,
                          n_constraints=args.n_constraints, ub=args.ub)
    print(f"生成 {len(ds)} 个 OMT(LIA) 实例 (seed={args.seed}):")
    for inst in ds:
        print(f"  {inst.instance_id}: vars={len(inst.variables)} "
              f"hard={len(inst.hard)} obj_coeffs={inst.obj_coeffs} "
              f"oracle={oracle_numeric_choice(inst)}")


__all__ = [
    "OMTInstance",
    "generate_instance",
    "generate_dataset",
    "oracle_numeric_choice",
]


if __name__ == "__main__":
    _main()
