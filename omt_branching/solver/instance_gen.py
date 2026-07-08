"""合成 OMT 实例生成器（训练/评测数据源）。

包含两类实例：

- **LIA**（:func:`generate_instance`）：有界整数线性优化。每个变量 ``0 <= x_j <= ub``，
  若干 ``Σ a_i x_i <= b`` 约束，目标 ``Σ c_i x_i``。变量有界且右端非负 -> 必然可行、
  目标有界。

- **LRA**（:func:`generate_lra_instance`，参照 ``docs/ref/LRA_script.py``）：带**布尔结构**
  （and/or/not、分段析取、蕴含）的实数线性优化。采用 **witness 驱动**：先采样一个满足
  赋值，再据此构造所有断言，使 witness 满足每条断言，从而实例高概率 SAT；配合非负 +
  松上界盒约束保证目标有界、可行域闭合（最优可达且为有理数）。

.. note::
    LRA 求解语义：真最优由 z3 原生 ``Optimize``（:func:`omt_branching.solver.solve_native`）
    直接给出（毫秒级）。GOMT 的**增量式线性搜索**（F-Split + 严格 Better 割）是为**离散**
    （LIA/B&B）设计的——对**实数**变量的连续域二分既非完备也不有限终止，故：

    - 实数变量**不做域二分**：GNN 只在**布尔结构**上分支，连续部分交给 z3
      （``f_sat_mode="hybrid"`` 的叶子 ``Optimize``）；见 ``strategy.py`` 中的整数门控。
    - LRA 训练/评测应使用**有界 episode**（``RLConfig.max_steps`` 预算）：incumbent 提升 +
      rlimit 代价构成有效的策略梯度信号；``optimal`` 可能为 ``False``（anytime），用
      ``solve_native`` 作为准确率/最优性的参照真值。

作脚本运行::

    python -m omt_branching.solver.instance_gen --theory lia --count 8 --seed 0
    python -m omt_branching.solver.instance_gen --theory lra --count 6 --seed 0 --validate
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Hashable, Optional, Sequence

import z3

from omt_branching.solver.interfaces import Sense


@dataclass
class OMTInstance:
    """一个单目标 OMT 实例（z3 表达式载体）。"""

    instance_id: str
    variables: list = field(default_factory=list)      # list[z3.ArithRef]
    hard: list = field(default_factory=list)           # 硬约束表达式列表
    objective: object = None                            # 目标项
    sense: Sense = Sense.MAX
    obj_coeffs: dict[str, float] = field(default_factory=dict)  # 变量名 -> 目标系数
    theory: str = "LIA"                                 # "LIA" | "LRA"
    family: str = "box"                                 # LRA 结构族，见 LRA_FAMILIES
    description: str = ""

    def as_tuple(self):
        """返回 ``(hard_tuple, objective, sense)``，可直接喂给求解器/训练器。"""
        return tuple(self.hard), self.objective, self.sense


# =========================================================================== #
# LIA 生成（整数线性优化）
# =========================================================================== #
def generate_instance(instance_id: str, rng: random.Random, *,
                      n_vars: int = 3, n_constraints: int = 3, ub: int = 8,
                      max_coeff: int = 4, sense: Sense = Sense.MAX) -> OMTInstance:
    """生成单个 LIA 实例。变量名以 ``instance_id`` 前缀保证跨实例不冲突。"""
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
        theory="LIA", family="box", description=f"LIA box, {n_vars} vars",
    )


def generate_dataset(count: int, seed: int = 0, *, id_prefix: str = "inst",
                     **kwargs) -> list[OMTInstance]:
    """生成 ``count`` 个 LIA 实例的数据集（同一 ``seed`` 可复现）。"""
    rng = random.Random(seed)
    return [generate_instance(f"{id_prefix}{i}", rng, **kwargs) for i in range(count)]


def generate_bool_lia_instance(instance_id: str, rng: random.Random, *,
                               n_vars: int = 5, n_disj: int = 8, k: int = 3, ub: int = 8,
                               chi: int = 4, pool_mult: int = 2, sense: Sense = Sense.MAX) -> OMTInstance:
    """带**布尔结构**的有界整数 OMT：从一个**共享原子池**采样构造 ``k`` 元析取(``Or``)——
    原子在多个子句间**复用**(度>1),使子句共现图有信息、布尔传播可学(供 UserPropagator 学习
    分支研究)。witness 驱动保证 SAT + 有界。"""
    xs = [z3.Int(f"{instance_id}_x{j}") for j in range(n_vars)]
    names = [f"{instance_id}_x{j}" for j in range(n_vars)]
    witness = [rng.randint(0, ub) for _ in range(n_vars)]

    hard: list = []
    for x in xs:
        hard.append(x >= 0)
        hard.append(x <= ub)

    # 共享原子池：每个原子记录在 witness 上是否成立（用于保证每个子句 SAT）。
    n_pool = max(k + 2, n_vars * pool_mult)
    pool: list = []
    for _ in range(n_pool):
        coeffs = [rng.randint(1, chi) for _ in range(n_vars)]
        lhs = z3.Sum([c * x for c, x in zip(coeffs, xs)])
        v = sum(c * w for c, w in zip(coeffs, witness))
        b = v + rng.randint(-3, 4)             # 有时 witness 满足、有时不满足
        pool.append((lhs <= b, sum(c * w for c, w in zip(coeffs, witness)) <= b))
    if not any(h for _, h in pool):            # 保证池里至少有一个 witness-true 原子
        pool[0] = (xs[0] <= ub, True)

    true_pool = [ah for ah in pool if ah[1]]
    for _ in range(n_disj):
        chosen = rng.sample(pool, min(k, len(pool)))
        if not any(h for _, h in chosen):      # 每个子句至少含一个 witness-true 原子 -> SAT
            chosen[0] = rng.choice(true_pool)
        hard.append(z3.Or(*[a for a, _ in chosen]))

    obj_c = [rng.randint(1, chi) for _ in range(n_vars)]
    objective = z3.Sum([c * x for c, x in zip(obj_c, xs)])
    return OMTInstance(
        instance_id=instance_id, variables=xs, hard=hard, objective=objective, sense=sense,
        obj_coeffs={n: float(c) for n, c in zip(names, obj_c)},
        theory="LIA", family="bool", description=f"bool LIA, {n_vars} vars {n_disj} disj (shared pool)",
    )


def generate_bool_lia_dataset(count: int, seed: int = 0, *, id_prefix: str = "blia",
                              min_vars: int = 5, max_vars: int = 7, **kwargs) -> list[OMTInstance]:
    """生成 ``count`` 个带布尔结构的有界整数 OMT 实例。"""
    rng = random.Random(seed)
    out: list[OMTInstance] = []
    for i in range(count):
        n_vars = rng.randint(min_vars, max_vars)
        out.append(generate_bool_lia_instance(f"{id_prefix}{i}", rng, n_vars=n_vars, **kwargs))
    return out


def generate_hard_bool_lia_dataset(count: int, seed: int = 0, *, id_prefix: str = "hblia",
                                   min_vars: int = 6, max_vars: int = 8, **kwargs) -> list[OMTInstance]:
    """更难的布尔结构整数 OMT：更多析取(n_disj=24)、更大子句(k=4)、更紧原子池(pool_mult=1)，
    使 z3 VSIDS 探索更多冲突 -> 给学习分支留 headroom。"""
    hard_defaults = dict(n_disj=24, k=4, ub=10, chi=5, pool_mult=1)
    hard_defaults.update(kwargs)
    rng = random.Random(seed)
    out: list[OMTInstance] = []
    for i in range(count):
        n_vars = rng.randint(min_vars, max_vars)
        out.append(generate_bool_lia_instance(f"{id_prefix}{i}", rng, n_vars=n_vars, **hard_defaults))
    return out


def generate_hard_lia_instance(instance_id: str, rng: random.Random, *,
                               n_vars: int = 6, n_constraints: int = 4, ub: int = 8,
                               coeff_lo: int = 1, coeff_hi: int = 5, slack: int = 1,
                               sense: Optional[Sense] = None) -> OMTInstance:
    """witness 驱动的 knapsack 型 LIA：紧 **packing** 约束（``Σa x ≤ b``）+ MAX 目标，
    整数-LP 间隙使 plain 模式 B&B 搜索非平凡，但 z3 ``Optimize`` 仍可秒级求解（可作 native
    参照与 strong-branching 教师）。

    只用 packing（不用 covering）：covering 会令 z3 ``Optimize`` 自身爆炸式变慢，无法生成
    strong-branching 标签；MAX + packing 是"分支有意义且 z3 可解"的可行区。
    """
    xs = [z3.Int(f"{instance_id}_x{j}") for j in range(n_vars)]
    names = [f"{instance_id}_x{j}" for j in range(n_vars)]
    witness = [rng.randint(0, ub) for _ in range(n_vars)]

    hard: list = []
    for x in xs:
        hard.append(x >= 0)
        hard.append(x <= ub)

    for _ in range(n_constraints):
        coeffs = [rng.randint(coeff_lo, coeff_hi) for _ in range(n_vars)]
        if all(c == 0 for c in coeffs):
            coeffs[rng.randrange(n_vars)] = 1
        lhs = z3.Sum([c * x for c, x in zip(coeffs, xs)])
        lhs_val = sum(c * w for c, w in zip(coeffs, witness))
        hard.append(lhs <= lhs_val + rng.randint(0, slack))     # packing：紧上界（witness 满足）

    obj_c = [rng.randint(coeff_lo, coeff_hi) for _ in range(n_vars)]
    objective = z3.Sum([c * x for c, x in zip(obj_c, xs)])
    # packing-only 下 MIN 平凡（最优≈0），故默认 MAX（knapsack）。
    if sense is None:
        sense = Sense.MAX

    return OMTInstance(
        instance_id=instance_id, variables=xs, hard=hard, objective=objective, sense=sense,
        obj_coeffs={n: float(c) for n, c in zip(names, obj_c)},
        theory="LIA", family="knapsack", description=f"knapsack LIA, {n_vars} vars {n_constraints} cons",
    )


def generate_hard_lia_dataset(count: int, seed: int = 0, *, id_prefix: str = "hlia",
                              min_vars: int = 5, max_vars: int = 7, **kwargs) -> list[OMTInstance]:
    """生成 ``count`` 个 knapsack LIA 实例（变量数在 [min_vars, max_vars] 间随机）。"""
    rng = random.Random(seed)
    out: list[OMTInstance] = []
    for i in range(count):
        n_vars = rng.randint(min_vars, max_vars)
        out.append(generate_hard_lia_instance(f"{id_prefix}{i}", rng, n_vars=n_vars, **kwargs))
    return out


# =========================================================================== #
# LRA 生成（带布尔结构的实数线性优化，参照 LRA_script.py）
# =========================================================================== #
Rat = Fraction
LRA_FAMILIES = ("piecewise", "nested", "imply", "mixed")


def _q(f: Rat) -> z3.ArithRef:
    """Fraction -> z3 有理常量。"""
    f = Fraction(f).limit_denominator(10_000)
    return z3.RealVal(f"{f.numerator}/{f.denominator}")


def _lin_value(coeffs: Sequence[Rat], witness: Sequence[Rat]) -> Rat:
    return sum((c * w for c, w in zip(coeffs, witness)), Rat(0))


def _lin_z3(coeffs: Sequence[Rat], xs: Sequence) -> z3.ArithRef:
    """构造 z3 线性表达式 Σ c_i x_i（跳过 0 系数）。"""
    terms = [_q(c) * x for c, x in zip(coeffs, xs) if c != 0]
    if not terms:
        return z3.RealVal(0)
    return z3.Sum(terms) if len(terms) > 1 else terms[0]


@dataclass
class _Atom:
    """线性比较原子，可在 witness 上求值，也可转成 z3 表达式。"""

    op: str                # ">=" | "<=" | "="
    coeffs: tuple
    bound: Rat

    def holds(self, witness: Sequence[Rat]) -> bool:
        v = _lin_value(self.coeffs, witness)
        if self.op == ">=":
            return v >= self.bound
        if self.op == "<=":
            return v <= self.bound
        return v == self.bound

    def to_z3(self, xs: Sequence) -> z3.BoolRef:
        lhs = _lin_z3(self.coeffs, xs)
        b = _q(self.bound)
        if self.op == ">=":
            return lhs >= b
        if self.op == "<=":
            return lhs <= b
        return lhs == b


@dataclass
class _Bool:
    """and/or/not/atom 布尔组合，可在 witness 上求值，也可转成 z3 表达式。"""

    kind: str              # "atom" | "not" | "and" | "or"
    atom: Optional[_Atom] = None
    children: tuple = ()

    def holds(self, witness: Sequence[Rat]) -> bool:
        if self.kind == "atom":
            return self.atom.holds(witness)
        if self.kind == "not":
            return not self.children[0].holds(witness)
        if self.kind == "and":
            return all(c.holds(witness) for c in self.children)
        return any(c.holds(witness) for c in self.children)  # or

    def to_z3(self, xs: Sequence) -> z3.BoolRef:
        if self.kind == "atom":
            return self.atom.to_z3(xs)
        if self.kind == "not":
            return z3.Not(self.children[0].to_z3(xs))
        kids = [c.to_z3(xs) for c in self.children]
        return z3.And(*kids) if self.kind == "and" else z3.Or(*kids)


def _rand_coeffs(rng: random.Random, n: int) -> tuple:
    c = tuple(Rat(rng.randint(-6, 6)) for _ in range(n))
    if all(v == 0 for v in c):
        c = tuple(Rat(1) if i == 0 else Rat(0) for i in range(n))
    return c


def _rand_atom(rng: random.Random, n: int, witness, *, force: Optional[bool] = None) -> _Atom:
    """随机原子；``force=True/False`` 时构造出在 witness 上必真/必假的原子。"""
    coeffs = _rand_coeffs(rng, n)
    v = _lin_value(coeffs, witness)
    op = rng.choice([">=", "<=", "="])
    slack = Rat(rng.randint(1, 8))
    if force is True:
        bound = v - slack if op == ">=" else (v + slack if op == "<=" else v)
    elif force is False:
        bound = v + slack if op == ">=" else (v - slack if op == "<=" else v + Rat(1, 2))
    else:
        bound = v + Rat(rng.randint(-5, 5), rng.randint(1, 4))
    return _Atom(op, coeffs, bound)


def _rand_bool(rng: random.Random, n: int, witness, depth: int, *, must_hold: bool) -> _Bool:
    """witness 驱动的随机布尔树；``must_hold`` 保证在 witness 上取该真值。"""
    if depth <= 0 or rng.random() < 0.35:
        return _Bool("atom", atom=_rand_atom(rng, n, witness, force=must_hold))
    op = rng.choice(["and", "or", "not"])
    if op == "not":
        return _Bool("not", children=(_rand_bool(rng, n, witness, depth - 1, must_hold=not must_hold),))
    arity = rng.randint(2, 3)
    if op == "and":
        # must_hold: 全部真；否则全部假（都不满足即可使 and 为假）。
        kids = tuple(_rand_bool(rng, n, witness, depth - 1, must_hold=must_hold) for _ in range(arity))
        return _Bool("and", children=kids)
    # or
    if must_hold:
        idx = rng.randrange(arity)
        kids = tuple(_rand_bool(rng, n, witness, depth - 1, must_hold=(i == idx)) for i in range(arity))
    else:
        kids = tuple(_rand_bool(rng, n, witness, depth - 1, must_hold=False) for _ in range(arity))
    return _Bool("or", children=kids)


def _implication(rng: random.Random, n: int, witness) -> _Bool:
    """构造 witness 满足的 (or (not P) Q)（即 P -> Q）。"""
    p = _rand_atom(rng, n, witness)
    q = _rand_atom(rng, n, witness, force=True if p.holds(witness) else None)
    return _Bool("or", children=(_Bool("not", children=(_Bool("atom", atom=p),)),
                                 _Bool("atom", atom=q)))


def _piecewise_or(rng: random.Random, n: int, witness, num_branches: int) -> _Bool:
    """分段析取：(or (and 区间_i 附加约束_i) ...)，witness 落在某一活跃区间。"""
    sel = _rand_coeffs(rng, n)
    center = _lin_value(sel, witness)
    width = Rat(rng.randint(2, 6))
    gaps = [Rat(rng.randint(1, 4)) for _ in range(num_branches - 1)]
    intervals: list[tuple[Rat, Rat]] = []
    lo = center - width * Rat(num_branches // 2 + 1)
    for i in range(num_branches):
        hi = lo + width
        intervals.append((lo, hi))
        lo = hi + (gaps[i] if i < len(gaps) else Rat(0))

    active = min(range(num_branches),
                 key=lambda i: abs(center - (intervals[i][0] + intervals[i][1]) / 2))
    lo_a, hi_a = intervals[active]
    if not (lo_a <= center <= hi_a):
        shift = center - (lo_a + hi_a) / 2
        intervals[active] = (lo_a + shift, hi_a + shift)

    branches: list[_Bool] = []
    for i, (lo_i, hi_i) in enumerate(intervals):
        kids = [_Bool("atom", atom=_Atom(">=", sel, lo_i)),
                _Bool("atom", atom=_Atom("<=", sel, hi_i))]
        if i == active:
            pin = rng.randrange(n)
            pin_c = tuple(Rat(1) if j == pin else Rat(0) for j in range(n))
            kids.append(_Bool("atom", atom=_Atom("=", pin_c, witness[pin])))
            if n > 1 and rng.random() < 0.7:
                combo = _rand_coeffs(rng, n)
                kids.append(_Bool("atom", atom=_Atom("=", combo, _lin_value(combo, witness))))
        else:
            for _ in range(rng.randint(0, 2)):
                kids.append(_Bool("atom", atom=_rand_atom(rng, n, witness)))
        branches.append(_Bool("and", children=tuple(kids)))
    return _Bool("or", children=tuple(branches))


def _halfspace(rng: random.Random, n: int, witness, op: str) -> _Atom:
    coeffs = _rand_coeffs(rng, n)
    v = _lin_value(coeffs, witness)
    slack = Rat(rng.randint(1, 20))
    return _Atom("<=", coeffs, v + slack) if op == "<=" else _Atom(">=", coeffs, v - slack)


def generate_lra_instance(instance_id: str, rng: random.Random, *,
                          n_vars: int = 4, family: str = "mixed",
                          num_branches: int = 4, bool_depth: int = 3,
                          num_implications: int = 3, num_halfspaces: int = 4,
                          sense: Optional[Sense] = None) -> OMTInstance:
    """生成单个带布尔结构的 LRA 实例（witness 驱动，高概率 SAT、目标有界）。"""
    xs = [z3.Real(f"{instance_id}_x{j}") for j in range(n_vars)]
    names = [f"{instance_id}_x{j}" for j in range(n_vars)]
    witness = [Rat(rng.randint(1, 12), rng.randint(1, 6)) for _ in range(n_vars)]

    hard: list = []
    # 非负 + 松上界盒约束（witness 驱动）：保证目标有界、可行域闭合。
    for i, (x, w) in enumerate(zip(xs, witness)):
        hard.append(x >= 0)
        hard.append(x <= _q(w + Rat(rng.randint(5, 30))))

    # 若干包含 witness 的线性半空间。
    for _ in range(num_halfspaces):
        atom = _halfspace(rng, n_vars, witness, rng.choice(["<=", ">="]))
        hard.append(atom.to_z3(xs))

    if family == "piecewise":
        hard.append(_piecewise_or(rng, n_vars, witness, num_branches).to_z3(xs))
        desc = f"piecewise OR-of-AND, {num_branches} 分段, {n_vars} vars"
    elif family == "nested":
        hard.append(_rand_bool(rng, n_vars, witness, bool_depth, must_hold=True).to_z3(xs))
        for _ in range(rng.randint(1, 2)):
            hard.append(_rand_bool(rng, n_vars, witness, max(1, bool_depth - 1),
                                   must_hold=True).to_z3(xs))
        desc = f"nested and/or/not depth~{bool_depth}, {n_vars} vars"
    elif family == "imply":
        for _ in range(num_implications):
            hard.append(_implication(rng, n_vars, witness).to_z3(xs))
        hard.append(_piecewise_or(rng, n_vars, witness, max(2, num_branches // 2)).to_z3(xs))
        desc = f"{num_implications} 蕴含 + 分段守卫, {n_vars} vars"
    elif family == "mixed":
        hard.append(_piecewise_or(rng, n_vars, witness, num_branches).to_z3(xs))
        hard.append(_rand_bool(rng, n_vars, witness, bool_depth, must_hold=True).to_z3(xs))
        for _ in range(num_implications):
            hard.append(_implication(rng, n_vars, witness).to_z3(xs))
        desc = f"mixed 分段+嵌套布尔+蕴含, {n_vars} vars"
    else:
        raise ValueError(f"未知 LRA family: {family}")

    obj_c = [rng.randint(-7, 7) for _ in range(n_vars)]
    if all(c == 0 for c in obj_c):
        obj_c[0] = 1
    objective = z3.Sum([c * x for c, x in zip(obj_c, xs)])
    if sense is None:
        sense = rng.choice([Sense.MIN, Sense.MAX])

    return OMTInstance(
        instance_id=instance_id, variables=xs, hard=hard,
        objective=objective, sense=sense,
        obj_coeffs={n: float(c) for n, c in zip(names, obj_c)},
        theory="LRA", family=family, description=desc,
    )


def generate_lra_dataset(count: int, seed: int = 0, *, id_prefix: str = "lra",
                         family: str = "random", min_vars: int = 3,
                         max_vars: int = 6, **kwargs) -> list[OMTInstance]:
    """生成 ``count`` 个 LRA 实例（``family="random"`` 时在各结构族间轮换）。"""
    rng = random.Random(seed)
    out: list[OMTInstance] = []
    for i in range(count):
        n_vars = rng.randint(min_vars, max_vars)
        fam = rng.choice(LRA_FAMILIES) if family == "random" else family
        out.append(generate_lra_instance(f"{id_prefix}{i}", rng, n_vars=n_vars,
                                          family=fam, **kwargs))
    return out


# =========================================================================== #
# 专家标签
# =========================================================================== #
def oracle_numeric_choice(instance: OMTInstance) -> Hashable:
    """启发式专家标签：选目标系数绝对值最大的数值变量（heuristic distillation）。

    用作 imitation 冷启动标签与评测“准确率”的参照专家。
    """
    if not instance.obj_coeffs:
        return None
    return max(instance.obj_coeffs, key=lambda n: abs(instance.obj_coeffs[n]))


def _validate(instance: OMTInstance) -> bool:
    """用 z3 检查硬约束是否可满足（生成正确性自检）。"""
    s = z3.Solver()
    s.add(*instance.hard)
    return s.check() == z3.sat


def _main() -> None:
    parser = argparse.ArgumentParser(description="生成合成 OMT(LIA/LRA) 实例并打印摘要")
    parser.add_argument("--theory", choices=["lia", "lra"], default="lia")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-vars", type=int, default=3, help="LIA 变量数")
    parser.add_argument("--n-constraints", type=int, default=3, help="LIA 约束数")
    parser.add_argument("--ub", type=int, default=8, help="LIA 变量上界")
    parser.add_argument("--min-vars", type=int, default=3, help="LRA 最小变量数")
    parser.add_argument("--max-vars", type=int, default=6, help="LRA 最大变量数")
    parser.add_argument("--family", default="random",
                        choices=[*LRA_FAMILIES, "random"], help="LRA 结构族")
    parser.add_argument("--validate", action="store_true", help="用 z3 校验可满足")
    args = parser.parse_args()

    if args.theory == "lia":
        ds = generate_dataset(args.count, seed=args.seed, n_vars=args.n_vars,
                              n_constraints=args.n_constraints, ub=args.ub)
    else:
        ds = generate_lra_dataset(args.count, seed=args.seed, family=args.family,
                                  min_vars=args.min_vars, max_vars=args.max_vars)

    print(f"生成 {len(ds)} 个 OMT({args.theory.upper()}) 实例 (seed={args.seed}):")
    for inst in ds:
        line = (f"  {inst.instance_id}: theory={inst.theory} family={inst.family} "
                f"vars={len(inst.variables)} hard={len(inst.hard)} "
                f"sense={inst.sense.value} oracle={oracle_numeric_choice(inst)}")
        if args.validate:
            line += f" sat={_validate(inst)}"
        print(line)


__all__ = [
    "OMTInstance",
    "generate_instance",
    "generate_dataset",
    "generate_hard_lia_instance",
    "generate_hard_lia_dataset",
    "generate_bool_lia_instance",
    "generate_bool_lia_dataset",
    "generate_hard_bool_lia_dataset",
    "generate_lra_instance",
    "generate_lra_dataset",
    "LRA_FAMILIES",
    "oracle_numeric_choice",
]


if __name__ == "__main__":
    _main()
