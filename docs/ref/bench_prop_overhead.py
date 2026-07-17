"""诊断：把"挂 UserPropagator 变慢"归因到具体来源（关预处理 / Python 回调 / 原子注册）。

方法：同一 OMT 实例在多种 propagator 配置下各跑一遍线性搜索回路，**同时**记 wall-time 与
z3 自报的 rlimit/conflicts/decisions。若耗时暴涨而 rlimit 不变 → Python 开销；若 rlimit 同步
暴涨 → 才是 z3 少做预处理 / 多做搜索。从仓库根目录运行：
    python docs/ref/bench_prop_overhead.py

结论（见 docs/findings-propagator-overhead.md）：主因是 `_on_fixed` 每次回调对原子 `str()`
（`atom_key` 的 id 缓存因 z3 每次新建包装而恒 miss），占总耗时约 65%；预处理**不是**慢因。
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import z3

# 允许从任意目录运行：把仓库根（本文件的上上级）加入 import 路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omt_branching.solver.instance_gen import generate_bool_lia_instance
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import atom_key, prepare_propagator_formula


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


def _num(ref):
    if z3.is_int_value(ref):
        return ref.as_long()
    if z3.is_rational_value(ref):
        return Fraction(ref.numerator_as_long(), ref.denominator_as_long())
    return Fraction(str(ref))


class BaseNoop(z3.UserPropagateBase):
    """空操作 propagator；注册原子/是否 add_fixed/回调体可控，decide 立即返回（退回 VSIDS）。

    ``fixed_mode``: "off" 不加 add_fixed；"strkey" 现状（atom_key=str）；
    "idkey" 用 get_id() 建表查（避免 str）；"empty" 仅计数（隔离 z3 marshalling）。
    """

    def __init__(self, s, atoms, *, fixed_mode: str = "off"):
        super().__init__(s)
        self.atoms = list(atoms)
        self.fixed_mode = fixed_mode
        self.n_fixed = 0
        self.n_decide = 0
        self._val: dict = {}
        self._trail: list = []
        self._lim: list = []
        self._fixed_ids: set = set()
        self._id2key = {a.get_id(): atom_key(a) for a in self.atoms}
        if fixed_mode != "off":
            self.add_fixed(self._on_fixed)
        self.add_decide(self._on_decide)
        for a in self.atoms:
            self.add(a)

    def push(self):
        self._lim.append(len(self._trail))

    def pop(self, num_scopes):
        for _ in range(num_scopes):
            lim = self._lim.pop()
            while len(self._trail) > lim:
                self._val.pop(self._trail.pop(), None)

    def fresh(self, new_ctx):
        return BaseNoop(new_ctx, self.atoms, fixed_mode=self.fixed_mode)

    def _on_fixed(self, t, v):
        self.n_fixed += 1
        if self.fixed_mode == "empty":
            return
        if self.fixed_mode == "idkey":
            k = self._id2key.get(t.get_id())
        else:  # strkey：现状
            self._fixed_ids.add(id(t))
            k = atom_key(t)
        if k is not None and k not in self._val:
            self._val[k] = z3.is_true(v)
            self._trail.append(k)

    def _on_decide(self, t, idx, phase):
        self.n_decide += 1
        return


def run(hard, obj, sense, *, arm, atoms=None, max_iters=100000):
    ctx = z3.Context()
    hard_iso = [h.translate(ctx) for h in hard]
    obj_iso = obj.translate(ctx)
    s = z3.Solver(ctx=ctx)
    prop = None
    if arm == "attach0":
        prop = BaseNoop(s, [], fixed_mode="off")
    elif arm in ("reg_nofix", "fix_empty", "fix_idkey", "full_noop"):
        atoms_iso = [a.translate(ctx) for a in atoms]
        mode = {
            "reg_nofix": "off",
            "fix_empty": "empty",
            "fix_idkey": "idkey",
            "full_noop": "strkey",
        }[arm]
        prop = BaseNoop(s, atoms_iso, fixed_mode=mode)

    t0 = perf_counter()
    s.add(*hard_iso)
    if s.check() != z3.sat:
        raise ValueError("硬约束不可满足")
    m = s.model()
    best = m.eval(obj_iso, model_completion=True)
    iters = 0
    for iters in range(1, max_iters + 1):
        cut = obj_iso > best if sense is Sense.MAX else obj_iso < best
        s.add(cut)
        if s.check() != z3.sat:
            break
        m = s.model()
        best = m.eval(obj_iso, model_completion=True)
    dt = perf_counter() - t0
    return {
        "arm": arm,
        "value": _num(best),
        "time_s": dt,
        "rlimit": _stat(s, "rlimit count"),
        "conflicts": _stat(s, "conflicts"),
        "decisions": _stat(s, "decisions"),
        "n_fixed": (prop.n_fixed if prop else 0),
        "distinct_fixed_ids": (len(prop._fixed_ids) if prop else 0),
    }


def main():
    rng = random.Random(7)
    inst = generate_bool_lia_instance(
        "bench", rng, n_vars=12, n_disj=90, k=5, ub=20, chi=8, pool_mult=1,
        sense=Sense.MAX,
    )
    hard, obj, sense = inst.as_tuple()
    pp, reg_atoms = prepare_propagator_formula(hard)
    print(f"实例：{len(inst.variables)} vars, {len(hard)} hard, "
          f"预处理后 {len(pp)} 断言, 注册原子 {len(reg_atoms)}")

    arms = [
        ("none_raw", hard, None),
        ("none_pp", pp, None),
        ("attach0", hard, None),
        ("reg_nofix", pp, reg_atoms),
        ("fix_empty", pp, reg_atoms),
        ("fix_idkey", pp, reg_atoms),
        ("full_noop", pp, reg_atoms),
    ]
    rows = []
    for arm, hh, at in arms:
        r = run(hh, obj, sense, arm=arm, atoms=at)
        rows.append(r)
        print(f"[{arm:10s}] t={r['time_s']:7.3f}s  rlimit={r['rlimit']:>10}  "
              f"confl={r['conflicts']:>8}  decis={r['decisions']:>8}  "
              f"n_fixed={r['n_fixed']:>9}  distinct_ids={r['distinct_fixed_ids']:>8}  "
              f"val={r['value']}")

    base = rows[0]["time_s"]
    print("\n相对基线 none_raw 的倍数：")
    for r in rows:
        rl_ratio = r["rlimit"] / max(1, rows[0]["rlimit"])
        print(f"  {r['arm']:10s}  time×{r['time_s']/base:6.2f}   rlimit×{rl_ratio:6.2f}")


if __name__ == "__main__":
    main()
