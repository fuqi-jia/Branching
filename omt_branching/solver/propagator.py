"""LearnedDecidePropagator：经 z3 UserPropagator 的 add_decide/next_split **接管 z3 内部
布尔分支决策**（不改 z3）。decider 不自信（返回 None）时直接放行 -> 退回 z3 原生 VSIDS。

注意：基类占用属性名 ``fixed``/``decide``，本类用 ``_val``/``_on_decide`` 等避让。
"""
from __future__ import annotations

from typing import Callable

import z3

from omt_branching.solver.propagator_snapshot import atom_key


class LearnedDecidePropagator(z3.UserPropagateBase):
    def __init__(self, s, atoms, decider: Callable):
        super().__init__(s)
        self.atoms = list(atoms)
        self.key2atom = {atom_key(a): a for a in self.atoms}
        # z3 每次回调都新建 t 的 Python 包装（id() 不稳定），但底层 AST 的 get_id() 稳定。
        # 注册原子被 self.atoms 钉住存活整个求解，其 get_id() 不会被回收复用，故可安全建表：
        # _on_fixed 里用 get_id() O(1) 命中，避免每次回调对原子做 str()（实测占总耗时 ~65%）。
        self._id2key = {a.get_id(): k for k, a in self.key2atom.items()}
        self.decider = decider
        self._val: dict = {}          # key -> bool（当前赋值）
        self._trail: list = []
        self._lim: list = []
        self.n_decisions = 0
        self.add_fixed(self._on_fixed)
        self.add_decide(self._on_decide)
        for a in self.atoms:
            self.add(a)               # 注册原子，z3 才会在其上回调 decide

    def push(self):
        self._lim.append(len(self._trail))

    def pop(self, num_scopes):
        for _ in range(num_scopes):
            lim = self._lim.pop()
            while len(self._trail) > lim:
                self._val.pop(self._trail.pop(), None)

    def fresh(self, new_ctx):
        return LearnedDecidePropagator(new_ctx, self.atoms, self.decider)

    def _on_fixed(self, t, v):
        # get_id() 命中已注册原子（z3 只对 add 过的项回调 fixed），避免 str(t)。
        k = self._id2key.get(t.get_id())
        if k is not None and k not in self._val:
            self._val[k] = z3.is_true(v)
            self._trail.append(k)

    def _on_decide(self, t, idx, phase):
        undecided = [k for k in self.key2atom if k not in self._val]
        if not undecided:
            return
        # self._val 只读传给 decider（各 decider 仅在 refocus 时即时读取、不留引用、不改），
        # 免去每次 decide 一次 dict 拷贝。
        choice = self.decider(undecided, self._val)
        if choice is None:
            return                    # 退回 VSIDS
        key, ph = choice
        atom = self.key2atom.get(key)
        if atom is None:
            return
        self.n_decisions += 1
        # z3 next_split 的 phase 是 Z3_lbool：真=1，假=-1，未定=0（≠ Python bool）
        z3_phase = z3.Z3_L_TRUE if ph else z3.Z3_L_FALSE
        self.next_split(atom, 0, z3_phase)


__all__ = ["LearnedDecidePropagator"]
