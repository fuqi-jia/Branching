"""PolicyDecider：把 GNN 策略包成 propagator 的 decide 函数（周期 refocus）。

每 ``refocus_every`` 次决策重算一次原子优先级（调用 GNN，代价可控），decide 时 O(1) 取
最高优先级的未定原子。策略 ``use_gnn=False`` 时返回 None -> propagator 退回 VSIDS。
"""
from __future__ import annotations

from typing import Optional

from omt_branching.service import BranchingPolicyService
from omt_branching.solver.propagator_snapshot import build_bool_snapshot


class PolicyDecider:
    def __init__(self, service: BranchingPolicyService, assertions,
                 refocus_every: int = 200):
        self.service = service
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self._pri: Optional[dict] = None
        self._phase: dict = {}
        self._since = self.refocus_every   # 首次即 refocus

    def _refocus(self, assignment):
        snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
        try:
            advice = self.service.advise(snap)
        except Exception:
            self._pri = None
            return
        if not advice.use_gnn:
            self._pri = None
            return
        self._pri = dict(advice.activity_priors)
        self._phase = dict(advice.phase_suggestions)

    def __call__(self, undecided_keys, assignment) -> Optional[tuple]:
        if self._since >= self.refocus_every:
            self._refocus(assignment)
            self._since = 0
        self._since += 1
        if not self._pri:
            return None
        cand = [k for k in undecided_keys if k in self._pri]
        if not cand:
            return None
        best = max(cand, key=lambda k: self._pri[k])
        return best, bool(self._phase.get(best, True))


__all__ = ["PolicyDecider"]
