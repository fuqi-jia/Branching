# UserPropagator 学习分支 —— Phase 1（管道+测量）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 z3 UserPropagator 让（未训练的）GNN 策略接管 z3 内部布尔分支决策，在 OMT
Solver 回路里跑通并达到 `== native` 最优，且能测量 learned-decide vs VSIDS 的 rlimit/decisions。

**Architecture:** OMT = 单 z3.Solver 上的线性搜索回路（Solve + Better-cut，直到 UNSAT）；挂
`LearnedDecidePropagator` 接管内部 decide；GNN 经周期 refocus 产出原子优先级。**本 Phase 不训练**——
只证明管道正确 + 可测量。

**Tech Stack:** Python 3.10 / z3-solver 4.15.4 / PyTorch；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`（从 `Branching/`）。
- **不改 z3**；`z3.Optimize` 不支持 propagator，故 OMT 走 Solver 回路。
- docstring/注释中文；标识符/类型注解英文。
- 可行性锚点：`docs/ref/spike_userpropagator_decide.py`（add_decide/next_split 已验证）。
- 关键契约（复用）：`SolverSnapshot(bool_vars, clauses, theory_atoms, numeric_vars, objective, search_state, candidate_bool_ids, candidate_numeric_ids, snapshot_id)`；`BooleanVarInfo(var_id, assignment, is_candidate, ...)`；`ClauseInfo(clause_id, literals=list[(var_id, is_positive)], ...)`；`BranchingPolicyService(policy).advise(snapshot) -> BranchingAdvice(activity_priors: dict[var_id->float], phase_suggestions: dict[var_id->bool], use_gnn: bool, ...)`。
- z3 UserPropagator：`UserPropagateBase.__init__(self, s)`；`add(atom)`；`add_fixed(cb)`；`add_decide(cb)`；`next_split(t, idx, phase)`；须实现 `push`/`pop`/`fresh`。**避免属性名 `fixed`/`decide`（基类占用）。**

---

### Task 1: 布尔快照构建器（原子 + 子句共现图）

**Files:**
- Create: `omt_branching/solver/propagator_snapshot.py`
- Test: `tests/solver/test_propagator_snapshot.py`

**Interfaces:**
- Produces:
  - `atom_key(e) -> str`（z3 原子的稳定键，`= str(e)`）
  - `collect_atoms(assertions) -> list`（z3 比较原子 + 布尔常量，去重，保序）
  - `build_bool_snapshot(assertions, assignment=None, stats=None, snapshot_id="prop") -> tuple[SolverSnapshot, dict[str, object]]`（返回快照 + `key->z3atom` 映射）

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_propagator_snapshot.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator_snapshot import (
    atom_key, collect_atoms, build_bool_snapshot,
)


def test_collect_atoms_and_clause_cooccurrence():
    x = z3.Int("x")
    a, b, c = x >= 5, x <= 2, z3.Bool("c")
    asserts = [z3.Or(a, b), z3.Or(c, z3.Not(a))]
    atoms = collect_atoms(asserts)
    keys = {atom_key(t) for t in atoms}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= keys

    snap, amap = build_bool_snapshot(asserts)
    bkeys = {bv.var_id for bv in snap.bool_vars}
    assert {atom_key(a), atom_key(b), atom_key(c)} <= bkeys
    # 每个顶层 assertion 的原子共现为一个 clause
    assert len(snap.clauses) == 2
    lits0 = {vid for vid, _ in snap.clauses[0].literals}
    assert lits0 == {atom_key(a), atom_key(b)}
    # ¬a 的极性为 False
    pol = dict((vid, pos) for vid, pos in snap.clauses[1].literals)
    assert pol[atom_key(a)] is False
    # 映射能取回 z3 原子
    assert amap[atom_key(a)] is not None


def test_assignment_and_candidates():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    snap, _ = build_bool_snapshot([z3.Or(a, b)], assignment={atom_key(a): True})
    amap = {bv.var_id: bv for bv in snap.bool_vars}
    assert amap[atom_key(a)].assignment is True
    assert amap[atom_key(b)].assignment is None
    assert set(snap.candidate_bool_ids) == {atom_key(a), atom_key(b)}
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_propagator_snapshot.py -q`
Expected: FAIL（`ModuleNotFoundError: propagator_snapshot`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/propagator_snapshot.py
"""从 z3 布尔公式构造 SolverSnapshot（供 UserPropagator 学习分支）。

抽取原子（比较原子 + 布尔常量）与**子句共现图**（每个顶层 assertion 的原子构成一个 clause），
配合 propagator 提供的动态赋值/统计，喂给现有 GNN 策略。子句图是学习"好决策"的关键结构
（见 spec §5：无子句图极可能学不动）。
"""
from __future__ import annotations

from typing import Optional

import z3

from omt_branching.input.solver_state import (
    BooleanVarInfo, ClauseInfo, SearchStateInfo, SolverSnapshot,
)

_CMP = {z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT, z3.Z3_OP_EQ}


def atom_key(e) -> str:
    """z3 原子的稳定字符串键（同一进程内对同一原子稳定）。"""
    return str(e)


def _is_atom(e) -> bool:
    if not z3.is_bool(e):
        return False
    op = e.decl().kind()
    if op in _CMP and e.num_args() >= 1 and z3.is_arith(e.arg(0)):
        return True
    return z3.is_const(e) and op == z3.Z3_OP_UNINTERPRETED


def _lit(e):
    """返回 (atom_expr, is_positive)；``Not(a)`` -> (a, False)。"""
    if z3.is_not(e):
        return e.arg(0), False
    return e, True


def _walk_atoms(e, out, seen):
    eid = e.get_id()
    if eid in seen:
        return
    seen.add(eid)
    if _is_atom(e):
        out.append(e)
        return
    if not z3.is_bool(e):
        return
    for ch in e.children():
        _walk_atoms(ch, out, seen)


def collect_atoms(assertions) -> list:
    out: list = []
    seen: set = set()
    dedup: dict = {}
    for a in assertions:
        _walk_atoms(a, out, seen)
    # 按 atom_key 去重、保序
    uniq = []
    for t in out:
        k = atom_key(t)
        if k not in dedup:
            dedup[k] = t
            uniq.append(t)
    return uniq


def _clause_literals(assertion):
    """把一个顶层 assertion 拉平成 (atom_key, is_positive) 列表（Or 展开，其余取自身原子）。"""
    lits = []
    seen = set()

    def add(e):
        atom, pos = _lit(e)
        if _is_atom(atom):
            k = atom_key(atom)
            if k not in seen:
                seen.add(k)
                lits.append((k, pos))
        else:
            for ch in atom.children():
                add(ch)

    if z3.is_or(assertion):
        for ch in assertion.children():
            add(ch)
    else:
        add(assertion)
    return lits


def build_bool_snapshot(assertions, assignment: Optional[dict] = None,
                        stats: Optional[dict] = None, snapshot_id: str = "prop"):
    assignment = assignment or {}
    stats = stats or {}
    atoms = collect_atoms(assertions)
    amap = {atom_key(t): t for t in atoms}

    bool_vars = [
        BooleanVarInfo(var_id=k, assignment=assignment.get(k), is_candidate=True)
        for k in amap
    ]
    clauses = []
    for i, a in enumerate(assertions):
        lits = [(k, p) for (k, p) in _clause_literals(a) if k in amap]
        if lits:
            clauses.append(ClauseInfo(clause_id=f"c{i}", literals=lits))

    search_state = SearchStateInfo(
        decision_level=int(stats.get("decisions", 0)),
        conflict_count=int(stats.get("conflicts", 0)),
        trail_length=len(assignment),
    )
    snap = SolverSnapshot(
        bool_vars=bool_vars, clauses=clauses, theory_atoms=[], numeric_vars=[],
        search_state=search_state,
        candidate_bool_ids=list(amap.keys()), candidate_numeric_ids=[],
        snapshot_id=snapshot_id,
    )
    return snap, amap


__all__ = ["atom_key", "collect_atoms", "build_bool_snapshot"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_propagator_snapshot.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/propagator_snapshot.py tests/solver/test_propagator_snapshot.py
git commit -m "feat: propagator 布尔快照构建器（原子+子句共现图）"
```

---

### Task 2: LearnedDecidePropagator（接管 z3 内部决策）

**Files:**
- Create: `omt_branching/solver/propagator.py`
- Test: `tests/solver/test_propagator.py`

**Interfaces:**
- Consumes: `atom_key`（Task 1）。
- Produces: `LearnedDecidePropagator(s, atoms, decider)`，其中 `decider(undecided_keys: list[str], assignment: dict[str,bool]) -> Optional[tuple[str, bool]]`（返回 (选中原子键, phase) 或 None=退回 VSIDS）；属性 `.n_decisions`（我们强制的决策数）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_propagator.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import atom_key


def _sat_instance():
    xs = [z3.Bool(f"b{i}") for i in range(12)]
    clauses = [z3.Or(xs[i], z3.Not(xs[(i + 1) % 12]), xs[(i + 2) % 12]) for i in range(12)]
    return xs, clauses


def _solve(decider):
    xs, clauses = _sat_instance()
    s = z3.Solver()
    p = LearnedDecidePropagator(s, xs, decider)
    s.add(*clauses)
    return s.check(), p.n_decisions


def test_propagator_controls_decisions_and_preserves_correctness():
    idx = lambda k: int(k[1:])
    resA, nA = _solve(lambda und, asg: (min(und, key=idx), True))
    resB, nB = _solve(lambda und, asg: (max(und, key=idx), True))
    assert resA == resB == z3.sat        # 正确性不变
    assert nA > 0 and nB > 0             # 两个 decider 都真的强制了决策


def test_none_decider_falls_back():
    resN, nN = _solve(lambda und, asg: None)   # 永远 None = 退回 VSIDS
    assert resN == z3.sat
    assert nN == 0                        # 我们没强制任何决策
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_propagator.py -q`
Expected: FAIL（`ModuleNotFoundError: propagator`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/propagator.py
"""LearnedDecidePropagator：经 z3 UserPropagator 的 add_decide/next_split **接管 z3 内部
布尔分支决策**（不改 z3）。decider 不自信（返回 None）时直接放行 -> 退回 z3 原生 VSIDS。

注意：基类占用属性名 ``fixed``/``decide``，本类用 ``_val``/``_decide`` 等避让。
"""
from __future__ import annotations

from typing import Callable, Optional

import z3

from omt_branching.solver.propagator_snapshot import atom_key


class LearnedDecidePropagator(z3.UserPropagateBase):
    def __init__(self, s, atoms, decider: Callable):
        super().__init__(s)
        self.atoms = list(atoms)
        self.key2atom = {atom_key(a): a for a in self.atoms}
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
        k = atom_key(t)
        if k not in self._val:
            self._val[k] = z3.is_true(v)
            self._trail.append(k)

    def _on_decide(self, t, idx, phase):
        undecided = [k for k in self.key2atom if k not in self._val]
        if not undecided:
            return
        choice = self.decider(undecided, dict(self._val))
        if choice is None:
            return                    # 退回 VSIDS
        key, ph = choice
        atom = self.key2atom.get(key)
        if atom is None:
            return
        self.n_decisions += 1
        self.next_split(atom, 0, 1 if ph else 0)   # phase: 1=真, 0=假（见 spike 验证）


__all__ = ["LearnedDecidePropagator"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_propagator.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/propagator.py tests/solver/test_propagator.py
git commit -m "feat: LearnedDecidePropagator（接管 z3 内部布尔决策，None 退回 VSIDS）"
```

---

### Task 3: GNN 优先级 decider（周期 refocus）

**Files:**
- Create: `omt_branching/solver/policy_decider.py`
- Test: `tests/solver/test_policy_decider.py`

**Interfaces:**
- Consumes: `build_bool_snapshot`, `atom_key`（Task 1）；`BranchingPolicyService.advise`。
- Produces: `PolicyDecider(service, assertions, refocus_every=200)`，可调用 `__call__(undecided_keys, assignment) -> Optional[tuple[str,bool]]`（供 propagator 用）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_policy_decider.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver.policy_decider import PolicyDecider
from omt_branching.solver.propagator_snapshot import atom_key


def test_policy_decider_returns_valid_or_fallback():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=100)
    und = [atom_key(a), atom_key(b)]
    choice = dec(und, {})
    # 要么退回(None)，要么返回一个合法的未定原子键 + bool 相位
    assert choice is None or (choice[0] in und and isinstance(choice[1], bool))


def test_refocus_cadence():
    x = z3.Int("x")
    asserts = [x >= 0, x <= 10, z3.Or(x >= 5, x <= 2)]
    svc = BranchingPolicyService(policy=BranchingPolicy())
    dec = PolicyDecider(svc, asserts, refocus_every=3)
    calls = {"n": 0}
    orig = dec._refocus
    def counting(asg):
        calls["n"] += 1
        return orig(asg)
    dec._refocus = counting
    for _ in range(7):
        dec([atom_key(x >= 5)], {})
    assert calls["n"] == 3     # 第1次 + 每满3次一次（7 次调用 -> refocus 于 1,4,7）
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_policy_decider.py -q`
Expected: FAIL（`ModuleNotFoundError: policy_decider`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/policy_decider.py
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
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_policy_decider.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/policy_decider.py tests/solver/test_policy_decider.py
git commit -m "feat: PolicyDecider（GNN 优先级 + 周期 refocus，喂给 propagator）"
```

---

### Task 4: OMT-over-Solver 回路（挂 propagator）+ 指标

**Files:**
- Create: `omt_branching/solver/decide_omt.py`
- Test: `tests/solver/test_decide_omt.py`

**Interfaces:**
- Consumes: `LearnedDecidePropagator`（Task 2）、`collect_atoms`（Task 1）、`solve_native`（现有）。
- Produces: `solve_omt_with_decider(hard, objective, sense, decider_factory=None, max_iters=100000) -> dict{value, rlimit, conflicts, decisions, iters}`。`decider_factory=None` = VSIDS 臂（不挂 propagator）；否则 `decider_factory(assertions) -> decider`。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_decide_omt.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import generate_hard_lia_dataset, solve_native
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.policy_decider import PolicyDecider


def test_vsids_arm_matches_native():
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    r = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
    assert r["value"] == solve_native(hard, obj, sense)
    assert r["decisions"] is None       # VSIDS 臂不挂 propagator


def test_learned_arm_matches_native_and_fires():
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    svc = BranchingPolicyService(policy=BranchingPolicy())
    r = solve_omt_with_decider(
        hard, obj, sense,
        decider_factory=lambda a: PolicyDecider(svc, a, refocus_every=50))
    assert r["value"] == solve_native(hard, obj, sense)   # 正确性：== native
    assert r["decisions"] is not None                     # propagator 在回路里生效
    assert r["rlimit"] > 0
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_decide_omt.py -q`
Expected: FAIL（`ModuleNotFoundError: decide_omt`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/decide_omt.py
"""OMT = 单 z3.Solver 线性搜索回路（Solve + Better-cut，直到 UNSAT），可挂
LearnedDecidePropagator 接管内部布尔决策。z3.Optimize 不支持 propagator，故必须走此回路。

三臂对比：decider_factory=None -> VSIDS 臂；给 PolicyDecider -> learned 臂；native 见 solve_native。
"""
from __future__ import annotations

from fractions import Fraction

import z3

from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.propagator_snapshot import collect_atoms


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


def solve_omt_with_decider(hard, objective, sense: Sense,
                           decider_factory=None, max_iters: int = 100000) -> dict:
    s = z3.Solver()
    prop = None
    if decider_factory is not None:
        atoms = collect_atoms(list(hard))
        decider = decider_factory(list(hard))
        prop = LearnedDecidePropagator(s, atoms, decider)
    s.add(*hard)

    if s.check() != z3.sat:
        raise ValueError("硬约束不可满足")
    best_val = objective  # placeholder
    m = s.model()
    best_val = m.eval(objective, model_completion=True)

    iters = 0
    for iters in range(1, max_iters + 1):
        cut = objective > best_val if sense is Sense.MAX else objective < best_val
        s.add(cut)
        if s.check() != z3.sat:
            break
        m = s.model()
        best_val = m.eval(objective, model_completion=True)

    return {
        "value": _num(best_val),
        "rlimit": _stat(s, "rlimit count"),
        "conflicts": _stat(s, "conflicts"),
        "decisions": (prop.n_decisions if prop is not None else None),
        "iters": iters,
    }


__all__ = ["solve_omt_with_decider"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_decide_omt.py -q`
Expected: PASS（2 passed；learned 臂与 VSIDS 臂均 == native）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/decide_omt.py tests/solver/test_decide_omt.py
git commit -m "feat: OMT-over-Solver 回路挂 propagator（learned/VSIDS 臂均 == native）"
```

---

### Task 5: 三臂对比实验 + 导出 + 冒烟

**Files:**
- Create: `examples/decide_branch.py`
- Modify: `omt_branching/solver/__init__.py`
- Test: `tests/solver/test_demo_smoke.py`（追加）

**Interfaces:**
- Consumes: Task 1-4 全部。
- Produces: `examples/decide_branch.py`（main 打印 native/VSIDS/learned 的 rlimit/conflicts/decisions/match 对比）；`solver/__init__.py` 导出 `solve_omt_with_decider`, `PolicyDecider`, `LearnedDecidePropagator`, `build_bool_snapshot`。

- [ ] **Step 1: 更新导出**

在 `omt_branching/solver/__init__.py` 增加 import 与 `__all__` 条目：
```python
from omt_branching.solver.propagator import LearnedDecidePropagator
from omt_branching.solver.policy_decider import PolicyDecider
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.propagator_snapshot import build_bool_snapshot, collect_atoms, atom_key
```
`__all__` 追加：`"LearnedDecidePropagator"`, `"PolicyDecider"`, `"solve_omt_with_decider"`, `"build_bool_snapshot"`, `"collect_atoms"`, `"atom_key"`。

- [ ] **Step 2: 写实验脚本**

```python
# examples/decide_branch.py
"""三臂对比：native z3 Optimize / VSIDS-decide / learned-decide（未训练 GNN）。

证明管道正确（learned 臂 == native）并测量 rlimit/conflicts/decisions。**本 Phase 不训练**，
故 learned 未必优于 VSIDS——目的是管道 + 可测量，为 Phase 2（look-ahead imitation + RL）铺路。
"""
from __future__ import annotations

import argparse

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import (
    Z3Backend, generate_hard_lia_dataset, solve_native, solve_omt_with_decider,
)
from omt_branching.solver.policy_decider import PolicyDecider


def _native_cost(hard, obj, sense):
    b = Z3Backend()
    b.optimize(b.conjoin(*hard), obj, sense)
    return {"rlimit": b.rlimit_count}


def main() -> None:
    ap = argparse.ArgumentParser(description="UserPropagator 学习分支三臂对比")
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--min-vars", type=int, default=4)
    ap.add_argument("--max-vars", type=int, default=5)
    ap.add_argument("--refocus", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(0)
    insts = generate_hard_lia_dataset(args.test, seed=99,
                                      min_vars=args.min_vars, max_vars=args.max_vars)
    svc = BranchingPolicyService(policy=BranchingPolicy())

    agg = {"native": {"rlimit": 0.0}, "vsids": {"rlimit": 0.0, "conflicts": 0.0, "match": 0.0},
           "learned": {"rlimit": 0.0, "conflicts": 0.0, "decisions": 0.0, "match": 0.0}}
    for inst in insts:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)
        agg["native"]["rlimit"] += _native_cost(hard, obj, sense)["rlimit"]
        v = solve_omt_with_decider(hard, obj, sense, decider_factory=None)
        agg["vsids"]["rlimit"] += v["rlimit"]; agg["vsids"]["conflicts"] += v["conflicts"]
        agg["vsids"]["match"] += 1.0 if v["value"] == native else 0.0
        ln = solve_omt_with_decider(hard, obj, sense,
                                    decider_factory=lambda a: PolicyDecider(svc, a, args.refocus))
        agg["learned"]["rlimit"] += ln["rlimit"]; agg["learned"]["conflicts"] += ln["conflicts"]
        agg["learned"]["decisions"] += ln["decisions"]; agg["learned"]["match"] += 1.0 if ln["value"] == native else 0.0

    n = max(1, len(insts))
    print(f"=== 三臂对比（{len(insts)} 实例，未训练 GNN；rlimit/conflicts 越小越好，match=1 为正确）===")
    print(f"  native(z3 Optimize): rlimit={agg['native']['rlimit']/n:.0f}")
    print(f"  VSIDS-decide       : rlimit={agg['vsids']['rlimit']/n:.0f} "
          f"conflicts={agg['vsids']['conflicts']/n:.1f} match={agg['vsids']['match']/n:.2f}")
    print(f"  learned-decide     : rlimit={agg['learned']['rlimit']/n:.0f} "
          f"conflicts={agg['learned']['conflicts']/n:.1f} decisions={agg['learned']['decisions']/n:.1f} "
          f"match={agg['learned']['match']/n:.2f}")
    print("\nPhase 1 目标：learned 臂 match=1（管道正确）+ 可测量。Phase 2 再训练使其优于 VSIDS。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 追加冒烟测试**

在 `tests/solver/test_demo_smoke.py` 追加：
```python
def test_decide_branch_smoke():
    out = subprocess.run(
        [sys.executable, "-m", "examples.decide_branch",
         "--test", "5", "--min-vars", "4", "--max-vars", "4"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert "三臂对比" in out.stdout
```

- [ ] **Step 4: 运行冒烟 + 全量**

Run: `conda run -n omt python -m examples.decide_branch --test 5 --min-vars 4 --max-vars 4`
Expected: 打印三臂对比；**learned match=1.00**（管道正确）；decisions>0（propagator 生效）。

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add examples/decide_branch.py omt_branching/solver/__init__.py tests/solver/test_demo_smoke.py
git commit -m "feat: 三臂对比实验(native/VSIDS/learned) + 导出 + 冒烟（Phase 1 管道打通）"
```

---

## Self-Review

**Spec coverage（Phase 1 部分）：**
- §3.1 propagator → Task 2 ✅；§3.2 快照(含子句图) → Task 1 ✅；§3.3 优先级策略 → Task 3 ✅；
  §2 OMT-over-Solver 回路 → Task 4 ✅；§3.7 实验+指标 → Task 5 ✅；§6 Phase 1 里程碑(==native+可测量) → Task 4/5 ✅。
- Phase 2（look-ahead 教师 / imitation / RL，spec §3.4-3.6）**不在本 plan**——待 Phase 1 出数后另写 plan。

**Placeholder scan：** 无 TBD/TODO；每步含完整代码。

**Type consistency：** `atom_key`/`collect_atoms`/`build_bool_snapshot` 在 Task 1 定义、Task 2/3/4 一致调用；`LearnedDecidePropagator(s, atoms, decider)` 与 `decider(undecided_keys, assignment)->Optional[(key,bool)]` 在 Task 2 定义、Task 3 `PolicyDecider.__call__` 与 Task 4 一致；`solve_omt_with_decider(hard, obj, sense, decider_factory, max_iters)` 返回 dict 键在 Task 4/5 一致。

**风险提示（供执行时注意）：**
- 若 Task 4 learned 臂 match≠1：检查 propagator push/pop 是否平衡（`num_scopes()`）、`next_split` 相位取值（spike 用 1=真）、以及 decider 是否在 `use_gnn=False` 时正确返回 None。
- 若 knapsack LIA 布尔原子太少导致 decisions≈0：Phase 2 换**含布尔结构**（析取）的有界整数实例做真正的分支质量研究（本 Phase 管道验证不依赖它）。
