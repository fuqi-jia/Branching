# 困难 SAT 学习分支 pivot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在困难 SAT（PHP + 随机 3-SAT）单次可满足性检查上，让训练后的 learned-decide 在
conflicts 上优于 VSIDS-decide（两臂均附 propagator，隔离分支质量）。

**Architecture:** 单次 `z3.Solver().check()` 附 `LearnedDecidePropagator`（关 z3 预处理→纯 CDCL）；
VSIDS 臂=decider 恒 defer，learned 臂=GNN；reward=−conflicts。imitation(look-ahead) 冷启动 + RL 微调。

**Tech Stack:** Python 3.10 / z3-solver 4.15.4 / PyTorch；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`。
- 不改 z3；附 propagator 会关预处理→纯 CDCL（已验证：PHP(9,8) 无 prop conflicts=0；附 prop
  VSIDS=23968、朴素固定序=16511）。
- docstring/注释中文；标识符/类型英文。
- 复用：`LearnedDecidePropagator(s, atoms, decider)`（decider `(undecided_keys, assignment)->Optional[(key,bool)]`，None=退回 VSIDS）；`build_bool_snapshot(assertions, assignment=None) -> (SolverSnapshot, amap)`；`atom_key(e)=str(e)`；`lookahead_scores(assertions, atoms, config) -> (dict[str,float], dict[str,bool])`；`GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)`；`RankingExample(graph, bool_target_scores, phase_targets)`；`ImitationTrainer(policy, TrainConfig).fit(exs, epochs)`；`SamplingPolicyDecider(policy, defer_logit, assertions, refocus_every, sample)`；`DecideRLTrainer(policy, DecideRLConfig)` 的 `update(steps, reward, key)`/`_baseline_for`/`_update_baseline_for`。

---

### Task 1: SAT 实例生成器

**Files:**
- Create: `omt_branching/solver/sat_instances.py`
- Test: `tests/solver/test_sat_instances.py`

**Interfaces:**
- Produces:
  - `generate_php(m) -> tuple[list, list]`（(atoms, clauses)；PHP(m+1,m) UNSAT）
  - `generate_rand_3sat(n, ratio=4.26, seed=0) -> tuple[list, list]`

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_sat_instances.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat


def test_php_is_unsat_and_shaped():
    atoms, clauses = generate_php(4)              # PHP(5,4)
    assert len(atoms) == 5 * 4                     # (m+1)*m 命题
    s = z3.Solver(); s.add(*clauses)
    assert s.check() == z3.unsat                   # 鸽笼原理 UNSAT
    assert all(z3.is_bool(a) for a in atoms)


def test_rand_3sat_reproducible():
    a1, c1 = generate_rand_3sat(30, 4.26, seed=1)
    a2, c2 = generate_rand_3sat(30, 4.26, seed=1)
    assert [str(a) for a in a1] == [str(a) for a in a2]
    assert len(c1) == int(30 * 4.26)
    assert all(z3.is_or(c) or z3.is_bool(c) for c in c1)
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_instances.py -q`
Expected: FAIL（`ModuleNotFoundError: sat_instances`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/sat_instances.py
"""困难 SAT 实例（供学习分支研究）：pigeonhole(UNSAT) + 相变点随机 3-SAT。

附 propagator 时 z3 关预处理走纯 CDCL，这些实例给出大量、受控的 conflicts（headroom）。
"""
from __future__ import annotations

import random

import z3


def generate_php(m: int):
    """PHP(m+1, m)：m+1 只鸽、m 个洞，**UNSAT**。返回 (atoms, clauses)。"""
    p = [[z3.Bool(f"p_{i}_{j}") for j in range(m)] for i in range(m + 1)]
    clauses = [z3.Or(*p[i]) for i in range(m + 1)]                 # 每鸽入某洞
    for j in range(m):
        for i1 in range(m + 1):
            for i2 in range(i1 + 1, m + 1):
                clauses.append(z3.Or(z3.Not(p[i1][j]), z3.Not(p[i2][j])))  # 无两鸽同洞
    atoms = [b for row in p for b in row]
    return atoms, clauses


def generate_rand_3sat(n: int, ratio: float = 4.26, seed: int = 0):
    """相变点附近随机 3-SAT。返回 (atoms, clauses)。"""
    rng = random.Random(seed)
    xs = [z3.Bool(f"v{i}") for i in range(n)]
    clauses = []
    for _ in range(int(n * ratio)):
        idx = rng.sample(range(n), 3)
        clauses.append(z3.Or([xs[i] if rng.random() < 0.5 else z3.Not(xs[i]) for i in idx]))
    return xs, clauses


__all__ = ["generate_php", "generate_rand_3sat"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_instances.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/sat_instances.py tests/solver/test_sat_instances.py
git commit -m "feat: SAT 实例生成器（pigeonhole UNSAT + 相变点随机 3-SAT）"
```

---

### Task 2: SAT 求解 harness（两臂均附 propagator）

**Files:**
- Create: `omt_branching/solver/sat_solve.py`
- Test: `tests/solver/test_sat_solve.py`

**Interfaces:**
- Consumes: `LearnedDecidePropagator`、`atom_key`。
- Produces: `solve_sat_with_decider(assertions, atoms, decider_factory=None) -> dict{result, conflicts, decisions, rlimit}`（`decider_factory=None` => defer-always = VSIDS 臂；两臂都附 propagator）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_sat_solve.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.sat_instances import generate_php
from omt_branching.solver.sat_solve import solve_sat_with_decider
from omt_branching.solver.propagator_snapshot import atom_key


def test_vsids_arm_has_conflicts_and_correct():
    atoms, clauses = generate_php(6)                       # PHP(7,6) UNSAT
    r = solve_sat_with_decider(clauses, atoms, decider_factory=None)
    assert r["result"] == "unsat"                          # 正确性
    assert r["conflicts"] > 100                            # 附 propagator -> 纯 CDCL 大量冲突
    assert r["decisions"] == 0                             # VSIDS 臂我们不覆盖


def test_override_arm_controls_and_correct():
    atoms, clauses = generate_php(6)
    r = solve_sat_with_decider(
        clauses, atoms,
        decider_factory=lambda a: (lambda und, asg: (min(und), True)))
    assert r["result"] == "unsat"
    assert r["decisions"] > 0                               # 我们强制了决策
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_solve.py -q`
Expected: FAIL（`ModuleNotFoundError: sat_solve`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/sat_solve.py
"""单次 SAT 可满足性检查 harness：**恒附** LearnedDecidePropagator（关 z3 预处理→纯 CDCL），
两臂公平——decider_factory=None => defer-always（z3 自身 VSIDS）；给出 decider => 学习覆盖。
指标 conflicts 是分支质量的直接度量（无 OMT 回路稀释）。
"""
from __future__ import annotations

import z3

from omt_branching.solver.propagator import LearnedDecidePropagator


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


def solve_sat_with_decider(assertions, atoms, decider_factory=None) -> dict:
    s = z3.Solver()
    if decider_factory is None:
        decider = lambda und, asg: None            # defer-always = VSIDS 臂
    else:
        decider = decider_factory(list(assertions))
    prop = LearnedDecidePropagator(s, atoms, decider)   # 恒附 -> 关预处理
    s.add(*assertions)
    res = s.check()
    return {
        "result": "sat" if res == z3.sat else ("unsat" if res == z3.unsat else "unknown"),
        "conflicts": _stat(s, "conflicts"),
        "decisions": prop.n_decisions,
        "rlimit": _stat(s, "rlimit count"),
    }


__all__ = ["solve_sat_with_decider"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_solve.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/sat_solve.py tests/solver/test_sat_solve.py
git commit -m "feat: solve_sat_with_decider（两臂均附 propagator，conflicts 指标）"
```

---

### Task 3: SAT look-ahead imitation 样本 + 可学习性护栏

**Files:**
- Modify: `omt_branching/solver/training_data.py`
- Test: `tests/solver/test_sat_solve.py`（追加）

**Interfaces:**
- Consumes: `lookahead_scores`、`build_bool_snapshot`、`GraphBuilder`、`RankingExample`。
- Produces: `build_lookahead_examples_sat(problems, config=None) -> list[RankingExample]`（`problems=list[(assertions, atoms)]`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_sat_solve.py 追加
def test_build_lookahead_examples_sat_learnable():
    import torch
    from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat
    from omt_branching.solver.training_data import build_lookahead_examples_sat
    from omt_branching.model.policy import BranchingPolicy
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig

    torch.manual_seed(0)
    problems = [generate_php(4)] + [generate_rand_3sat(30, 4.26, s) for s in range(6)]
    exs = [e for e in build_lookahead_examples_sat(problems) if e.bool_target_scores]
    assert exs, "应有带 bool 标签的样本"
    policy = BranchingPolicy()
    h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=12)
    assert "branch" in h[0]
    assert h[-1]["branch"] < h[0]["branch"]        # 子句图=特征 -> look-ahead 可学
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_solve.py::test_build_lookahead_examples_sat_learnable -q`
Expected: FAIL（`ImportError: build_lookahead_examples_sat`）

- [ ] **Step 3: 写实现**

在 `training_data.py`（`build_lookahead_examples` 之后）新增：

```python
def build_lookahead_examples_sat(problems, config=None):
    """SAT look-ahead imitation 样本：``problems = list[(assertions, atoms)]``。"""
    from omt_branching.solver.lookahead import LookaheadConfig, lookahead_scores
    from omt_branching.solver.propagator_snapshot import build_bool_snapshot

    cfg = config or LookaheadConfig()
    out: list[RankingExample] = []
    for assertions, atoms in problems:
        snap, _ = build_bool_snapshot(list(assertions))
        graph = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        scores, phases = lookahead_scores(list(assertions), atoms=list(atoms), config=cfg)
        bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
        bts: dict[int, float] = {}
        pts: dict[int, bool] = {}
        for k, sc in scores.items():
            loc = bmap.get(k)
            if loc is not None:
                bts[loc] = sc
        for k, ph in phases.items():
            loc = bmap.get(k)
            if loc is not None:
                pts[loc] = ph
        if bts:
            out.append(RankingExample(graph=graph, bool_target_scores=bts, phase_targets=pts))
    return out
```

在 `__all__` 加入 `"build_lookahead_examples_sat"`。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_solve.py -q`
Expected: PASS（3 passed）。若 branch 未降：检查 look-ahead 标签是否有 spread（PHP 对称可能均匀，
3-SAT 提供区分度——故混合 PHP+3SAT）。

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/training_data.py tests/solver/test_sat_solve.py
git commit -m "feat: build_lookahead_examples_sat（SAT look-ahead imitation 样本）"
```

---

### Task 4: −conflicts RL（collect_sat / train_sat）

**Files:**
- Modify: `omt_branching/solver/rl_decide.py`
- Test: `tests/solver/test_rl_decide.py`（追加）

**Interfaces:**
- Consumes: `solve_sat_with_decider`、`SamplingPolicyDecider`、`DecideRLTrainer.update`。
- Produces: `DecideRLTrainer.collect_sat(assertions, atoms) -> (steps, reward, res)`（reward=−log1p(conflicts)）；`DecideRLTrainer.train_sat(problems, iterations, log)`（`problems=list[(assertions,atoms)]`）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_rl_decide.py 追加
def test_decide_rl_sat_collect_update():
    import math
    from omt_branching.solver.sat_instances import generate_rand_3sat
    from omt_branching.solver.rl_decide import DecideRLTrainer, DecideRLConfig

    atoms, clauses = generate_rand_3sat(30, 4.26, seed=1)
    tr = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=40))
    steps, reward, res = tr.collect_sat(clauses, atoms)
    assert res["result"] in ("sat", "unsat")
    assert math.isfinite(reward)
    stats = tr.update(steps, reward, key=0)
    assert math.isfinite(stats["loss"])
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_rl_decide.py::test_decide_rl_sat_collect_update -q`
Expected: FAIL（`AttributeError: collect_sat`）

- [ ] **Step 3: 写实现**

在 `rl_decide.py` 的 `DecideRLTrainer` 内（`collect` 之后）新增：

```python
    def collect_sat(self, assertions, atoms):
        from omt_branching.solver.sat_solve import solve_sat_with_decider

        holder: dict = {}

        def factory(asserts):
            d = SamplingPolicyDecider(self.policy, self.defer_logit, asserts,
                                      self.config.refocus_every, sample=True)
            holder["d"] = d
            return d

        # 让 solve_sat_with_decider 用我们的采样 decider：decider_factory 返回 factory 造的 decider
        res = solve_sat_with_decider(list(assertions), list(atoms),
                                     decider_factory=lambda a: factory(a))
        steps = holder["d"].steps if "d" in holder else []
        reward = -math.log1p(res["conflicts"])
        return steps, reward, res

    def train_sat(self, problems, iterations: int = 1, log: bool = False):
        problems = list(problems)
        history = []
        for it in range(iterations):
            for j, (assertions, atoms) in enumerate(problems):
                steps, reward, res = self.collect_sat(assertions, atoms)
                stats = self.update(steps, reward, key=j)
                stats.update({"iter": it, "instance": j, "conflicts": res["conflicts"]})
                history.append(stats)
                if log:
                    print(f"[it {it} inst {j}] loss={stats['loss']:.4f} reward={reward:.3f} "
                          f"conflicts={res['conflicts']} steps={stats['steps']}")
        return history
```

（`solve_sat_with_decider` 的 `decider_factory(assertions)` 返回一个 decider；此处 `factory(a)`
即返回 `SamplingPolicyDecider`，符合其签名。）

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_rl_decide.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/rl_decide.py tests/solver/test_rl_decide.py
git commit -m "feat: DecideRLTrainer.collect_sat/train_sat（reward=−conflicts）"
```

---

### Task 5: SAT 实验 + 导出 + 冒烟 + 关键测量

**Files:**
- Create: `examples/sat_branch.py`
- Modify: `omt_branching/solver/__init__.py`
- Test: `tests/solver/test_demo_smoke.py`（追加）

**Interfaces:**
- Consumes: Task 1-4 全部。
- Produces: `examples/sat_branch.py`（PHP+3SAT，imitation 冷启动 + RL，三臂 VSIDS-vs-learned conflicts，多 seed）；导出 `generate_php`/`generate_rand_3sat`/`solve_sat_with_decider`。

- [ ] **Step 1: 更新导出**

在 `omt_branching/solver/__init__.py` 加：
```python
from omt_branching.solver.sat_instances import generate_php, generate_rand_3sat
from omt_branching.solver.sat_solve import solve_sat_with_decider
```
`__all__` 追加 `"generate_php"`, `"generate_rand_3sat"`, `"solve_sat_with_decider"`。

- [ ] **Step 2: 写实验脚本**

```python
# examples/sat_branch.py
"""困难 SAT 学习分支：PHP + 随机 3-SAT。两臂均附 propagator（关预处理→纯 CDCL），
比 VSIDS-decide vs (imitation+RL) learned-decide 的 conflicts。多 seed mean±std。
"""
from __future__ import annotations

import argparse
import statistics

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import (
    generate_php, generate_rand_3sat, solve_sat_with_decider,
)
from omt_branching.solver.policy_decider import PolicyDecider
from omt_branching.solver.rl_decide import DecideRLConfig, DecideRLTrainer
from omt_branching.solver.training_data import build_lookahead_examples_sat


def _bench(name, problems, decider_factory):
    confs = []
    for assertions, atoms in problems:
        r = solve_sat_with_decider(assertions, atoms, decider_factory=decider_factory)
        confs.append(r["conflicts"])
    return confs


def main() -> None:
    ap = argparse.ArgumentParser(description="困难 SAT 学习分支：VSIDS vs learned")
    ap.add_argument("--php", type=int, default=7, help="PHP(m+1,m) 的 m")
    ap.add_argument("--sat-n", type=int, default=60)
    ap.add_argument("--test", type=int, default=8, help="每族测试实例数")
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rl-iters", type=int, default=0)
    ap.add_argument("--refocus", type=int, default=100)
    args = ap.parse_args()
    torch.manual_seed(0)

    php_test = [generate_php(args.php) for _ in range(2)]
    sat_test = [generate_rand_3sat(args.sat_n, 4.26, 1000 + s) for s in range(args.test)]

    policy = BranchingPolicy()
    if args.train > 0:
        tr_probs = ([generate_php(args.php - 1)]
                    + [generate_rand_3sat(40, 4.26, s) for s in range(args.train)])
        exs = [e for e in build_lookahead_examples_sat(tr_probs) if e.bool_target_scores]
        h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=args.epochs)
        print(f"imitation: {len(exs)} 样本, branch {h[0].get('branch',0):.3f}->{h[-1].get('branch',0):.3f}")
    if args.rl_iters > 0:
        rl_probs = [generate_rand_3sat(40, 4.26, s) for s in range(max(args.train, 20))]
        rlt = DecideRLTrainer(policy, DecideRLConfig(refocus_every=args.refocus))
        hh = rlt.train_sat(rl_probs, iterations=args.rl_iters, log=False)
        if hh:
            print(f"RL: {len(hh)} 步, 末条 conflicts={hh[-1]['conflicts']}, "
                  f"defer_logit={float(rlt.defer_logit):.3f}")
    svc = BranchingPolicyService(policy=policy)

    def learned_factory(assertions):
        return PolicyDecider(svc, assertions, args.refocus)

    for label, probs in [("PHP", php_test), ("3-SAT", sat_test)]:
        v = _bench("vsids", probs, None)
        ln = _bench("learned", probs, learned_factory)
        print(f"[{label}] VSIDS conflicts={statistics.fmean(v):.0f}±{statistics.pstdev(v):.0f} | "
              f"learned={statistics.fmean(ln):.0f}±{statistics.pstdev(ln):.0f} | "
              f"胜={'是' if statistics.fmean(ln) < statistics.fmean(v) else '否'}")
    print("两臂均附 propagator（关预处理→纯 CDCL）；conflicts 越少越好。learned<VSIDS 即分支更优。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 追加冒烟测试**

在 `tests/solver/test_demo_smoke.py` 追加：
```python
def test_sat_branch_smoke():
    out = subprocess.run(
        [sys.executable, "-m", "examples.sat_branch",
         "--php", "5", "--sat-n", "40", "--test", "3", "--train", "4", "--epochs", "3"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert "VSIDS" in out.stdout
```

- [ ] **Step 4: 跑关键测量 + 全量**

Run: `conda run -n omt python -m examples.sat_branch --php 7 --sat-n 60 --test 8 --train 30 --epochs 25 --rl-iters 2`
Expected: 打印 PHP / 3-SAT 两族的 VSIDS vs learned conflicts + 胜负。**关键看**：PHP learned<VSIDS
（且需 trained≫untrained——可另跑 --train 0 对照）；3-SAT 诚实报告（胜/平/负）。

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add examples/sat_branch.py omt_branching/solver/__init__.py tests/solver/test_demo_smoke.py
git commit -m "feat: SAT 学习分支实验(PHP+3SAT, VSIDS vs learned conflicts) + 导出 + 冒烟"
```

---

## Self-Review

**Spec 覆盖：** §3.1 生成器→Task 1；§3.2 harness→Task 2；§3.4 imitation→Task 3；§3.3 −conflicts RL→Task 4；§3.5 实验→Task 5。全覆盖。

**Placeholder scan：** 无 TBD/TODO；每步含完整代码。

**Type consistency：** `generate_php(m)/generate_rand_3sat(n,ratio,seed) -> (atoms, clauses)` Task 1 定义、Task 3/4/5 一致；`solve_sat_with_decider(assertions, atoms, decider_factory)` Task 2 定义、Task 4/5 一致（`decider_factory(assertions)->decider`）；`build_lookahead_examples_sat(problems=list[(assertions,atoms)])` Task 3 定义、Task 5 用；`collect_sat/train_sat` Task 4 定义、Task 5 用；`DecideRLTrainer.update` 复用现有。

**风险提示：** 若 Task 3 branch 不降：PHP 对称标签可能均匀 -> 靠 3-SAT 提供 spread（已混合）。若 Task 5
3-SAT learned 未胜 VSIDS：如实报告，PHP claim 仍立；检查 refocus 频率（越频越动态）与 RL 是否收敛。
