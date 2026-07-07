# UserPropagator 学习分支 Phase 2a（look-ahead imitation）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 SAT look-ahead 教师(consequences 传播计数)监督训练 GNN,让 imitation-trained
learned-decide 在同回路里优于 VSIDS-decide（等 `== native` 下更少 conflicts/rlimit）。

**Architecture:** look-ahead 分数(每原子假设 → 强制其他原子数,product) → imitation 训练 bool/phase
head → PolicyDecider 用其 activity_priors 驱动 z3 内部 decide。RL 微调是 Phase 2b（另计划）。

**Tech Stack:** Python 3.10 / z3-solver 4.15.4 / PyTorch；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`（从 `Branching/`）。
- **不改 z3**；look-ahead 用 `z3.Solver.consequences([lit], atoms) -> (result, implications)`，
  每个 implication 是 `Implies(assump, consequent)`（`imp.arg(1)` = consequent；`z3.is_implies`）。
- docstring/注释中文；标识符/类型注解英文。
- 复用：`build_bool_snapshot`（Phase 1，返回 `(snap, amap: key->z3atom)`）、`atom_key`（=`str(atom)`）、
  `collect_atoms`、`GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap) -> HeteroGraph`（`graph.id_maps[NodeType.BOOL_VAR]: {atom_key->local}`）、`RankingExample(graph, bool_target_scores: dict[int,float], phase_targets: dict[int,bool])`、`ImitationTrainer(policy, TrainConfig).fit(examples, epochs) -> list[dict]`（含 `'branch'` 损失项）、`generate_bool_lia_dataset`、`BranchingPolicyService`、`PolicyDecider`、`solve_omt_with_decider`。

---

### Task 1: look-ahead 教师

**Files:**
- Create: `omt_branching/solver/lookahead.py`
- Test: `tests/solver/test_lookahead.py`

**Interfaces:**
- Consumes: `collect_atoms`, `atom_key`（Phase 1）。
- Produces:
  - `LookaheadConfig(max_atoms=32, sentinel=1e6, eps=1e-9)`
  - `lookahead_scores(assertions, atoms=None, config=LookaheadConfig()) -> tuple[dict[str,float], dict[str,bool]]`（(score_by_atomkey, phase_by_atomkey)）

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_lookahead.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")

from omt_branching.solver.lookahead import lookahead_scores, LookaheadConfig
from omt_branching.solver.propagator_snapshot import atom_key


def test_propagating_atom_outranks_isolated():
    x = [z3.Int(f"x{i}") for i in range(3)]
    a, b, c = x[0] >= 5, x[1] <= 2, x[2] >= 3
    hard = [x[0] >= 0, x[0] <= 10, x[1] >= 0, x[1] <= 10, x[2] >= 0, x[2] <= 10,
            z3.Or(a, b), z3.Or(z3.Not(a), c), z3.Or(b, c)]
    sc, ph = lookahead_scores(hard, atoms=[a, b, c])
    # a 两侧都传播(a=T->c, a=F->b)，b 相对孤立 -> score(a) > score(b)
    assert sc[atom_key(a)] > sc[atom_key(b)]
    assert atom_key(a) in ph and isinstance(ph[atom_key(a)], bool)


def test_failed_literal_gets_sentinel():
    x = z3.Int("x")
    # a: x>=8, 但另有 x<=3 硬约束 -> 假设 a=True 不可行 -> a 被强制为假(大哨兵)
    a = x >= 8
    hard = [x >= 0, x <= 10, x <= 3, z3.Or(a, x >= 1)]
    sc, ph = lookahead_scores(hard, atoms=[a], config=LookaheadConfig(sentinel=1e6))
    assert sc[atom_key(a)] >= 1e6      # failed literal
    assert ph[atom_key(a)] is False    # 可行侧是 a=False
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py -q`
Expected: FAIL（`ModuleNotFoundError: lookahead`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/lookahead.py
"""SAT look-ahead 教师：假设某原子，用 z3 consequences 计"强制了多少其他原子"（传播强度），
作 imitation 监督标签。传播强度是**子句共现图的函数**——正是 GNN 已见的特征，故可学
（对比 LIA 分离度需缺失的 LP 特征）。
"""
from __future__ import annotations

from dataclasses import dataclass

import z3

from omt_branching.solver.propagator_snapshot import atom_key, collect_atoms


@dataclass(frozen=True)
class LookaheadConfig:
    max_atoms: int = 32
    sentinel: float = 1e6
    eps: float = 1e-9


def _strip_not(e):
    while z3.is_not(e):
        e = e.arg(0)
    return e


def _count_other(imps, self_key: str) -> int:
    """统计蕴含到的**其他**原子数（剥离 Not，排除自身/双重否定）。"""
    seen = set()
    for imp in imps:
        cons = imp.arg(1) if (z3.is_implies(imp) and imp.num_args() == 2) else imp
        k = atom_key(_strip_not(cons))
        if k != self_key:
            seen.add(k)
    return len(seen)


def lookahead_scores(assertions, atoms=None, config: LookaheadConfig = LookaheadConfig()):
    atom_exprs = list(atoms) if atoms is not None else collect_atoms(list(assertions))
    atom_exprs = atom_exprs[: config.max_atoms]
    s = z3.Solver()
    s.add(*assertions)

    scores: dict = {}
    phases: dict = {}
    for a in atom_exprs:
        k = atom_key(a)
        try:
            res_t, imp_t = s.consequences([a], atom_exprs)
            res_f, imp_f = s.consequences([z3.Not(a)], atom_exprs)
        except z3.Z3Exception:
            continue
        t_unsat = res_t == z3.unsat
        f_unsat = res_f == z3.unsat
        if t_unsat and f_unsat:
            continue                      # 两侧皆不可行：矛盾/无关，跳过
        if t_unsat:                       # a=True 不可行 -> a 被强制为假
            scores[k] = config.sentinel
            phases[k] = False
            continue
        if f_unsat:
            scores[k] = config.sentinel
            phases[k] = True
            continue
        pt = _count_other(imp_t, k)
        pf = _count_other(imp_f, k)
        scores[k] = (pt + 1.0) * (pf + 1.0)   # march 风格 product：两侧都传播多者优
        phases[k] = pt >= pf                  # 先探传播更多的一侧
    return scores, phases


__all__ = ["LookaheadConfig", "lookahead_scores"]
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/lookahead.py tests/solver/test_lookahead.py
git commit -m "feat: SAT look-ahead 教师（consequences 传播计数 product + failed literal）"
```

---

### Task 2: build_lookahead_examples（imitation 样本）

**Files:**
- Modify: `omt_branching/solver/training_data.py`
- Test: `tests/solver/test_lookahead.py`（追加）

**Interfaces:**
- Consumes: `lookahead_scores`（Task 1）、`build_bool_snapshot`、`GraphBuilder`、`RankingExample`。
- Produces: `build_lookahead_examples(instances, config=LookaheadConfig()) -> list[RankingExample]`。

- [ ] **Step 1: 写失败测试**

在 `tests/solver/test_lookahead.py` 追加：

```python
def test_build_lookahead_examples_has_bool_labels():
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.training_data import build_lookahead_examples
    from omt_branching.interfaces import NodeType

    ds = generate_bool_lia_dataset(6, seed=3, min_vars=5, max_vars=6)
    exs = build_lookahead_examples(ds)
    assert exs and any(e.bool_target_scores for e in exs)
    e = next(e for e in exs if e.bool_target_scores)
    n_bool = e.graph.num_nodes(NodeType.BOOL_VAR)
    assert all(0 <= k < n_bool for k in e.bool_target_scores)
    assert e.phase_targets   # phase 标签也在
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py::test_build_lookahead_examples_has_bool_labels -q`
Expected: FAIL（`ImportError: build_lookahead_examples`）

- [ ] **Step 3: 写实现**

在 `training_data.py` 末尾（`__all__` 之前）新增：

```python
def build_lookahead_examples(instances, config=None):
    """从布尔结构实例构造 look-ahead imitation 样本（bool head ranking + phase）。"""
    from omt_branching.solver.lookahead import LookaheadConfig, lookahead_scores
    from omt_branching.solver.propagator_snapshot import build_bool_snapshot

    cfg = config or LookaheadConfig()
    out: list[RankingExample] = []
    for inst in instances:
        hard = list(inst.hard)
        snap, amap = build_bool_snapshot(hard)
        graph = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        scores, phases = lookahead_scores(hard, atoms=list(amap.values()), config=cfg)
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

在 `__all__` 加入 `"build_lookahead_examples"`。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/training_data.py tests/solver/test_lookahead.py
git commit -m "feat: build_lookahead_examples（look-ahead 标签 -> imitation 样本）"
```

---

### Task 3: imitation 可学习性回归护栏

**Files:**
- Test: `tests/solver/test_lookahead.py`（追加）

**Interfaces:**
- Consumes: `build_lookahead_examples`、`ImitationTrainer`、`TrainConfig`、`BranchingPolicy`。
- Produces: 无新代码——护栏，证明 look-ahead 标签**可被 GNN 特征学习**（branch 损失下降）。

- [ ] **Step 1: 写测试**

在 `tests/solver/test_lookahead.py` 追加：

```python
def test_lookahead_imitation_reduces_branch_loss():
    import torch
    from omt_branching.solver import generate_bool_lia_dataset
    from omt_branching.solver.training_data import build_lookahead_examples
    from omt_branching.model.policy import BranchingPolicy
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig

    torch.manual_seed(0)
    ds = generate_bool_lia_dataset(24, seed=7, min_vars=5, max_vars=6)
    exs = [e for e in build_lookahead_examples(ds) if e.bool_target_scores]
    assert exs, "应有带 bool 标签的样本"
    policy = BranchingPolicy()
    h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=15)
    assert "branch" in h[0]
    assert h[-1]["branch"] < h[0]["branch"]   # look-ahead 标签可学（子句图 = 特征）
```

- [ ] **Step 2: 运行**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py::test_lookahead_imitation_reduces_branch_loss -q`
Expected: PASS（look-ahead 标签是子句图函数，bool head 应能学）。若冻结在 uniform，检查
`build_bool_snapshot` 的子句图是否为空（回看 Phase 1 Task 1）。

- [ ] **Step 3: （无实现——护栏）**

- [ ] **Step 4: 重跑确认**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add tests/solver/test_lookahead.py
git commit -m "test: 护栏——look-ahead imitation 可学（branch 损失下降）"
```

---

### Task 4: 实验加 --train（imitation 冷启动）+ 测量 vs VSIDS

**Files:**
- Modify: `examples/decide_branch.py`
- Test: `tests/solver/test_demo_smoke.py`（改现有 decide_branch 冒烟加 --train）

**Interfaces:**
- Consumes: `build_lookahead_examples`、`ImitationTrainer`、Phase 1 三臂 harness。
- Produces: `examples/decide_branch.py` 支持 `--train N --epochs E`：先 look-ahead imitation
  训练 policy，再用**训练后** policy 跑 learned 臂。

- [ ] **Step 1: 改实验脚本**

`examples/decide_branch.py`：argparse 增 `--train`（默认 0=不训练）与 `--epochs`（默认 20）；
在构造 `svc` 前插入训练：

```python
    ap.add_argument("--train", type=int, default=0, help="look-ahead imitation 训练集规模(0=不训练)")
    ap.add_argument("--epochs", type=int, default=20)
```
```python
    policy = BranchingPolicy()
    if args.train > 0:
        from omt_branching.model.trainer import ImitationTrainer, TrainConfig
        from omt_branching.solver.training_data import build_lookahead_examples
        train = generate_bool_lia_dataset(args.train, seed=1,
                                          min_vars=args.min_vars, max_vars=args.max_vars)
        exs = [e for e in build_lookahead_examples(train) if e.bool_target_scores]
        hist = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=args.epochs)
        print(f"look-ahead imitation: {len(exs)} 样本, branch loss "
              f"{hist[0].get('branch', 0):.3f} -> {hist[-1].get('branch', 0):.3f}")
    svc = BranchingPolicyService(policy=policy)
```

（把原来 `svc = BranchingPolicyService(policy=BranchingPolicy())` 替换为上面这段。）

- [ ] **Step 2: 改冒烟测试加 --train**

把 `tests/solver/test_demo_smoke.py` 的 `test_decide_branch_smoke` 参数加上小规模训练：
```python
        [sys.executable, "-m", "examples.decide_branch",
         "--test", "5", "--train", "6", "--epochs", "3", "--min-vars", "5", "--max-vars", "5"],
```

- [ ] **Step 3: 跑实验测量（关键经验验证）**

Run: `conda run -n omt python -m examples.decide_branch --test 30 --train 60 --epochs 25 --min-vars 5 --max-vars 6`
Expected: 打印 imitation branch loss 下降 + 三臂对比。**关键看**：trained learned-decide 的
conflicts/rlimit 是否 **< VSIDS**（Phase 2a 成功判据）；`match=1` 必须保持。
（若 learned 仍劣于 VSIDS：记录数字，考虑 Phase 2b RL 或调 look-ahead 打分/refocus 频率——诚实报告。）

- [ ] **Step 4: 全量**

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add examples/decide_branch.py tests/solver/test_demo_smoke.py
git commit -m "feat: decide_branch 加 --train（look-ahead imitation 冷启动）+ 测量 vs VSIDS"
```

---

## Self-Review

**Spec 覆盖（Phase 2a）：** §2.1 look-ahead → Task 1 ✅；§2.2 imitation 样本 → Task 2 ✅；
可学习性验证 → Task 3 ✅；§2.4 实验(trained) → Task 4 ✅。§2.3 RL 为 Phase 2b（另计划，待 2a 出数）。

**Placeholder scan：** 无 TBD/TODO；每步含完整代码。

**Type consistency：** `lookahead_scores(assertions, atoms, config) -> (dict[str,float], dict[str,bool])`
Task 1 定义、Task 2/4 一致调用；`build_lookahead_examples(instances, config) -> list[RankingExample]`
Task 2 定义、Task 3/4 一致；`RankingExample(graph, bool_target_scores, phase_targets)` 与
`ImitationTrainer.fit` 的 `'branch'` 损失键一致。

**风险提示：** 若 Task 4 learned 未优于 VSIDS——不是 bug，是学习效果问题：诚实报告数字，
Phase 2b（RL）或调 look-ahead product→sum / refocus 频率 / 训练规模。Task 3 护栏先确保"能学"。
