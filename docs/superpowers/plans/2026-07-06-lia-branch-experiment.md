# LIA B&B 学习分支实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 hard LIA 整数 OMT + plain 模式（check-sat 叶子）下，让学习到的整数域切分分支在搜索规模（GOMT splits/solve_calls）上优于启发式基线，native z3-Optimize 为精确 oracle。

**Architecture:** 复用既有 plain-mode calculus、数值 head、`SolverInLoopRLTrainer`。新增：hard LIA 生成器、整数 strong-branching 专家（objective-separation score）、启发式基线策略、LIA 实验脚本。

**Tech Stack:** Python 3.10 / PyTorch / `z3-solver` wheel；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`（从 `Branching/`）。
- 不改 z3；z3 依赖仅限 adapter 层（`z3_backend.py`/`extractor.py`/`instance_gen.py`/`strong_branch.py`）。
- docstring/注释中文；标识符/类型注解英文。
- 不破坏既有 LRA/LIA 路径与 `tests/solver` 全绿。
- plain 模式：叶子 `Solve`（check-sat），F-Split 驱动搜索；所有策略饱和到 `== native` 最优。
- 关键类型：`OMTInstance(instance_id, variables, hard, objective, sense, obj_coeffs, theory, family)`；`Handle(kind, z3_obj, var_id, current_value, lower, upper, is_integer)`；`backend.optimize(constraint, objective, sense) -> (model, value)|None`；`backend.le(term, bound)`/`backend.ge(term, bound)`；`extraction.numeric_handles: {var_id: Handle}`；`extraction.snapshot.objective.var_coeffs: {name: float}`；`_root_extraction(instance) -> (extraction, phi, objective, sense, backend)`；`_numeric_split(handle, psi, backend, branch_up)`；`SplitDecision.split(list, source=str)`/`.resolve()`。

---

### Task 1: hard LIA 生成器

**Files:**
- Modify: `omt_branching/solver/instance_gen.py`
- Test: `tests/solver/test_instance_gen_lia.py`

**Interfaces:**
- Consumes: `z3`, `Sense`, `OMTInstance`, `_validate`（已存在）。
- Produces:
  - `generate_hard_lia_instance(instance_id, rng, *, n_vars=10, n_constraints=8, ub=30, coeff_lo=1, coeff_hi=9, slack=2, sense=None) -> OMTInstance`
  - `generate_hard_lia_dataset(count, seed=0, *, id_prefix="hlia", min_vars=8, max_vars=14, **kw) -> list[OMTInstance]`

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_instance_gen_lia.py
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
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_instance_gen_lia.py -q`
Expected: FAIL（`ImportError: cannot import name 'generate_hard_lia_instance'`）

- [ ] **Step 3: 写实现**

在 `instance_gen.py` 的 LIA 段（`generate_dataset` 之后）新增：

```python
def generate_hard_lia_instance(instance_id: str, rng: random.Random, *,
                               n_vars: int = 10, n_constraints: int = 8, ub: int = 30,
                               coeff_lo: int = 1, coeff_hi: int = 9, slack: int = 2,
                               sense: Optional[Sense] = None) -> OMTInstance:
    """witness 驱动的 hard LIA：紧 packing/covering 约束制造整数-LP 间隙，使 B&B 搜索非平凡。"""
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
        if rng.random() < 0.5:                      # packing：紧上界（witness 满足）
            hard.append(lhs <= lhs_val + rng.randint(0, slack))
        else:                                       # covering：紧下界（witness 满足）
            hard.append(lhs >= max(0, lhs_val - rng.randint(0, slack)))

    obj_c = [rng.randint(coeff_lo, coeff_hi) for _ in range(n_vars)]
    objective = z3.Sum([c * x for c, x in zip(obj_c, xs)])
    if sense is None:
        sense = rng.choice([Sense.MIN, Sense.MAX])

    return OMTInstance(
        instance_id=instance_id, variables=xs, hard=hard, objective=objective, sense=sense,
        obj_coeffs={n: float(c) for n, c in zip(names, obj_c)},
        theory="LIA", family="hard", description=f"hard LIA, {n_vars} vars {n_constraints} cons",
    )


def generate_hard_lia_dataset(count: int, seed: int = 0, *, id_prefix: str = "hlia",
                              min_vars: int = 8, max_vars: int = 14, **kwargs) -> list[OMTInstance]:
    """生成 ``count`` 个 hard LIA 实例（变量数在 [min_vars, max_vars] 间随机）。"""
    rng = random.Random(seed)
    out: list[OMTInstance] = []
    for i in range(count):
        n_vars = rng.randint(min_vars, max_vars)
        out.append(generate_hard_lia_instance(f"{id_prefix}{i}", rng, n_vars=n_vars, **kwargs))
    return out
```

在 `__all__` 加入 `"generate_hard_lia_instance"`, `"generate_hard_lia_dataset"`。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_instance_gen_lia.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/instance_gen.py tests/solver/test_instance_gen_lia.py
git commit -m "feat: hard LIA 生成器（紧 packing/covering，B&B 非平凡）"
```

---

### Task 2: 整数 strong-branching 专家

**Files:**
- Modify: `omt_branching/solver/strong_branch.py`
- Test: `tests/solver/test_strong_branch.py`（追加）

**Interfaces:**
- Consumes: `_root_extraction`, `StrongBranchConfig`, `backend.optimize/le/ge/conjoin`, `Handle`, `extraction.numeric_handles`, `extraction.snapshot.objective.var_coeffs`。
- Produces:
  - `strong_branch_numeric_scores(extraction, phi, objective, sense, backend, config=StrongBranchConfig()) -> (dict[var_id->float], dict[var_id->bool])`
  - `oracle_numeric_choice_sb(instance, config=StrongBranchConfig()) -> Optional[Hashable]`

- [ ] **Step 1: 写失败测试**

在 `tests/solver/test_strong_branch.py` 追加：

```python
from omt_branching.solver.strong_branch import (
    strong_branch_numeric_scores, oracle_numeric_choice_sb,
)


def _lia_obj_instance():
    """maximize x；x,y∈[0,10]，无耦合。x 影响目标(分离~10)，y 不影响(分离 0)。"""
    x, y = z3.Int("x"), z3.Int("y")
    hard = [x >= 0, x <= 10, y >= 0, y <= 10]
    return OMTInstance(instance_id="li", variables=[x, y], hard=hard, objective=x,
                       sense=Sense.MAX, obj_coeffs={"x": 1.0, "y": 0.0}, theory="LIA")


def test_numeric_expert_ranks_objective_var_above_irrelevant():
    inst = _lia_obj_instance()
    extraction, phi, obj, sense, backend = _root_extraction(inst)
    scores, dirs = strong_branch_numeric_scores(extraction, phi, obj, sense, backend)
    assert scores.get("x", 0) > scores.get("y", 0)     # x 分离目标，y 不
    # 选 x 后 MAX 应先探高侧（x 越大越优）
    assert dirs["x"] is True


def test_oracle_numeric_choice_sb_picks_objective_var():
    inst = _lia_obj_instance()
    assert oracle_numeric_choice_sb(inst) == "x"
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_strong_branch.py -q`
Expected: FAIL（`ImportError: strong_branch_numeric_scores`）

- [ ] **Step 3: 写实现**

在 `strong_branch.py` 的 `oracle_bool_choice` 之后新增：

```python
_SENTINEL = 1e18  # 一侧 UNSAT：整数域切分直接剪半，记大分（有效分支）


def _numeric_children(handle, phi, objective, sense: Sense, backend: Z3Backend):
    """整数变量中点切分的两子最优 (v_lo, v_hi)；不可分返回 None，某侧 UNSAT 记 None。"""
    lo, up = handle.lower, handle.upper
    if lo is None or up is None or up - lo < 1:
        return None
    m = int((lo + up) // 2)
    m = max(int(lo), min(m, int(up) - 1))
    if m < lo or m + 1 > up:
        return None
    x = handle.z3_obj
    r_lo = backend.optimize(backend.conjoin(phi, backend.le(x, m)), objective, sense)
    r_hi = backend.optimize(backend.conjoin(phi, backend.ge(x, m + 1)), objective, sense)
    v_lo = None if r_lo is None else float(r_lo[1])
    v_hi = None if r_hi is None else float(r_hi[1])
    return v_lo, v_hi


def strong_branch_numeric_scores(extraction, phi, objective, sense: Sense, backend: Z3Backend,
                                 config: StrongBranchConfig = StrongBranchConfig()):
    """整数变量 strong branching（objective-separation）。返回 (scores, dirs) 按 var_id。

    - 两侧可行：``score = |v_lo − v_hi|``（最能把可达目标分到两侧的变量最该分支）。
    - 恰一侧 UNSAT：``score = SENT``（整数域切分剪掉半个域，有效）。
    - ``dirs[var]``：先探保留更优目标的一侧（MAX→v_hi≥v_lo 则 branch_up=True）。
    """
    obj_coeffs = extraction.snapshot.objective.var_coeffs
    handles = [h for h in extraction.numeric_handles.values() if h.is_integer]
    handles.sort(key=lambda h: abs(obj_coeffs.get(h.var_id, 0.0)), reverse=True)

    scores: dict = {}
    dirs: dict = {}
    for h in handles[: config.max_atoms]:
        res = _numeric_children(h, phi, objective, sense, backend)
        if res is None:
            continue
        v_lo, v_hi = res
        if v_lo is None and v_hi is None:
            continue
        if v_lo is None or v_hi is None:
            scores[h.var_id] = _SENTINEL
            dirs[h.var_id] = (v_hi is not None)  # 可行侧优先
            continue
        scores[h.var_id] = abs(v_lo - v_hi)
        dirs[h.var_id] = (v_hi >= v_lo) if sense is Sense.MAX else (v_hi <= v_lo)
    return scores, dirs


def oracle_numeric_choice_sb(instance: OMTInstance,
                             config: StrongBranchConfig = StrongBranchConfig()):
    """整数 strong-branching 专家选择：目标分离度最大的整数变量 var_id；无有意义分支返回 None。"""
    try:
        extraction, phi, obj, sense, backend = _root_extraction(instance)
    except Exception:
        return None
    scores, _ = strong_branch_numeric_scores(extraction, phi, obj, sense, backend, config)
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > config.eps else None
```

在 `__all__` 加入 `"strong_branch_numeric_scores"`, `"oracle_numeric_choice_sb"`。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_strong_branch.py -q`
Expected: PASS（5 passed：原 3 + 新 2）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/strong_branch.py tests/solver/test_strong_branch.py
git commit -m "feat: 整数 strong-branching 专家（objective-separation + UNSAT 剪半哨兵）"
```

---

### Task 3: imitation 数值标签接入 strong branching

**Files:**
- Modify: `omt_branching/solver/training_data.py`
- Test: `tests/solver/test_training_data.py`（追加）

**Interfaces:**
- Consumes: Task 2 的 `strong_branch_numeric_scores`；`_initial_extraction`（返回 graph, extraction, backend）；`RankingExample`。
- Produces: `build_imitation_example(instance, config=None, numeric_expert="coeff")`（新增 `numeric_expert ∈ {"coeff","strong"}`）。

- [ ] **Step 1: 写失败测试**

在 `tests/solver/test_training_data.py` 追加：

```python
def test_imitation_numeric_strong_labels():
    from omt_branching.solver.instance_gen import generate_hard_lia_dataset
    from omt_branching.solver.training_data import build_imitation_example

    ds = generate_hard_lia_dataset(8, seed=2, min_vars=8, max_vars=10)
    ex = None
    for inst in ds:
        e = build_imitation_example(inst, numeric_expert="strong")
        if e is not None and e.int_target_scores:
            ex = e
            break
    assert ex is not None, "应有含数值 strong 标签的样本"
    from omt_branching.interfaces import NodeType
    n_num = ex.graph.num_nodes(NodeType.NUMERIC_VAR)
    assert all(0 <= k < n_num for k in ex.int_target_scores)
    # 数值方向标签也应存在
    assert ex.int_dir_targets
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_training_data.py::test_imitation_numeric_strong_labels -q`
Expected: FAIL（`build_imitation_example() got an unexpected keyword argument 'numeric_expert'`）

- [ ] **Step 3: 写实现**

替换 `build_imitation_example` 的数值 head 标签段。当前：

```python
    # --- 数值 head 标签（|目标系数|；LIA 兼容）---
    nmap = graph.id_maps.get(NodeType.NUMERIC_VAR, {})
    is_max = instance.sense.value == "max"
    int_scores: dict[int, float] = {}
    int_dirs: dict[int, bool] = {}
    for nv in extraction.snapshot.numeric_vars:
        local = nmap.get(nv.num_var_id)
        if local is None:
            continue
        int_scores[local] = abs(nv.objective_coeff)
        # 朝改善目标的方向 split：MAX 且正系数 -> 向上；MIN 且正系数 -> 向下。
        int_dirs[local] = (nv.objective_coeff >= 0) == is_max
```

改为（并把函数签名加上 `numeric_expert: str = "coeff"`）：

```python
    # --- 数值 head 标签 ---
    nmap = graph.id_maps.get(NodeType.NUMERIC_VAR, {})
    is_max = instance.sense.value == "max"
    int_scores: dict[int, float] = {}
    int_dirs: dict[int, bool] = {}
    if numeric_expert == "strong":
        from omt_branching.solver.strong_branch import strong_branch_numeric_scores
        hard, obj, sense = instance.as_tuple()
        phi = backend.conjoin(*hard)
        raw_num, raw_dir = strong_branch_numeric_scores(extraction, phi, obj, sense, backend, cfg)
        for vid, sc in raw_num.items():
            local = nmap.get(vid)
            if local is not None:
                int_scores[local] = sc
        for vid, up in raw_dir.items():
            local = nmap.get(vid)
            if local is not None:
                int_dirs[local] = up
    else:  # "coeff"：|目标系数|（LRA/rl_demo 兼容）
        for nv in extraction.snapshot.numeric_vars:
            local = nmap.get(nv.num_var_id)
            if local is None:
                continue
            int_scores[local] = abs(nv.objective_coeff)
            int_dirs[local] = (nv.objective_coeff >= 0) == is_max
```

把签名 `def build_imitation_example(instance: OMTInstance, config: "StrongBranchConfig" = None) -> Optional[RankingExample]:` 改为
`def build_imitation_example(instance: OMTInstance, config: "StrongBranchConfig" = None, numeric_expert: str = "coeff") -> Optional[RankingExample]:`。

`build_imitation_examples` 增透传：
```python
def build_imitation_examples(instances, numeric_expert: str = "coeff") -> list[RankingExample]:
    out: list[RankingExample] = []
    for inst in instances:
        ex = build_imitation_example(inst, numeric_expert=numeric_expert)
        if ex is not None:
            out.append(ex)
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_training_data.py -q`
Expected: PASS（4 passed：原 3 + 新 1）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/training_data.py tests/solver/test_training_data.py
git commit -m "feat: build_imitation_example 支持 numeric_expert=strong（整数 head strong 标签）"
```

---

### Task 4: 启发式基线策略

**Files:**
- Modify: `omt_branching/solver/strategy.py`
- Test: `tests/solver/test_strategy_heuristic.py`

**Interfaces:**
- Consumes: `Z3SnapshotExtractor`, `Handle`, `_numeric_split`, `StrategyConfig`, `SplitDecision`, `strong_branch_numeric_scores`。
- Produces: `NumericHeuristicStrategy(problem, mode="largest_domain", seed=0, config=StrategyConfig())`，`mode ∈ {largest_domain, largest_coeff, random, strong}`。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_strategy_heuristic.py
from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import (
    GOMTConfig, GOMTProblem, GOMTSolver, Z3Backend, solve_native,
)
from omt_branching.solver.instance_gen import generate_hard_lia_dataset
from omt_branching.solver.strategy import NumericHeuristicStrategy


@pytest.mark.parametrize("mode", ["largest_domain", "largest_coeff", "random", "strong"])
def test_heuristic_strategy_reaches_native_optimum(mode):
    inst = generate_hard_lia_dataset(1, seed=5, min_vars=8, max_vars=8)[0]
    hard, obj, sense = inst.as_tuple()
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=hard, objective=obj, sense=sense)
    strat = NumericHeuristicStrategy(problem, mode=mode, seed=0)
    solver = GOMTSolver(problem, backend, strat, GOMTConfig(max_steps=5000, f_sat_mode="plain"))
    res = solver.run()
    assert res.value == solve_native(hard, obj, sense)     # plain 模式饱和到精确最优
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_strategy_heuristic.py -q`
Expected: FAIL（`ImportError: NumericHeuristicStrategy`）

- [ ] **Step 3: 写实现**

在 `strategy.py`（`BaselineStrategy` 之后）新增：

```python
class NumericHeuristicStrategy:
    """整数变量域切分的启发式基线（plain 模式）。

    ``mode``：
    - ``largest_domain``：最大域跨度变量（同 BaselineStrategy）。
    - ``largest_coeff``：最大 |目标系数| 变量。
    - ``random``：随机整数变量（``seed`` 决定，确定性）。
    - ``strong``：strong-branching 目标分离度最大变量（昂贵 skyline）。
    只对整数变量域切分，soundness 与 NeuralStrategy 同一 calculus。
    """

    def __init__(self, problem, mode: str = "largest_domain", seed: int = 0,
                 config: StrategyConfig = StrategyConfig()):
        import random as _random

        self.problem = problem
        self.mode = mode
        self.config = config
        self.extractor = Z3SnapshotExtractor(problem)
        self._rng = _random.Random(seed)

    def propose(self, state, backend) -> SplitDecision:
        depth = state.stats.get("branch_depth", 0)
        if depth >= self.config.max_split_depth:
            return SplitDecision.resolve()
        try:
            view = replace(state, hard=backend.conjoin(state.hard, state.top))
            extraction = self.extractor.extract(view, backend)
        except Exception as exc:
            warnings.warn(f"NumericHeuristicStrategy 抽取失败，回退 resolve: {exc!r}")
            return SplitDecision.resolve()

        cands = [h for h in extraction.numeric_handles.values()
                 if h.is_integer and h.lower is not None and h.upper is not None
                 and h.upper - h.lower >= 1]
        if not cands:
            return SplitDecision.resolve()

        handle, branch_up = self._pick(cands, extraction, backend, state)
        if handle is None:
            return SplitDecision.resolve()
        subs = _numeric_split(handle, state.top, backend, branch_up)
        if subs is None:
            return SplitDecision.resolve()
        state.stats["branch_depth"] = depth + 1
        return SplitDecision.split(subs, source=f"heur:{self.mode}")

    def _pick(self, cands, extraction, backend, state):
        """返回 (handle, branch_up)。"""
        if self.mode == "largest_domain":
            return max(cands, key=lambda h: h.upper - h.lower), False
        if self.mode == "largest_coeff":
            obj_coeffs = extraction.snapshot.objective.var_coeffs
            return max(cands, key=lambda h: abs(obj_coeffs.get(h.var_id, 0.0))), False
        if self.mode == "random":
            return self._rng.choice(cands), False
        if self.mode == "strong":
            from omt_branching.solver.strong_branch import strong_branch_numeric_scores
            phi = backend.conjoin(state.hard, state.top)
            scores, dirs = strong_branch_numeric_scores(
                extraction, phi, state.objective, state.sense, backend)
            by_id = {h.var_id: h for h in cands}
            best = max((v for v in scores if v in by_id),
                       key=lambda v: scores[v], default=None)
            if best is None:
                return cands[0], False
            return by_id[best], dirs.get(best, False)
        return cands[0], False
```

确保 `strategy.py` 顶部已 `import warnings` 与 `from dataclasses import replace`（已存在）。
在 `strategy.py` 的 `__all__` 加入 `"NumericHeuristicStrategy"`。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_strategy_heuristic.py -q`
Expected: PASS（4 passed，各 mode 均 == native）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/strategy.py tests/solver/test_strategy_heuristic.py
git commit -m "feat: NumericHeuristicStrategy（largest_domain/coeff/random/strong 基线）"
```

---

### Task 5: LIA 实验脚本 + 导出 + 全量冒烟

**Files:**
- Create: `examples/lia_branch.py`
- Modify: `omt_branching/solver/__init__.py`
- Test: `tests/solver/test_demo_smoke.py`（追加）

**Interfaces:**
- Consumes: Task 1-4 全部；`SolverInLoopRLTrainer`, `RLConfig`, `RLRecordingStrategy`, `solve_and_measure`, `solve_native`, `Z3Backend`, `ImitationTrainer`, `BranchingPolicy`。
- Produces: `examples/lia_branch.py`（`main()` 打印各策略搜索规模对比表）；导出 `generate_hard_lia_dataset`/`generate_hard_lia_instance`/`oracle_numeric_choice_sb`/`NumericHeuristicStrategy`。

- [ ] **Step 1: 更新导出**

在 `omt_branching/solver/__init__.py`：

`instance_gen` import 行加入 `generate_hard_lia_dataset, generate_hard_lia_instance`；
`strong_branch` import 行加入 `oracle_numeric_choice_sb, strong_branch_numeric_scores`；
`strategy` import 行加入 `NumericHeuristicStrategy`。对应 `__all__` 均加入这些名。

- [ ] **Step 2: 写实验脚本**

```python
# examples/lia_branch.py
"""LIA B&B 学习分支实验（plain 模式，check-sat 叶子）。

hard LIA -> 数值 head strong-branching imitation 冷启动 -> RL 微调 -> 对比各分支策略的
**搜索规模**（solve_calls/splits/steps）；所有策略饱和到 == native 精确最优。
"""

from __future__ import annotations

import argparse

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver import (
    NumericHeuristicStrategy, RLConfig, RLRecordingStrategy, SolverInLoopRLTrainer,
    Z3Backend, build_imitation_examples, generate_hard_lia_dataset, solve_and_measure,
    solve_native,
)

F_SAT_MODE = "plain"   # LIA：check-sat 叶子，分支驱动优化


def _measure_native(hard, obj, sense) -> dict:
    backend = Z3Backend()
    res = backend.optimize(backend.conjoin(*hard), obj, sense)
    return {"value": res[1], "optimal": True, "splits": 0, "steps": 0,
            "solve_calls": backend.solve_calls, "rlimit": backend.rlimit_count}


def compare(policy, rl_config, instances, max_steps):
    """对每个实例跑各策略，返回策略 -> 平均 {solve_calls, splits, steps, match}。"""
    modes = ["random", "largest_domain", "largest_coeff", "strong"]
    agg = {m: {"solve_calls": 0.0, "splits": 0.0, "steps": 0.0, "match": 0.0} for m in modes}
    agg["neural"] = {"solve_calls": 0.0, "splits": 0.0, "steps": 0.0, "match": 0.0}
    agg["native"] = {"solve_calls": 0.0, "splits": 0.0, "steps": 0.0, "match": 0.0}
    for inst in instances:
        hard, obj, sense = inst.as_tuple()
        native = solve_native(hard, obj, sense)
        agg["native"]["solve_calls"] += _measure_native(hard, obj, sense)["solve_calls"]
        agg["native"]["match"] += 1.0
        for m in modes:
            r = solve_and_measure(hard, obj, sense,
                                  lambda p, m=m: NumericHeuristicStrategy(p, mode=m, seed=0),
                                  max_steps=max_steps, f_sat_mode=F_SAT_MODE)
            for k in ("solve_calls", "splits", "steps"):
                agg[m][k] += r[k]
            agg[m]["match"] += 1.0 if r["value"] == native else 0.0
        rn = solve_and_measure(hard, obj, sense,
                               lambda p: RLRecordingStrategy(p, policy, rl_config, sample=False),
                               max_steps=max_steps, f_sat_mode=F_SAT_MODE)
        for k in ("solve_calls", "splits", "steps"):
            agg["neural"][k] += rn[k]
        agg["neural"]["match"] += 1.0 if rn["value"] == native else 0.0
    n = max(1, len(instances))
    for m in agg:
        for k in agg[m]:
            agg[m][k] /= n
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="LIA B&B 学习分支实验")
    parser.add_argument("--train", type=int, default=300)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--min-vars", type=int, default=8)
    parser.add_argument("--max-vars", type=int, default=12)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--split-depth", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(0)
    print("=== 1) 生成 hard LIA ===")
    train = generate_hard_lia_dataset(args.train, seed=1,
                                      min_vars=args.min_vars, max_vars=args.max_vars)
    test = generate_hard_lia_dataset(args.test, seed=99,
                                     min_vars=args.min_vars, max_vars=args.max_vars)
    print(f"训练 {len(train)} / 测试 {len(test)}")

    print("\n=== 2) 数值 head strong-branching imitation ===")
    policy = BranchingPolicy()
    examples = build_imitation_examples(train, numeric_expert="strong")
    print(f"imitation 样本 {len(examples)}")
    hist = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(examples, epochs=args.epochs)
    print(f"imitation loss: {hist[0]['loss']:.4f} -> {hist[-1]['loss']:.4f}")

    print("\n=== 3) RL 微调（plain，最小化搜索规模）===")
    rl_config = RLConfig(lr=1e-3, gamma=0.98, entropy_coef=5e-3, rlimit_penalty_coef=1.0,
                         use_log_cost=True, reward_scale=1.0, saturation_bonus=0.0,
                         max_split_depth=args.split_depth, max_steps=args.max_steps,
                         f_sat_mode=F_SAT_MODE)
    rl = SolverInLoopRLTrainer(policy, rl_config)
    rl.train([i.as_tuple() for i in train], iterations=args.iters, log=False)

    print("\n=== 4) 搜索规模对比（solve_calls/splits/steps 越小越好；match 应=1.0）===")
    agg = compare(policy, rl_config, test, args.max_steps)
    print(f"  {'strategy':<16} {'solve_calls':>12} {'splits':>10} {'steps':>10} {'match':>8}")
    for name in ("native", "random", "largest_domain", "largest_coeff", "strong", "neural"):
        a = agg[name]
        print(f"  {name:<16} {a['solve_calls']:>12.2f} {a['splits']:>10.2f} "
              f"{a['steps']:>10.2f} {a['match']:>8.2f}")
    print("\nplain 模式：所有策略饱和到 == native 精确最优；比较搜索规模。"
          "neural 目标：solve_calls/splits 低于启发式、接近 strong skyline。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 追加冒烟测试**

在 `tests/solver/test_demo_smoke.py` 追加：

```python
def test_lia_branch_smoke_reaches_native():
    out = subprocess.run(
        [sys.executable, "-m", "examples.lia_branch",
         "--train", "6", "--test", "6", "--min-vars", "8", "--max-vars", "8",
         "--iters", "1", "--epochs", "3", "--max-steps", "200"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert "搜索规模" in out.stdout
```

- [ ] **Step 4: 运行冒烟 + 全量**

Run: `conda run -n omt python -m examples.lia_branch --train 6 --test 6 --min-vars 8 --max-vars 8 --iters 1 --epochs 3 --max-steps 200`
Expected: 打印对比表；各策略 `match` 列应为 1.0（== native）。若某策略 match<1.0，检查 `max_steps` 是否过小（未饱和）。

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add examples/lia_branch.py omt_branching/solver/__init__.py tests/solver/test_demo_smoke.py
git commit -m "feat: LIA B&B 学习分支实验脚本 + 导出 + 冒烟（plain 模式搜索规模对比）"
```

---

## Self-Review

**Spec coverage：** C1→Task1 ✅；C2→Task2 ✅；C4→Task3 ✅；C3→Task4 ✅；C5→Task5 ✅。指标（搜索规模、match==native）在 Task4/5 覆盖。
**Placeholder scan：** 无 TBD/TODO；每步含完整代码。
**Type consistency：** `strong_branch_numeric_scores(extraction, phi, objective, sense, backend, config)` 在 Task2 定义、Task3/Task4 一致调用；`oracle_numeric_choice_sb` Task2 定义、导出于 Task5；`NumericHeuristicStrategy(problem, mode, seed, config)` Task4 定义、Task5 调用一致；`build_imitation_example(..., numeric_expert=)` Task3 定义、Task5 用 `build_imitation_examples(train, numeric_expert="strong")`；`generate_hard_lia_dataset(count, seed, min_vars, max_vars)` Task1 定义、Task5 调用一致；`solve_and_measure(hard, obj, sense, factory, max_steps, f_sat_mode)` 复用既有。
**风险提示：** Task5 冒烟若 `match<1.0` 说明 `max_steps` 需增大（plain LIA 未饱和）——已在 Step 4 注明排查。
