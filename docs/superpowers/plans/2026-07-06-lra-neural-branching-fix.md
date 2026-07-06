# LRA 神经分支学习管线修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 LRA 神经分支的 imitation、准确率指标、RL 都作用在**实际驱动求解的 bool/atom head** 上，并用 strong-branching 专家冷启动，使 neural 清晰且正确地优于 baseline。

**Architecture:** 新增 strong-branching 布尔专家（一层前瞻，目标分离度打分）作 imitation 教师与准确率参照；`build_imitation_example` 增补 bool head 标签；准确率改量 bool head；RL 换 per-instance EMA baseline + 归一化 reward。所有 z3 交互经 `Z3Backend`（不改系统 z3）。

**Tech Stack:** Python 3.10 / PyTorch / `z3-solver` wheel；测试 pytest。

## Global Constraints

- 运行/测试：conda env **`omt`**；`conda run -n omt python -m pytest tests/solver`（从 `Branching/`）。
- **不修改、不重编译 z3**；z3 依赖仅限 `z3_backend.py` / `extractor.py` / `instance_gen.py` / 新 `strong_branch.py`。
- docstring/注释用**中文**；`__all__`/标识符/类型注解用英文。
- 不破坏 LIA 路径（`rl_demo.py` / 数值 head）。
- LRA 实例为纯实数 `z3.Real`；bool/atom head 特征来自 `NodeType.BOOL_VAR`。
- 类型：`Sense.MIN`/`Sense.MAX`；`Z3Backend.optimize(constraint, objective, sense) -> (model, value) | None`；`backend.conjoin(*c)` / `backend.negate(c)`；`graph.local_index(NodeType.BOOL_VAR, bid) -> int|None`；`graph.solver_id(NodeType.BOOL_VAR, local) -> bid`；`graph.id_maps[NodeType.BOOL_VAR]: {bid: local}`；`out.masked_bool_probs()`；`out.candidate_bool_local: list[int]`。

---

### Task 1: strong-branching 布尔专家

**Files:**
- Create: `omt_branching/solver/strong_branch.py`
- Test: `tests/solver/test_strong_branch.py`

**Interfaces:**
- Consumes: `Z3Backend`, `Z3SnapshotExtractor`, `GOMTProblem`, `OMTInstance`, `Sense`, `Extraction`（`.snapshot.theory_atoms[i].{bool_var_id,var_coeffs}`、`.snapshot.objective.var_coeffs`、`.atom_handles[bid].z3_obj`）。
- Produces:
  - `StrongBranchConfig(max_atoms:int=24, eps:float=1e-9)`
  - `_root_extraction(instance) -> (extraction, phi, objective, sense, backend)`（从原始 φ 抽取）
  - `strong_branch_scores(extraction, phi, objective, sense, backend, config=StrongBranchConfig()) -> (dict[bid->float], dict[bid->bool])`
  - `oracle_bool_choice(instance, config=StrongBranchConfig()) -> Optional[Hashable]`

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_strong_branch.py
from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import OMTInstance, Sense
from omt_branching.solver.strong_branch import (
    StrongBranchConfig, oracle_bool_choice, strong_branch_scores, _root_extraction,
)


def _sep_instance():
    """maximize x, 0<=x<=10；Or(x<=3, x>=7) 的两原子分离目标(分数~7)，
    Or(x<=50, x>=-5) 与盒约束原子被蕴含(分数 0)。"""
    x = z3.Real("x")
    hard = [x >= 0, x <= 10, z3.Or(x <= 3, x >= 7), z3.Or(x <= 50, x >= -5)]
    return OMTInstance(instance_id="t", variables=[x], hard=hard, objective=x,
                       sense=Sense.MAX, obj_coeffs={"x": 1.0}, theory="LRA")


def test_separating_atom_outranks_entailed():
    inst = _sep_instance()
    extraction, phi, obj, sense, backend = _root_extraction(inst)
    scores, phases = strong_branch_scores(extraction, phi, obj, sense, backend)
    assert scores, "应有候选原子打分"
    # 分离原子分数约为 7（x∈[0,3] 的 max=3 vs x∈[7,10] 的 max=10）
    assert max(scores.values()) == pytest.approx(7.0, abs=1e-6)
    # 存在被蕴含原子（一侧 UNSAT）得 0 分
    assert min(scores.values()) == pytest.approx(0.0, abs=1e-9)


def test_oracle_bool_choice_returns_separating_atom():
    inst = _sep_instance()
    bid = oracle_bool_choice(inst)
    assert bid is not None
    # 该 bid 对应的原子确为分离原子（其目标分离度为最大）
    extraction, phi, obj, sense, backend = _root_extraction(inst)
    scores, _ = strong_branch_scores(extraction, phi, obj, sense, backend)
    assert scores[bid] == max(scores.values())


def test_oracle_none_when_no_separation():
    """无布尔结构（只有盒约束，所有原子被蕴含）时返回 None。"""
    x = z3.Real("x")
    hard = [x >= 0, x <= 10]
    inst = OMTInstance(instance_id="flat", variables=[x], hard=hard, objective=x,
                       sense=Sense.MAX, obj_coeffs={"x": 1.0}, theory="LRA")
    assert oracle_bool_choice(inst) is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_strong_branch.py -q`
Expected: FAIL（`ModuleNotFoundError: omt_branching.solver.strong_branch`）

- [ ] **Step 3: 写实现**

```python
# omt_branching/solver/strong_branch.py
"""strong-branching 布尔专家：一层前瞻，用目标分离度给布尔原子打分。

在根状态对每个候选布尔原子 a 做一层前瞻 optimize(φ∧a) 与 optimize(φ∧¬a)，以两子最优
目标的分离度 |v_a − v_na| 打分——最能把目标分到两侧的原子，一侧出 incumbent 后另一侧会被
GOMT 的 Better 割廉价剪掉。恰有一侧 UNSAT 的原子近乎被 φ 蕴含、不产生真实进展，故记 0 分。
phase 目标为先探保留更优目标的一侧。候选原子取自**原始硬约束 φ**（不含增量 Better 割 δ0），
故不会把"目标上界割"误当结构原子。所有 z3 交互经 Z3Backend（不改系统 z3）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Optional

from omt_branching.solver.extractor import Z3SnapshotExtractor
from omt_branching.solver.instance_gen import OMTInstance
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.problem import GOMTProblem
from omt_branching.solver.z3_backend import Z3Backend


@dataclass(frozen=True)
class StrongBranchConfig:
    """strong-branching 专家配置。``max_atoms`` 限制每实例评测的候选原子数（按目标系数质量预筛）。"""

    max_atoms: int = 24
    eps: float = 1e-9


def _atom_obj_mass(var_coeffs: dict, obj_coeffs: dict) -> float:
    """原子涉及变量的 |目标系数| 质量，用于预筛（越大越可能与目标相关）。"""
    return sum(abs(obj_coeffs.get(v, 0.0)) for v in var_coeffs)


def _prefilter(snapshot, config: StrongBranchConfig) -> list[Hashable]:
    """按目标系数质量降序取前 ``max_atoms`` 个候选原子 bool_var_id。"""
    obj_coeffs = snapshot.objective.var_coeffs
    ranked = sorted(snapshot.theory_atoms,
                    key=lambda a: _atom_obj_mass(a.var_coeffs, obj_coeffs), reverse=True)
    return [a.bool_var_id for a in ranked[: config.max_atoms]]


def _root_extraction(instance: OMTInstance):
    """构造实例根状态并从**原始 φ**（不含 δ0 割）抽取。

    返回 ``(extraction, phi, objective, sense, backend)``；``φ`` 不可满足时抛异常（由调用方兜底）。
    """
    hard, obj, sense = instance.as_tuple()
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=hard, objective=obj, sense=sense)
    state = problem.initial_state(backend)          # 可能抛 Infeasible
    extraction = Z3SnapshotExtractor(problem).extract(state, backend)  # state.hard = 原始 φ
    phi = backend.conjoin(*hard)
    return extraction, phi, obj, sense, backend


def strong_branch_scores(extraction, phi, objective, sense: Sense, backend: Z3Backend,
                         config: StrongBranchConfig = StrongBranchConfig()):
    """给候选布尔原子打分。返回 ``(scores, phases)``，均以原子 bool_var_id 为键。

    - ``scores[bid] = |optimize(φ∧a) − optimize(φ∧¬a)|``；恰有一侧 UNSAT → 0（近乎被蕴含）。
    - ``phases[bid]``：先探保留更优目标的一侧（MAX→更高，MIN→更低）。
    """
    scores: dict[Hashable, float] = {}
    phases: dict[Hashable, bool] = {}
    for bid in _prefilter(extraction.snapshot, config):
        handle = extraction.atom_handles.get(bid)
        if handle is None:
            continue
        a = handle.z3_obj
        ra = backend.optimize(backend.conjoin(phi, a), objective, sense)
        rna = backend.optimize(backend.conjoin(phi, backend.negate(a)), objective, sense)
        if ra is None or rna is None:
            scores[bid] = 0.0
            continue
        try:
            va, vna = float(ra[1]), float(rna[1])
        except (TypeError, ValueError, OverflowError):
            continue
        scores[bid] = abs(va - vna)
        phases[bid] = (va >= vna) if sense is Sense.MAX else (va <= vna)
    return scores, phases


def oracle_bool_choice(instance: OMTInstance,
                       config: StrongBranchConfig = StrongBranchConfig()) -> Optional[Hashable]:
    """strong-branching 专家选择：目标分离度最大的原子 bool_var_id；无有意义分支返回 None。"""
    try:
        extraction, phi, obj, sense, backend = _root_extraction(instance)
    except Exception:
        return None
    scores, _ = strong_branch_scores(extraction, phi, obj, sense, backend, config)
    if not scores:
        return None
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > config.eps else None


__all__ = ["StrongBranchConfig", "strong_branch_scores", "oracle_bool_choice"]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_strong_branch.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/strong_branch.py tests/solver/test_strong_branch.py
git commit -m "feat: strong-branching 布尔专家（目标分离度打分）"
```

---

### Task 2: bool head imitation 标签 + 根 φ 抽取

**Files:**
- Modify: `omt_branching/solver/training_data.py`
- Test: `tests/solver/test_training_data.py`

**Interfaces:**
- Consumes: Task 1 的 `StrongBranchConfig` / `strong_branch_scores`；`RankingExample(graph, bool_target_scores, phase_targets, int_target_scores, int_dir_targets)`；`graph.id_maps[NodeType.BOOL_VAR]`。
- Produces:
  - `_initial_extraction(instance) -> (graph, extraction, backend)`（改为从原始 φ 抽取，返回 backend）
  - `build_imitation_example(instance, config=StrongBranchConfig())`（增补 `bool_target_scores` + `phase_targets`）
  - `bool_branch_hit(policy, instance, config=StrongBranchConfig()) -> Optional[bool]`

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_training_data.py
from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from omt_branching.solver import generate_lra_dataset
from omt_branching.solver.strong_branch import StrongBranchConfig
from omt_branching.solver.training_data import (
    build_imitation_example, bool_branch_hit,
)
from omt_branching.model.policy import BranchingPolicy


def _lra_inst():
    # 取一个含布尔结构、能产生分离原子的 LRA 实例
    ds = generate_lra_dataset(20, seed=3, min_vars=4, max_vars=5)
    for inst in ds:
        ex = build_imitation_example(inst)
        if ex is not None and ex.bool_target_scores:
            return inst
    pytest.skip("未生成含分离原子的实例")


def test_imitation_example_has_bool_labels():
    inst = _lra_inst()
    ex = build_imitation_example(inst)
    assert ex is not None
    assert ex.bool_target_scores, "bool head 应有非空 imitation 标签"
    # 标签键是合法的图内 BOOL_VAR 局部索引
    from omt_branching.interfaces import NodeType
    n_bool = ex.graph.num_nodes(NodeType.BOOL_VAR)
    assert all(0 <= k < n_bool for k in ex.bool_target_scores)


def test_bool_branch_hit_returns_bool():
    inst = _lra_inst()
    policy = BranchingPolicy()
    r = bool_branch_hit(policy, inst, StrongBranchConfig())
    assert r in (True, False)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_training_data.py -q`
Expected: FAIL（`ImportError: cannot import name 'bool_branch_hit'` 及 `bool_target_scores` 为空）

- [ ] **Step 3: 写实现**

改 `_initial_extraction`（从原始 φ 抽取、返回 backend），并更新其两个既有调用方，改写 `build_imitation_example`，新增 `bool_branch_hit`。替换 `training_data.py` 中对应片段：

`_initial_extraction`（替换整个函数体）：

```python
def _initial_extraction(instance: OMTInstance):
    """构造实例根状态并从**原始 φ** 抽取 (graph, extraction, backend)；不可行返回 (None, None, None)。

    从 ``state.hard``（原始 φ，不含 δ0 割）抽取，使布尔候选=结构原子，排除目标上界割。
    """
    hard, obj, sense = instance.as_tuple()
    backend = Z3Backend()
    problem = GOMTProblem(hard_list=hard, objective=obj, sense=sense)
    try:
        state = problem.initial_state(backend)
    except Exception:
        return None, None, None
    extractor = Z3SnapshotExtractor(problem)
    extraction = extractor.extract(state, backend)
    builder = GraphBuilder(DEFAULT_FEATURE_SPEC)
    graph = builder.build(extraction.snapshot)
    return graph, extraction, backend
```

`build_imitation_example`（替换整个函数体）：

```python
def build_imitation_example(instance: OMTInstance,
                            config: "StrongBranchConfig" = None) -> Optional[RankingExample]:
    """把单实例根图打包成 :class:`RankingExample`：数值 head（LIA 保留）+ bool head
    （strong-branching 专家，LRA 主路径）标签。"""
    from omt_branching.solver.strong_branch import StrongBranchConfig, strong_branch_scores

    cfg = config or StrongBranchConfig()
    graph, extraction, backend = _initial_extraction(instance)
    if graph is None:
        return None

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
        int_dirs[local] = (nv.objective_coeff >= 0) == is_max

    # --- bool head 标签（strong branching；LRA 主路径）---
    hard, obj, sense = instance.as_tuple()
    phi = backend.conjoin(*hard)
    raw_scores, raw_phases = strong_branch_scores(extraction, phi, obj, sense, backend, cfg)
    bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
    bool_scores: dict[int, float] = {}
    phase_targets: dict[int, bool] = {}
    for bid, sc in raw_scores.items():
        local = bmap.get(bid)
        if local is not None:
            bool_scores[local] = sc
    for bid, ph in raw_phases.items():
        local = bmap.get(bid)
        if local is not None:
            phase_targets[local] = ph

    if not int_scores and not bool_scores:
        return None
    return RankingExample(graph=graph, int_target_scores=int_scores, int_dir_targets=int_dirs,
                          bool_target_scores=bool_scores, phase_targets=phase_targets)
```

在文件末尾（`__all__` 之前）新增：

```python
def bool_branch_hit(policy: BranchingPolicy, instance: OMTInstance,
                    config: "StrongBranchConfig" = None) -> Optional[bool]:
    """在**同一根抽取**上比较策略 bool-head top-1 与 strong-branching 专家是否一致。

    无有意义分离原子（专家空/全 0）或无 bool 候选时返回 ``None``（该实例不计入准确率）。
    单次抽取保证专家与策略共享同一套原子 id，规避跨抽取 id 漂移。
    """
    import torch

    from omt_branching.solver.strong_branch import StrongBranchConfig, strong_branch_scores

    cfg = config or StrongBranchConfig()
    graph, extraction, backend = _initial_extraction(instance)
    if graph is None:
        return None
    hard, obj, sense = instance.as_tuple()
    phi = backend.conjoin(*hard)
    scores, _ = strong_branch_scores(extraction, phi, obj, sense, backend, cfg)
    if not scores or max(scores.values()) <= cfg.eps:
        return None
    oracle_bid = max(scores, key=lambda k: scores[k])

    out = policy.infer(graph)
    probs = out.masked_bool_probs()
    if probs.numel() == 0 or not out.candidate_bool_local:
        return None
    local = int(torch.argmax(probs).item())
    return graph.solver_id(NodeType.BOOL_VAR, local) == oracle_bid
```

更新既有两个调用方（把二元解包改三元）：`baseline_numeric_choice` 与 `policy_numeric_choice` 中的
`graph, extraction = _initial_extraction(instance)` → `graph, extraction, _ = _initial_extraction(instance)`。

更新 `__all__`：加入 `"bool_branch_hit"`。

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_training_data.py -q`
Expected: PASS（2 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/training_data.py tests/solver/test_training_data.py
git commit -m "feat: imitation 增补 bool head 标签(strong-branching) + bool_branch_hit 准确率"
```

---

### Task 3: 验证 imitation 真正训练 bool head

**Files:**
- Test: `tests/solver/test_training_data.py`（追加）

**Interfaces:**
- Consumes: `ImitationTrainer`, `TrainConfig`, `build_imitation_examples`（返回含 bool 标签的样本）。
- Produces: 无新代码——本任务是回归护栏，确认 finding ① 已修复（bool head 有梯度、loss 下降）。

- [ ] **Step 1: 写失败测试（先确认 build_imitation_examples 走新路径）**

在 `tests/solver/test_training_data.py` 追加：

```python
def test_imitation_trains_bool_head_loss_decreases():
    from omt_branching.solver import generate_lra_dataset
    from omt_branching.solver.training_data import build_imitation_examples
    from omt_branching.model.policy import BranchingPolicy
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig
    import torch

    torch.manual_seed(0)
    ds = generate_lra_dataset(24, seed=7, min_vars=4, max_vars=5)
    examples = [e for e in build_imitation_examples(ds) if e.bool_target_scores]
    assert examples, "应有带 bool 标签的样本"

    policy = BranchingPolicy()
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    hist = trainer.fit(examples, epochs=8)
    # bool head 有被训练：'branch' 损失项出现，且末轮 < 首轮
    assert "branch" in hist[0], "bool branching 损失项应存在（bool head 有梯度）"
    assert hist[-1]["branch"] < hist[0]["branch"]
```

- [ ] **Step 2: 运行测试**

Run: `conda run -n omt python -m pytest tests/solver/test_training_data.py::test_imitation_trains_bool_head_loss_decreases -q`
Expected: 直接 PASS（Task 2 已让 `build_imitation_examples` 产出 bool 标签；`ImitationTrainer` 已支持 bool head 损失）。若 FAIL，检查 Task 2 的标签映射。

- [ ] **Step 3: （无需实现——护栏测试）**

若 Step 2 已 PASS 则跳过。

- [ ] **Step 4: 重跑确认**

Run: `conda run -n omt python -m pytest tests/solver/test_training_data.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add tests/solver/test_training_data.py
git commit -m "test: 回归护栏——imitation 确实训练 bool head（branch 损失下降）"
```

---

### Task 4: 强化 RL —— per-instance baseline + 归一化 reward

**Files:**
- Modify: `omt_branching/solver/rl.py`
- Test: `tests/solver/test_rl.py`

**Interfaces:**
- Consumes: `RLConfig`, `RLStep`, `RLEpisode`, `SolverInLoopRLTrainer`, `Sense`, `BranchingPolicy`。
- Produces:
  - `RLConfig` 增字段 `eps: float = 1e-9`
  - `SolverInLoopRLTrainer._baselines: dict`；`_baseline_for(key)`；`_update_baseline_for(key, value)`
  - `SolverInLoopRLTrainer.update(episode, key=None)`（新增可选 `key`）
  - `SolverInLoopRLTrainer._shaped_rewards(steps, final_val, sense) -> list[float]`（按目标幅度归一化）

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_rl.py
from __future__ import annotations

import statistics

import pytest

z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver import RLConfig, RLStep
from omt_branching.solver.rl import SolverInLoopRLTrainer
from omt_branching.solver.interfaces import Sense


def _trainer(**kw):
    return SolverInLoopRLTrainer(BranchingPolicy(), RLConfig(**kw))


def test_shaped_rewards_normalized_by_span():
    tr = _trainer(reward_scale=1.0)
    steps = [RLStep(graph=None, head="bool", chosen_local=0, direction=True, value_at_decision=0.0),
             RLStep(graph=None, head="bool", chosen_local=0, direction=True, value_at_decision=5.0)]
    r = tr._shaped_rewards(steps, final_val=10.0, sense=Sense.MAX)
    assert r[0] == pytest.approx(0.5) and r[1] == pytest.approx(0.5)


def test_shaped_rewards_zero_span_is_zero():
    tr = _trainer()
    steps = [RLStep(graph=None, head="bool", chosen_local=0, direction=True, value_at_decision=3.0)]
    r = tr._shaped_rewards(steps, final_val=3.0, sense=Sense.MAX)
    assert r == [0.0]


def test_per_instance_baseline_tracks_each_key_and_lowers_variance():
    tr = _trainer(baseline_momentum=0.5)
    for _ in range(30):
        tr._update_baseline_for("a", -1.0)
        tr._update_baseline_for("b", -100.0)
    ba, bb = tr._baseline_for("a"), tr._baseline_for("b")
    assert ba == pytest.approx(-1.0, abs=1e-3)
    assert bb == pytest.approx(-100.0, abs=1e-3)
    # per-instance 优势方差 << 单一全局 baseline 优势方差
    adv_per = [(-1.0) - ba, (-100.0) - bb]
    g = (-1.0 - 100.0) / 2
    adv_global = [(-1.0) - g, (-100.0) - g]
    assert statistics.pvariance(adv_per) < statistics.pvariance(adv_global)


def test_baseline_for_unknown_key_falls_back_to_global():
    tr = _trainer()
    tr._baseline = -7.0
    assert tr._baseline_for("never-seen") == -7.0
    assert tr._baseline_for(None) == -7.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_rl.py -q`
Expected: FAIL（`AttributeError: _shaped_rewards / _baseline_for / _update_baseline_for`）

- [ ] **Step 3: 写实现**

在 `rl.py` 的 `RLConfig` dataclass 末尾（`f_sat_mode` 之后）新增字段：

```python
    eps: float = 1e-9                # 数值下限（reward 归一化除零保护）
```

`SolverInLoopRLTrainer.__init__` 增加 per-instance baseline 容器（在 `self._baseline = 0.0` 之后）：

```python
        self._baselines: dict = {}   # 实例键 -> 移动平均 baseline（去除实例间代价方差）
```

新增三个方法（放在 `update` 之前）：

```python
    def _baseline_for(self, key) -> float:
        """取该实例键的 baseline；未见过或 key=None 时回退全局。"""
        if key is None:
            return self._baseline
        return self._baselines.get(key, self._baseline)

    def _update_baseline_for(self, key, value: float) -> None:
        """按动量 EMA 更新 baseline；首次见到某键以其回报初始化（避免全局污染）。"""
        m = self.config.baseline_momentum
        if key is None:
            self._baseline = m * self._baseline + (1 - m) * value
        elif key in self._baselines:
            self._baselines[key] = m * self._baselines[key] + (1 - m) * value
        else:
            self._baselines[key] = value

    def _shaped_rewards(self, steps: list, final_val, sense: Sense) -> list[float]:
        """逐步 incumbent 提升，按 episode 内目标幅度归一到 O(1)；幅度为 0 则全 0。"""
        sign = 1.0 if sense is Sense.MAX else -1.0
        known = [s.value_at_decision for s in steps if s.value_at_decision is not None]
        if final_val is not None:
            known.append(final_val)
        span = (max(known) - min(known)) if len(known) >= 2 else 0.0
        scale = (self.config.reward_scale / span) if span > self.config.eps else 0.0
        rewards: list[float] = []
        for i, step in enumerate(steps):
            nxt = steps[i + 1].value_at_decision if i + 1 < len(steps) else final_val
            cur = step.value_at_decision
            if cur is None or nxt is None or scale == 0.0:
                rewards.append(0.0)
            else:
                rewards.append(sign * (nxt - cur) * scale)
        return rewards
```

在 `_build_episode` 中，把原来的逐步 reward 循环：

```python
        rewards: list[float] = []
        for i, step in enumerate(steps):
            nxt = steps[i + 1].value_at_decision if i + 1 < len(steps) else final_val
            cur = step.value_at_decision
            if cur is None or nxt is None:
                rewards.append(0.0)
            else:
                rewards.append(sign * (nxt - cur) * self.config.reward_scale)
```

替换为：

```python
        rewards = self._shaped_rewards(steps, final_val, sense)
```

（`sign` 变量若因此不再使用可保留或删除；`terminal`/`cost` 逻辑不变。）

改 `update` 签名与 baseline 使用：

- 签名 `def update(self, episode: RLEpisode) -> ...` → `def update(self, episode: RLEpisode, key=None) -> ...`
- 无步数早退分支里的 `"baseline": self._baseline` → `"baseline": self._baseline_for(key)`（两处早退都改）
- `advantage = G - self._baseline` → `advantage = G - self._baseline_for(key)`
- 末尾更新块：

```python
        mean_return = float(sum(returns) / len(returns))
        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1 - m) * mean_return
        return {"loss": float(loss), "mean_return": mean_return,
                "baseline": self._baseline, "steps": n}
```

替换为：

```python
        mean_return = float(sum(returns) / len(returns))
        self._update_baseline_for(key, mean_return)
        return {"loss": float(loss), "mean_return": mean_return,
                "baseline": self._baseline_for(key), "steps": n}
```

在 `train` 主循环中把 `stats = self.update(ep)` 改为 `stats = self.update(ep, key=j)`（`j` 为实例索引）。

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_rl.py -q`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/rl.py tests/solver/test_rl.py
git commit -m "feat: RL 换 per-instance EMA baseline + 目标幅度归一化 reward"
```

---

### Task 5: 接线、demo 准确率、导出、全量与冒烟

**Files:**
- Modify: `omt_branching/solver/__init__.py`
- Modify: `examples/rl_LRA.py`
- Test: `tests/solver/test_demo_smoke.py`（追加 LRA 冒烟）

**Interfaces:**
- Consumes: Task 1-4 的全部公共名。
- Produces: `solver` 包导出 `StrongBranchConfig` / `strong_branch_scores` / `oracle_bool_choice` / `bool_branch_hit`；`rl_LRA.py` 用 bool 准确率。

- [ ] **Step 1: 更新导出**

在 `omt_branching/solver/__init__.py`：新增 import

```python
from omt_branching.solver.strong_branch import (
    StrongBranchConfig, oracle_bool_choice, strong_branch_scores,
)
```

`training_data` 的 import 行加入 `bool_branch_hit`：

```python
from omt_branching.solver.training_data import (
    build_imitation_example, build_imitation_examples, policy_numeric_choice,
    baseline_numeric_choice, bool_branch_hit,
)
```

`__all__` 增加：`"StrongBranchConfig"`, `"oracle_bool_choice"`, `"strong_branch_scores"`, `"bool_branch_hit"`。

- [ ] **Step 2: 改 `rl_LRA.py` 准确率量 bool head**

替换 `branch_accuracy`（整个函数）：

```python
def branch_accuracy(policy, instances, config=None) -> tuple[float, int]:
    """Neural bool-head top-1 与 strong-branching 专家一致的比例，及有效实例数。"""
    from omt_branching.solver import StrongBranchConfig
    from omt_branching.solver.training_data import bool_branch_hit

    cfg = config or StrongBranchConfig()
    hit = total = 0
    for inst in instances:
        r = bool_branch_hit(policy, inst, cfg)
        if r is None:
            continue
        total += 1
        hit += 1 if r else 0
    return (hit / total if total else 0.0), total
```

更新三处调用与打印（改为 bool 准确率 + 有效数，去掉无意义的 baseline 列）：

- 训练前：
```python
    acc_before, n_valid = branch_accuracy(policy, test_set)
    print(f"训练前 bool 分支准确率(vs strong-branching 专家): {acc_before:.2f} (有效 {n_valid} 例)")
```
- 训练后：
```python
    acc_after, _ = branch_accuracy(policy, test_set)
    print(f"训练后 bool 分支准确率: {acc_after:.2f}")
```
- 第 4 节测试：
```python
    acc_neural, n_valid = branch_accuracy(reloaded, test_set)
    print(f"[准确率] bool-head top-1 与 strong-branching 专家一致: {acc_neural:.2f} (有效 {n_valid} 例)")
```
删除 `oracle_numeric_choice`/数值 head 相关的旧准确率打印引用；`import` 行按需清理（保留 `policy_numeric_choice`/`oracle_numeric_choice` 若他处仍用，否则移除）。

- [ ] **Step 3: 改 RL 默认迭代与 reward_scale**

`rl_LRA.py`：`--iters` 默认 `1` → `3`；`RLConfig(...)` 里 `reward_scale=0.1` → `reward_scale=1.0`（配合归一化 reward）。

```python
    parser.add_argument("--iters", type=int, default=3, help="RL 训练轮数")
```
```python
    rl_config = RLConfig(
        lr=1e-3, gamma=0.98, entropy_coef=5e-3,
        rlimit_penalty_coef=1.0, use_log_cost=True, reward_scale=1.0,
        max_split_depth=args.split_depth, max_steps=args.max_steps,
        f_sat_mode=F_SAT_MODE,
    )
```

- [ ] **Step 4: 追加 LRA 冒烟测试**

在 `tests/solver/test_demo_smoke.py` 追加：

```python
def test_rl_lra_smoke_reports_bool_accuracy():
    out = subprocess.run(
        [sys.executable, "-m", "examples.rl_LRA",
         "--train", "6", "--test", "6", "--min-vars", "10", "--max-vars", "10",
         "--iters", "1", "--epochs", "3", "--max-steps", "40"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert "bool 分支准确率" in out.stdout or "bool-head" in out.stdout
```

- [ ] **Step 5: 全量测试 + 冒烟**

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿（含新增 test_strong_branch / test_training_data / test_rl / LRA 冒烟）。

- [ ] **Step 6: 提交**

```bash
git add omt_branching/solver/__init__.py examples/rl_LRA.py tests/solver/test_demo_smoke.py
git commit -m "feat: rl_LRA 准确率改量 bool head + 导出接线 + LRA 冒烟测试"
```

---

## Self-Review

**Spec coverage：**
- Part 1 strong-branch 专家 → Task 1 ✅
- Part 2 bool imitation + policy 查询 → Task 2（`bool_branch_hit` 取代 spec 的 `policy_bool_choice`，单抽取更 robust）✅
- Part 3 诚实准确率 → Task 5 Step 2 ✅
- Part 4 RL（per-instance baseline / reward 归一化 / iters）→ Task 4 + Task 5 Step 3 ✅
- Part 5 接线/测试 → Task 3（护栏）+ Task 5 ✅

**Placeholder scan：** 无 TBD/TODO；每步含完整代码或明确改动位置。

**Type consistency：** `_initial_extraction` 三元返回在 Task 2 统一（含更新既有两调用方）；`strong_branch_scores` 签名 `(extraction, phi, objective, sense, backend, config)` 在 Task 1/2 一致；`update(episode, key=None)` 在 Task 4 内部 `train` 同步；`bool_branch_hit(policy, instance, config)` 在 Task 2 定义、Task 5 调用一致；`branch_accuracy` 返回 `(rate, n)` 与三处调用一致。

**偏差记录：** spec 的 `policy_bool_choice` 用 `bool_branch_hit`（单次抽取内同时算专家与策略选择）替代——因布尔原子 id 为遍历序号、跨抽取不稳定，单抽取比较更 robust。
