# UserPropagator 学习分支 —— Phase 2（学习：imitation + RL）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Phase 1 打通的 propagator/decide 管道上**加入学习**：SAT look-ahead 教师产出布尔
分支标签做 **imitation 冷启动**，再用 **Solver-in-the-Loop REINFORCE** 在 z3 内部布尔决策回路
里微调。成功标准（spec §6 Phase 2）= learned-decide 在 **rlimit** 上优于 VSIDS-decide，且最优
值 `== native`（正确性）。

**Architecture:** 复用 Phase 1 的 `LearnedDecidePropagator`（接管 decide）+ `build_bool_snapshot`
（原子 + 子句共现图）+ `solve_omt_with_decider`（OMT-over-Solver 线性搜索回路）。Phase 2 新增：

```
SAT look-ahead 教师(lookahead.py) ─┐
                                   ├─▶ ImitationTrainer(复用) 冷启动 bool/phase head
build_bool_snapshot -> HeteroGraph ┘
                                   │
        (冷启动权重) ─▶ DecideRLTrainer(decide_rl.py): SamplingDecider 在每次 refocus
                        采样"聚焦原子" -> next_split 驱动 z3 内部决策；奖励 = -log(1+rlimit)
                        -> REINFORCE 微调
```

- **动作** = 每次 refocus 时策略在候选原子上给出的优先级分布中**采样**的聚焦原子（+相位）；
  该原子被优先 `next_split`，直接驱动 z3 内部布尔决策。
- **奖励** = 纯终局 `-log(1+rlimit)`（整体 OMT 求解开销）。线性搜索回路本身保证 `== native`
  最优（正确性与分支无关），故优化目标即"用更少开销到达最优"。

**Tech Stack:** Python 3.10 / z3-solver 4.15.x / PyTorch；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`（从 `Branching/`）。若无 `omt`
  环境，用装有 `z3-solver` + `torch` + `pytest` 的等价环境（如 base）。
- **不改 z3**；`z3.Optimize` 不支持 propagator，OMT 走 Solver 回路（Phase 1 结论）。
- docstring/注释中文；标识符/类型注解英文。
- **复用**（不改）：`ImitationTrainer`/`RankingExample`/`TrainConfig`（`model/trainer.py`）、
  `BranchingPolicy`/`_masked_softmax`（`model/policy.py`）、`solve_omt_with_decider`/
  `PolicyDecider`/`LearnedDecidePropagator`/`build_bool_snapshot`（Phase 1）、
  `save_policy`/`load_policy`/`save_history`（`model/persistence.py`）、
  `generate_bool_lia_dataset`/`solve_native`（`solver/`）。
- **降级为消融/基础设施**：GOMT F-Split 的 `solver/rl.py`（数值域二分 RL）、`strong_branch.py`
  与 `training_data.py`（GOMT 路径 imitation）保持不动，作对照，不在 decide 路径复用。
- 关键契约：`RankingExample(graph, bool_target_scores: dict[local->float], phase_targets:
  dict[local->bool], ...)`；`graph.id_maps[NodeType.BOOL_VAR]: {atom_key -> local}`；
  `graph.solver_id(NodeType.BOOL_VAR, local) -> atom_key`；`PolicyOutput.candidate_bool_local`、
  `.bool_branch_scores`、`.phase_logits`、`.masked_bool_probs()`；
  `solve_omt_with_decider(hard, obj, sense, decider_factory, max_iters) -> {value, rlimit,
  conflicts, decisions, iters}`；`decider(undecided_keys, assignment) -> Optional[(key, bool)]`。

---

### Task 1: SAT look-ahead 布尔专家（imitation 标签）

**Files:**
- Create: `omt_branching/solver/lookahead.py`
- Test: `tests/solver/test_lookahead.py`

**Interfaces:**
- Produces:
  - `LookaheadConfig(max_atoms=32, product_weight=1024.0, sentinel=1e6, eps=1e-9)`
  - `lookahead_scores(assertions, atoms=None, config) -> (scores: dict[key,float], phases: dict[key,bool])`
  - `build_lookahead_example(assertions, feature_spec, config) -> Optional[RankingExample]`
  - `build_lookahead_examples(assertion_lists, ...) -> list[RankingExample]`
  - `decide_bool_hit(policy, assertions, ...) -> Optional[bool]`（策略 top-1 vs 专家 top-1）
- Consumes: `build_bool_snapshot`/`collect_atoms`/`atom_key`（Phase 1）、`GraphBuilder`、
  `RankingExample`。

**打分原理：** 对每个候选原子 a，用 `z3.Solver.consequences([a], atoms)` /
`consequences([Not(a)], atoms)` 求赋真/赋假后的**蕴含闭包规模** `p⁺`/`p⁻`（单元传播 look-ahead）。

- 两侧可行：`score = w·p⁺·p⁻ + p⁺ + p⁻`（经典 March 式：偏好两侧都传播的平衡原子）。
- 恰一侧 UNSAT：原子被 φ 蕴含（等价单元）→ `score = sentinel`（极强），phase 取可行/强制侧。
- `phase = (p⁺ >= p⁻)`：先探传播更多的一侧。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_lookahead.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.lookahead import (
    LookaheadConfig, lookahead_scores, build_lookahead_example,
    build_lookahead_examples, decide_bool_hit,
)
from omt_branching.solver.instance_gen import generate_bool_lia_dataset
from omt_branching.solver.propagator_snapshot import atom_key


def test_lookahead_scores_forced_literal_gets_high_score():
    x = z3.Int("x")
    a, b = x >= 5, x <= 20
    asserts = [x >= 6, x <= 100, z3.Or(a, b)]     # x>=6 蕴含 a=(x>=5)
    scores, phases = lookahead_scores(asserts)
    ka = atom_key(a)
    assert ka in scores
    assert scores[ka] >= LookaheadConfig().sentinel - 1   # ¬a 侧 UNSAT -> sentinel
    assert phases[ka] is True                              # 探被强制的方向


def test_lookahead_scores_symmetric_atom_finite():
    x, y = z3.Int("x"), z3.Int("y")
    a = x >= y
    asserts = [x >= 0, x <= 10, y >= 0, y <= 10, z3.Or(a, x <= 3)]
    scores, phases = lookahead_scores(asserts)
    ka = atom_key(a)
    assert ka in scores
    assert 0.0 <= scores[ka] < LookaheadConfig().sentinel


def test_build_lookahead_example_maps_local_indices():
    inst = generate_bool_lia_dataset(1, seed=3, min_vars=4, max_vars=4)[0]
    ex = build_lookahead_example(list(inst.hard))
    assert ex is not None
    from omt_branching.interfaces import NodeType
    bmap = ex.graph.id_maps.get(NodeType.BOOL_VAR, {})
    assert bmap
    assert all(0 <= k < len(bmap) for k in ex.bool_target_scores)
    assert ex.bool_target_scores


def test_decide_bool_hit_returns_bool_or_none():
    inst = generate_bool_lia_dataset(1, seed=7, min_vars=4, max_vars=4)[0]
    hit = decide_bool_hit(BranchingPolicy(), list(inst.hard))
    assert hit is None or isinstance(hit, bool)


def test_build_lookahead_examples_batch():
    insts = generate_bool_lia_dataset(3, seed=1, min_vars=4, max_vars=4)
    examples = build_lookahead_examples([list(i.hard) for i in insts])
    assert 1 <= len(examples) <= 3
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py -q`
Expected: FAIL（`ModuleNotFoundError: lookahead`）

- [ ] **Step 3: 写实现（`omt_branching/solver/lookahead.py`）**

要点：
- `_propagation_count(solver, lit, probe)`：`solver.consequences([lit], probe)`；`unsat -> None`，
  否则返回被蕴含文字数；`z3.Z3Exception -> 0`（保守）。
- `lookahead_scores`：先 `Solver.check() == sat` 兜底（φ 本身 UNSAT 返回空）；`probe = atoms`；
  对 `atoms[:max_atoms]` 逐原子算 `p⁺/p⁻`，按上文规则合成 `score/phase`。
- `build_lookahead_example`：`build_bool_snapshot -> GraphBuilder.build`，`lookahead_scores`，
  经 `graph.id_maps[BOOL_VAR]` 把 `atom_key -> local`；无有意义分支返回 `None`。
- `decide_bool_hit`：同一建图上比 `argmax(masked_bool_probs)` 的 `solver_id` 与专家 top-1 键。

（完整实现见仓库 `omt_branching/solver/lookahead.py`，约 165 行。）

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_lookahead.py -q`
Expected: PASS（5 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/lookahead.py tests/solver/test_lookahead.py
git commit -m "feat: SAT look-ahead 布尔专家(单元传播打分)+imitation 标签构建"
```

---

### Task 2: look-ahead imitation 冷启动（复用 ImitationTrainer）

**Files:**
- Test: `tests/solver/test_decide_imitation.py`
- （无新实现：复用 `ImitationTrainer` + Task 1 的 `build_lookahead_examples` / `decide_bool_hit`。）

**Interfaces:** `ImitationTrainer(policy, TrainConfig).fit(examples, epochs)`，`examples =
build_lookahead_examples([...])`；准确率 `decide_bool_hit`。imitation 损失 = `L_branch + λ·L_phase`
（`trainer.py` 现有多任务损失，仅 bool/phase 项被 look-ahead 标签激活）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_decide_imitation.py
from __future__ import annotations
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.solver.instance_gen import generate_bool_lia_dataset
from omt_branching.solver.lookahead import build_lookahead_examples


def test_imitation_coldstart_reduces_loss():
    torch.manual_seed(0)
    insts = generate_bool_lia_dataset(6, seed=11, min_vars=4, max_vars=5)
    examples = build_lookahead_examples([list(i.hard) for i in insts])
    assert examples
    policy = BranchingPolicy()
    trainer = ImitationTrainer(policy, TrainConfig(lr=5e-3))
    history = trainer.fit(examples, epochs=15)
    assert history[-1]["loss"] < history[0]["loss"]     # 收敛
    assert "branch" in history[-1]                        # bool head 参与训练
```

- [ ] **Step 2-4: 运行确认失败→（无实现改动）→通过**

Run: `conda run -n omt python -m pytest tests/solver/test_decide_imitation.py -q`
Expected: 依赖 Task 1 完成后 PASS（1 passed）。

- [ ] **Step 5: 提交**

```bash
git add tests/solver/test_decide_imitation.py
git commit -m "test: look-ahead imitation 冷启动降低 bool/phase 损失"
```

---

### Task 3: Solver-in-the-Loop RL（decide 路径 REINFORCE）

**Files:**
- Create: `omt_branching/solver/decide_rl.py`
- Test: `tests/solver/test_decide_rl.py`

**Interfaces:**
- `DecideRLConfig(lr, gamma, entropy_coef, baseline_momentum, grad_clip, device, refocus_every=200,
  max_record=512, rlimit_penalty_coef=1.0, use_log_cost=True, max_iters=100000, eps)`
- `SamplingDecider(policy, assertions, config, feature_spec, sample=True)`：可调用
  `__call__(undecided_keys, assignment) -> Optional[(key,bool)]`，每次 refocus 采样聚焦原子并把
  `DecideStep(graph, cand_locals, chosen_local, phase)` 追加到 `.steps`。
- `DecideRLTrainer(policy, config)`：`collect_episode(hard,obj,sense) -> DecideEpisode`；
  `update(episode, key) -> {loss,return,baseline,steps}`；`train(instances, iterations, log)`；
  `evaluate(hard,obj,sense) -> dict`（argmax，同 `solve_omt_with_decider` 指标）；`save`/`load`。
- Consumes: `solve_omt_with_decider`（Phase 1，`decider_factory=lambda a: decider` 传入预建
  SamplingDecider 以便读回 `.steps`）、`build_bool_snapshot`、`GraphBuilder`、`_masked_softmax`。

**REINFORCE 细节：** 纯终局奖励 `G = -rlimit_penalty_coef·log(1+rlimit)`，所有 refocus 动作共享
`G`；`advantage = G - baseline(per-instance EMA)`；每步 `logp = log softmax(scores|cand)[chosen]
+ log Bernoulli(phase)`；`loss = (-Σ logp·adv - entropy_coef·Σ ent)/n`。rollout 用 `policy.infer`
无梯度采样并缓存图；更新时对缓存图重跑前向（梯度只流向参数）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_decide_rl.py
from __future__ import annotations
import math
import pytest
z3 = pytest.importorskip("z3")
torch = pytest.importorskip("torch")

from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver import generate_bool_lia_dataset, solve_native
from omt_branching.solver.decide_rl import DecideRLConfig, DecideRLTrainer, SamplingDecider
from omt_branching.solver.propagator_snapshot import atom_key


def test_sampling_decider_records_steps():
    x = z3.Int("x")
    a, b = x >= 5, x <= 2
    asserts = [x >= 0, x <= 10, z3.Or(a, b)]
    dec = SamplingDecider(BranchingPolicy(), asserts, DecideRLConfig(refocus_every=100), sample=True)
    und = [atom_key(a), atom_key(b)]
    choice = dec(und, {})
    assert choice is not None
    assert choice[0] in und and isinstance(choice[1], bool)
    assert len(dec.steps) == 1


def test_collect_episode_matches_native_and_updates():
    inst = generate_bool_lia_dataset(1, seed=5, min_vars=4, max_vars=4)[0]
    hard, obj, sense = inst.as_tuple()
    trainer = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=20))
    ep = trainer.collect_episode(hard, obj, sense)
    assert ep.value == solve_native(hard, obj, sense)
    assert ep.rlimit > 0
    assert ep.decisions is not None
    stats = trainer.update(ep, key=0)
    assert math.isfinite(stats["loss"])


def test_train_and_evaluate():
    insts = generate_bool_lia_dataset(2, seed=8, min_vars=4, max_vars=4)
    instances = [i.as_tuple() for i in insts]
    trainer = DecideRLTrainer(BranchingPolicy(), DecideRLConfig(refocus_every=20))
    history = trainer.train(instances, iterations=1, log=False)
    assert len(history) == 2
    hard, obj, sense = instances[0]
    res = trainer.evaluate(hard, obj, sense)
    assert res["value"] == solve_native(hard, obj, sense)
    assert res["rlimit"] > 0
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_decide_rl.py -q`
Expected: FAIL（`ModuleNotFoundError: decide_rl`）

- [ ] **Step 3: 写实现（`omt_branching/solver/decide_rl.py`）**

结构：`DecideRLConfig` / `DecideStep` / `DecideEpisode` dataclass；`SamplingDecider`
（`_refocus` 建图+推理+采样聚焦原子并记录，`__call__` 优先返回聚焦原子否则取当前最高优先级
未定原子，无候选返回 `None` 退回 VSIDS）；`DecideRLTrainer`（`collect_episode` 用预建 decider
跑 `solve_omt_with_decider` 读回 `.steps` 与 rlimit，`update` REINFORCE，`train`/`evaluate`/
`save`/`load`）。

（完整实现见仓库 `omt_branching/solver/decide_rl.py`，约 260 行。）

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_decide_rl.py -q`
Expected: PASS（3 passed；episode `== native`，update loss 有限）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/decide_rl.py tests/solver/test_decide_rl.py
git commit -m "feat: decide 路径 Solver-in-the-Loop REINFORCE(动作=refocus 聚焦原子, 奖励=-log rlimit)"
```

---

### Task 4: Phase 2 端到端实验 + 导出 + 冒烟

**Files:**
- Create: `examples/decide_learn.py`
- Modify: `omt_branching/solver/__init__.py`（导出 Task 1/3 公共 API）
- Test: `tests/solver/test_demo_smoke.py`（追加 `test_decide_learn_smoke`）

**Interfaces:** `examples/decide_learn.py` main：生成 bool-LIA 训练/测试集 → look-ahead imitation
冷启动（打印准确率 before/after + imitation 三臂对比）→ RL 微调 → 持久化 round-trip → RL 后三臂
对比（native / VSIDS / learned），打印 `Phase 2 结论`。

- [ ] **Step 1: 更新导出**

`omt_branching/solver/__init__.py` 增加：
```python
from omt_branching.solver.lookahead import (
    LookaheadConfig, build_lookahead_example, build_lookahead_examples,
    decide_bool_hit, lookahead_scores,
)
from omt_branching.solver.decide_rl import (
    DecideEpisode, DecideRLConfig, DecideRLTrainer, DecideStep, SamplingDecider,
)
```
`__all__` 追加：`"LookaheadConfig"`, `"lookahead_scores"`, `"build_lookahead_example"`,
`"build_lookahead_examples"`, `"decide_bool_hit"`, `"DecideRLConfig"`, `"DecideStep"`,
`"DecideEpisode"`, `"SamplingDecider"`, `"DecideRLTrainer"`。

- [ ] **Step 2: 写实验脚本 `examples/decide_learn.py`**

关键函数：`branch_accuracy(policy, instances)`（`decide_bool_hit` 命中率）、
`three_arm(policy, instances, refocus)`（native/VSIDS/learned 三臂，learned 走
`PolicyDecider` + `solve_omt_with_decider`）、`main`（imitation → RL → 两次三臂对比 + 持久化）。
（完整脚本见仓库 `examples/decide_learn.py`。）

- [ ] **Step 3: 追加冒烟测试**

```python
# tests/solver/test_demo_smoke.py（追加）
def test_decide_learn_smoke():
    out = subprocess.run(
        [sys.executable, "-m", "examples.decide_learn",
         "--train", "6", "--test", "4", "--min-vars", "4", "--max-vars", "4",
         "--iters", "1", "--epochs", "3", "--refocus", "20"],
        capture_output=True, text=True, timeout=900,
    )
    assert out.returncode == 0, out.stderr
    assert "Phase 2 结论" in out.stdout
```

- [ ] **Step 4: 运行冒烟 + 全量**

Run: `conda run -n omt python -m examples.decide_learn --train 8 --test 6 --min-vars 4 --max-vars 4 --iters 1 --epochs 8 --refocus 20`
Expected: 打印两次三臂对比 + `Phase 2 结论`；**learned match=1.00**（正确性 == native）；
decisions>0（propagator 生效）。

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add examples/decide_learn.py omt_branching/solver/__init__.py tests/solver/test_demo_smoke.py
git commit -m "feat: Phase 2 端到端(look-ahead imitation + decide RL)实验 + 导出 + 冒烟"
```

---

## Self-Review

**Spec coverage（Phase 2 部分）：**
- §3.4 look-ahead 教师 → Task 1 ✅（`consequences` 单元传播打分，bool + phase 标签）。
- §3.5 imitation 训练（复用 `ImitationTrainer`）→ Task 2 ✅。
- §3.6 RL 训练器（动作=refocus 优先级、奖励=−rlimit、REINFORCE、imitation 冷启动后微调）→
  Task 3 ✅（`SamplingDecider` + `DecideRLTrainer`）。
- §3.7/§6 实验 + 指标（learned vs VSIDS vs native，rlimit/conflicts/decisions，== native）→
  Task 4 ✅。
- §5 风险"无子句图学不动"：沿用 Phase 1 `build_bool_snapshot` 的子句共现图，未削弱。

**为什么不改 `solver/rl.py`：** 该模块是 GOMT **F-Split**（数值域二分）的 RL，与 decide 路径的
动作/奖励语义不同（外层 split vs z3 内部布尔决策）。spec §7 已将 GOMT F-Split 降级为消融，故
Phase 2 **新建** `decide_rl.py`（decide 路径专用），保留 `rl.py` 作对照，避免破坏既有实验。

**Placeholder scan：** 无 TBD/TODO；实现已落地并通过测试。

**Type consistency：** `lookahead_scores -> (dict[key,float], dict[key,bool])` 在 Task 1 定义、
Task 3/4 未直接依赖其内部；`build_lookahead_examples -> list[RankingExample]` 与
`ImitationTrainer.fit` 契约一致；`SamplingDecider.__call__` 与 propagator 的
`decider(undecided_keys, assignment) -> Optional[(key,bool)]` 一致；`DecideEpisode.value/rlimit/
decisions` 源自 `solve_omt_with_decider` 返回 dict，键一致。

**验证结果（base 等价环境，z3 4.15.x + torch 2.4）：**
- `tests/solver` 非 smoke 全量：**114 passed**（含新增 `test_lookahead` 5、`test_decide_imitation`
  1、`test_decide_rl` 3）。
- `examples.decide_learn` 冒烟：exit 0，两次三臂对比 **learned match=1.00**（== native），
  imitation loss 下降，RL 采集/更新正常。

**诚实的风险提示（供后续调参/换实例时注意）：**
- **rlimit 杠杆问题**：bool-LIA 小实例的总 rlimit 主要被**外层算术线性搜索**主导，decide 步数
  相对很小 → RL 奖励信号弱、learned 未必在少量训练下即优于 VSIDS。要达成 spec §6"learned <
  VSIDS rlimit"，需换**布尔分支主导求解代价**的实例（更多/更深析取、更弱算术），并增大
  `--train`/`--iters`，或在奖励里加 conflicts 项。
- **look-ahead 开销**：`consequences` 每原子两次调用，已用 `max_atoms` 子采样上限约束；大实例
  应下调 `max_atoms` 或离线预算。
- **原子身份稳定**：以 `atom_key = str(e)` 为键（Phase 1 约定）；跨 refocus 一致，但不同进程/
  重命名会漂移——训练/评测须在同一构造流程内闭环。
