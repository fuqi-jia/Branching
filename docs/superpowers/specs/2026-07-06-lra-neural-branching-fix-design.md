# LRA 神经分支学习管线修复 —— 设计

本设计修复 `examples/rl_LRA.py` 所暴露的**神经分支策略效果差**问题。诊断表明主因是
**学习管线接线错误**（训练/评测作用在一个不参与求解的 head 上），而非模型容量不足。

## 1. 背景与诊断

LRA 实例为**纯实数**（`instance_gen.py` 用 `z3.Real`）。求解时实数变量**不做域二分**
（连续优化交给 z3 `Optimize` 的 hybrid 叶子），故 GNN 只在**布尔原子结构**上分支——
即只用 **bool head**（`bool_branch_scores` / `phase_logits`）。

现状测得：`Neural top-1=0.45`、rlimit 804M / solve_calls 16 / splits 14.38 / gap 0.041 /
exact 0.75，相对 baseline（1049M / 21.5 / 0 / 0.045 / 0.75）仅小幅领先，远差于 native
z3（48M / 1 / 0 / 0 / 1.0）。

根因（按严重度）：

1. **imitation 训练错 head（冷启动作废）。** `build_imitation_example` 只写
   `int_target_scores` / `int_dir_targets`（数值 head）；`ImitationTrainer` 对无标签的 head
   不计损失（`trainer.py` 的 `_ranking_loss`/`_bce_loss` 返回 `None`）。而 LRA 求解只用 bool
   head → 30 轮 imitation 对实际求解**零贡献**，bool head 从随机初始化仅靠 RL 微调。
2. **准确率量错 head。** `branch_accuracy → policy_numeric_choice` 读 `masked_numeric_probs()`
   （数值 head），与求解决策路径无关。
3. **专家不对齐。** `oracle_numeric_choice` = 最大 |目标系数| 的实数变量，与"该分支哪个布尔
   原子"无关；布尔分支目前**没有专家**。
4. **RL 信号弱且未归一化。** 默认 `--iters 1`；优势用**单个全局**移动平均 baseline，而
   episode 回报被 `terminal = -log1p(rlimit)`（≈15–22，随**实例**而非动作变化）主导 →
   `G - baseline` 被实例间代价方差淹没，逐步 incumbent 奖励（×0.1）几不可见。

**范围外（本设计不处理）：** 根因 5——native z3 单次 Optimize 即精确求解、GOMT 增量回路
天生更贵、外层布尔分支难以超越 native——属**实验设计/框架**层问题，另行讨论。

## 2. 目标与成功标准

- **诚实且自洽的指标**：准确率量的是**实际驱动求解的 bool head**。
- **imitation 真正冷启动 bool head**：训练后 bool-head 与 strong-branching 专家的 top-1
  一致率显著高于随机初始化。
- **RL 稳定收敛**：per-instance baseline 后，优势方差显著下降；neural 在 rlimit / gap 上
  **清晰且正确地优于 baseline**（不再是接线巧合）。
- **不破坏 z3**（仍只用 pip wheel，不改系统 z3）；**不破坏 LIA 路径**（`rl_demo.py` /
  数值 head 仍可用）；`tests/solver` 全绿。

## 3. 设计（五部分）

### Part 1 —— strong-branching 布尔专家（新文件 `omt_branching/solver/strong_branch.py`）

任务对齐的教师，在**根状态**用现有 `Z3SnapshotExtractor` + `Z3Backend` 计算：

- 取候选原子 `a ∈ extraction.atom_handles`。每实例缓存 `v* = optimize(φ, obj)`。
- 一层前瞻：`v_a = optimize(φ∧a, obj)`、`v_na = optimize(φ∧¬a, obj)`。
- **打分 = 目标分离度**：两子都可行时 `score(a) = |v_a − v_na|`（最能把目标分到两侧的原子，
  一侧出 incumbent 后另一侧被 Better 割廉价剪掉）；**恰有一侧 UNSAT** 的原子（近乎被 φ 蕴含）
  **不产生真实进展 → 低分**；两侧皆 UNSAT → 无关。（该方向在 TDD 下经验校验：不可天真地把
  "UNSAT 子 = 高分"。）
- **phase 目标** = 先探保留更优目标的一侧（MAX→更高 `v`；MIN→更低）。
- `oracle_bool_choice(instance) → 原子 bool_var_id` = argmax score。
- 代价控制（`StrongBranchConfig`）：`max_atoms`（按 occurrence/系数质量预筛原子）、可选实例
  子采样；`v*` 每实例只算一次。UNSAT/异常一律安全降级为跳过该原子。

签名（拟）：
```python
def strong_branch_labels(instance, config=StrongBranchConfig()
    ) -> tuple[dict[Hashable, float], dict[Hashable, bool]]:  # (scores, phases) 按 bool_var_id
def oracle_bool_choice(instance, config=StrongBranchConfig()) -> Optional[Hashable]
```

### Part 2 —— 布尔 head imitation（`training_data.py`）

`build_imitation_example` 增补 bool 标签：把专家的 `bool_var_id → score/phase` 经
`graph.id_maps[NodeType.BOOL_VAR]` 映射到**图内局部 BOOL_VAR 索引**，填入
`RankingExample.bool_target_scores` / `phase_targets`。数值标签保留（LIA 兼容）。trainer 已按
head 分别累加损失，改动是**增量**的。新增：

```python
def policy_bool_choice(policy, instance) -> Optional[Hashable]  # 读 masked_bool_probs() 的 top-1
```

`build_imitation_examples` 增参数控制是否用 strong-branching（默认用；提供廉价关闭以便冒烟）。

### Part 3 —— 诚实的准确率（`rl_LRA.py`）

`branch_accuracy` 改为量 **bool head vs `oracle_bool_choice`**：对每个测试实例，
`policy_bool_choice(policy, inst) == oracle_bool_choice(inst)` 的比例。移除数值-head 准确率
（或保留为 LIA-only 的独立函数，不在 LRA 主输出中）。

### Part 4 —— 强化 RL（`rl.py`）

三处定向改动：

1. **per-instance EMA baseline**：以实例键（训练主循环的 `instance` 索引）维护
   `dict[key → ema]` 取代单个全局 `self._baseline`。`update()` 接收实例键；优势
   `G - baseline[key]`；更新后 EMA 该键。**核心修复**，无额外求解调用。
2. **奖励归一化**：逐步 incumbent 提升按实例目标幅度 `(v*−v₀)`（或 episode 内首末 incumbent
   跨度）缩放到 O(1)；终局 rlimit 代价按 per-instance 中心化（EMA 已隐式中心化，故此项主要是
   让逐步项与终局项**尺度可比**）。相关系数进 `RLConfig`，给合理默认。
3. **默认迭代数**：`rl_LRA.py` 的 `--iters` 默认 1 → 3。

> 备选（写入 spec 备注，不作默认）：终局代价以**参照 rollout**（如 baseline 策略在同实例的
> rlimit）中心化。默认选 per-instance EMA，因其**不引入额外求解**。

### Part 5 —— 接线、demo、测试

- `omt_branching/solver/__init__.py` 导出 `strong_branch_labels` / `oracle_bool_choice` /
  `StrongBranchConfig` / `policy_bool_choice`。
- 更新 `rl_LRA.py` 的 import 与准确率/训练调用。
- TDD 测试（`tests/solver/`）：
  - `test_strong_branch.py`：两子皆可行的分离原子 **outrank** 近乎被蕴含（一侧 UNSAT）的原子；
    小实例上 `oracle_bool_choice` 与手算一致；`v*` 与 `solve_native` 一致。
  - `test_training_data.py`（扩展）：`build_imitation_example` 产出非空 bool 标签；imitation 后
    `policy_bool_choice` 的 top-1 一致率相对随机初始化**上升**。
  - `test_rl.py`（扩展）：per-instance baseline 下优势方差 < 全局 baseline；`update()` 正确按键
    维护 EMA。
  - `test_demo_smoke`：`rl_LRA --train 6 --test 6 --iters 1` 端到端跑通并打印 bool 准确率。

## 4. 风险与缓解

- **标签生成慢**（strong branching 每原子 2 次 Optimize）：`max_atoms` 预筛 + 实例子采样；`v*`
  缓存；冒烟用小规模。
- **打分公式方向**（UNSAT 子的处理）：TDD 经验校验，必要时改用 `min(退化)` 或成本型分数。
- **不破坏 LIA**：数值 head 标签与 `rl_demo.py` 路径保持不变；测试覆盖。

## 5. 非目标

- 不改 z3、不重编译；不改 GOMT calculus 的 soundness 结构。
- 不处理根因 5（框架相对 native 的固有代价 / 实验重设计）——另行讨论。
- 不引入 critic/value 网络（per-instance baseline 足以，YAGNI）。
