# LIA B&B 学习分支实验 —— 设计（finding ⑤ 重设计）

针对 finding ⑤（native z3 单次 `Optimize` 即精确、开销远低于 GOMT 增量回路，外层分支无发挥
空间）的实验重设计：把实验搬到 **LIA 整数 OMT + plain 模式（check-sat 叶子）**，此时神经
F-Split **就是**优化器，学习分支相对启发式分支有真实的搜索规模差异。

## 1. 前提与核心洞察

**叶子 oracle 决定一切。** hybrid 模式下叶子是 `Optimize`（已求解整个 OMT），故分支冗余。
plain 模式下叶子是 `Solve`（check-sat），分支才真正驱动优化。LRA 因实数 plain 不终止而被迫
用 hybrid，恰好抹掉了分支的意义；**LIA 整数域 plain 模式 B&B 有限终止**，是学习分支的标准
战场（most-fractional/pseudocost → GNN）。

**比较范式：** 所有策略在 plain 模式下都饱和到**同一精确最优**（`== native`），差异只在
**搜索规模**（GOMT `splits` / `solve_calls`）。这是标准的 learn-to-branch 指标；native
z3-`Optimize` 的 rlimit 作为外部 skyline 参照（"没有原生 OMT 时的代价"）。

**诚实粒度说明（写入论文）：** 即便 check-sat 叶子也是一次完整 z3 SMT 求解（含 z3 内部
LIA 分支）。故本工作学习的是 **GOMT/理论组合层**的分支（如何切分框架驱动的搜索），**不**替代
z3 内部 MIP 节点；指标是 GOMT `solve_calls`/`splits`，比经典 MIP 节点数更粗，需明说。

## 2. 目标与成功标准

- 生成的 hard LIA 实例上，**不同分支策略的搜索规模明显分化**（否则实例太易，分支无从体现）。
- **neural（imitation(strong)+RL）在 solve_calls/splits 上优于启发式基线**（largest-domain /
  largest-coeff / random），并接近 strong-branching skyline。
- 所有策略最优值 **== native**（plain 模式饱和正确性；沿用 oracle 一致性护栏）。
- 不改 z3；不破坏既有 LRA/LIA 路径与 `tests/solver` 全绿。

## 3. 设计（组件）

### C1 —— hard LIA 生成器（`instance_gen.py`）

新增 `generate_hard_lia_instance` / `generate_hard_lia_dataset`。witness 驱动保证可行、有界：

- 采样整数 witness `w`（每维 `0..ub`，`ub` 取 20–50）。
- 盒约束 `0 <= x_j <= ub`。
- 若干 **packing** `Σ a_i x_i <= b`（`b` 取 witness 值 + 小 slack，使**紧**、LP 松弛易分数）与
  **covering** `Σ a_i x_i >= b`（`b` 取 witness 值 − 小 slack）；系数范围更宽（如 1–9）。
- 目标 `Σ c_i x_i`，`sense` 随机；`obj_coeffs` 记录。
- 硬度旋钮：`n_vars`、`n_constraints`、系数范围、slack 紧度。
- 自检：`z3` 校验 SAT（witness 即证），且节点数在不同策略间有差异（实现期经验校验）。

### C2 —— 整数 strong-branching 专家（`strong_branch.py` 扩展）

新增 `strong_branch_numeric_scores(extraction, phi, objective, sense, backend, config)`：对每个
**整数数值变量**（`handle.is_integer` 且域跨度≥1），取中点 `m`，一层前瞻
`v_lo = optimize(φ∧x≤m)`、`v_hi = optimize(φ∧x≥m+1)`，用 **objective-separation score**：

```
score(x) = |v_lo − v_hi|         # 两侧都可行
score(x) = SENT (大哨兵)          # 恰有一侧 UNSAT → 该分支直接剪掉半个域，有效
```

即奖励"最能把可达目标分到两侧"的变量（一侧出 incumbent、另一侧被 Better 割廉价剪）。与 LRA
布尔专家同一 `|·|` 家族。**不用 MIP product score**：本设置在**整数域中点**切分、且 z3 不暴露
LP 松弛，product score 对"最优落在某子一侧"的变量退化为 ≈0（反而避开真正有用的分支），
objective-separation 更贴合且可测。整数域切分**一侧 UNSAT = 有效剪枝**（非 LRA 的"被蕴含无进展"），
故记大哨兵而非 0。`oracle_numeric_choice_sb(instance)` = argmax score。

方向标签 `branch_up`：先探保留更优目标的一侧（MAX→`v_hi≥v_lo` 则先探高侧）。

### C3 —— 启发式基线策略（`strategy.py`）

新增 `NumericHeuristicStrategy(problem, mode)`，`mode ∈ {largest_domain, largest_coeff, random, strong}`：

- `largest_domain`：现 `BaselineStrategy` 行为（最大域中点）。
- `largest_coeff`：最大 |目标系数| 变量中点切分。
- `random`：随机整数变量（`rng` 由 problem/seed 派生，确定性）。
- `strong`：argmax C2 的 product score（昂贵 skyline）。

均只对整数变量域切分（复用 `_numeric_split`），plain 模式；soundness 与既有一致。

### C4 —— imitation 数值标签接入 strong branching（`training_data.py`）

`build_imitation_example` 增参 `numeric_expert ∈ {"coeff"(默认), "strong"}`。`"strong"` 时数值
head 的 `int_target_scores` 用 C2 的 product score（经 `numeric_handles`→局部索引映射），
`int_dir_targets` 用其方向。默认 `"coeff"` 保持既有行为（不破坏 LRA/rl_demo）。
`policy_numeric_choice`（已存在）用作数值 head 准确率查询；准确率参照改用
`oracle_numeric_choice_sb`。

### C5 —— 实验脚本（`examples/lia_branch.py`）

并列于 `rl_LRA.py`，**plain 模式**：

1. `generate_hard_lia_dataset` 训练/测试集。
2. imitation 冷启动（`numeric_expert="strong"`）。
3. RL 微调（复用 `SolverInLoopRLTrainer`；`f_sat_mode="plain"`；RLRecordingStrategy 已对整数
   变量走数值 head）。目标：更少 `solve_calls`/`splits`。
4. 对比表：native / random / largest_domain / largest_coeff / strong / neural 的均值
   `solve_calls` / `splits` / `steps`，并断言各策略最优值 `== native`。

## 4. 非目标

- **不做 most-fractional / pseudocost 基线**：需 LP 松弛值，z3 公开 API 不暴露（诚实局限，
  已在 API.md 记录）。
- 不改 z3、不重编译；不试图在 wall-clock 上正面超越 native `Optimize`（同叶子、比搜索规模）。
- 不引入 critic / 新网络结构。
- 不动 LRA 路径（并行实验，独立脚本）。

## 5. 测试与风险

- 测试（`tests/solver/`）：C1 生成器可行性 + 节点数分化；C2 objective-separation：影响目标的
  变量 outrank 不影响目标的变量、一侧 UNSAT 记大哨兵；C3 各 mode 确定性且 soundness（== native）；
  C4 `numeric_expert="strong"` 产出非空数值标签；`lia_branch` 小规模冒烟。
- **风险①**：实例太易 → 节点数不分化。缓解：硬度旋钮 + 生成期经验校验（strong 与 random 的
  solve_calls 需拉开差距）。
- **风险②**：strong 专家标签慢（每变量 2 optimize + 1 个 v*）。缓解：`max_vars` 上限 + 变量
  预筛（|obj coeff| 质量）+ 实例子采样。
- **风险③**：plain LIA 某些实例节点爆炸/超 `max_steps`。缓解：`max_split_depth` 预算 +
  `max_steps` 后备（anytime 兜底），并在指标中标注未饱和比例。
