# 学习布尔分支：困难 SMT(LIA) theory-atom（SAT 结果的 SMT 扩展）—— 设计

## 0. 背景

SAT 正结果（`docs/findings-sat-learned-branching.md`）：learned-decide 在困难 3-SAT 上 conflicts
< VSIDS（~28%）。**扩展到 SMT theory-atom**：原子改为**线性算术比较原子**（LIA），单次 SMT 可
满足性检查，reward=−conflicts。闭合到 OMT/SMT 论文目标（对理论原子分支）。

**可行性已 probe**：紧的随机 SMT(LIA)（多析取线性原子、小域）附 propagator → **610–786 conflicts**
（默认 z3 仅 43–72），分支于 **106 个理论原子**。witness-驱动的 hard_bool_lia 太易(0–6 conflicts)
——**紧性是关键**。

## 1. 目标与成功标准

- **成功**：trained learned-decide 在困难 SMT(LIA) 上 conflicts **< VSIDS-decide**，trained≫untrained。
  两臂均附 propagator（关 z3 预处理→纯 CDCL），同一搜索仅决策不同 = 隔离分支质量。多 seed。
- 不改 z3；复用 SAT 全部管道。LRA 为后续（LIA 已 probe）。

## 2. 架构（几乎全复用 SAT）

```
单次 z3.Solver().check(assertions=SMT(LIA) 子句)，两臂均附 LearnedDecidePropagator（关预处理）
  VSIDS-decide = decider 恒 defer；learned-decide = GNN(理论原子分支)
  指标 = conflicts
```

`solve_sat_with_decider(assertions, atoms, decider_factory)`、`build_bool_snapshot`、
`lookahead_scores`、`build_lookahead_examples_sat`、`DecideRLTrainer.collect_sat/train_sat`
**均 atom-type-agnostic，直接复用**（理论原子即原子）。

## 3. 组件

### 3.1 困难 SMT(LIA) 生成器（`sat_instances.py`，新）

```python
def generate_hard_smt_lia(n_vars=8, n_disj=30, k=3, ub=6, chi=4, seed=0) -> tuple[list, list]:
    # 紧随机 SMT(LIA)：n_vars 整数变量 + 盒约束；n_disj 个 k 元析取(Σc x <=/>= b，系数 [-chi,chi]，
    # 域小[0,ub])。返回 (atoms=collect_atoms(clauses) 的理论原子, clauses)。紧性 -> 数百 conflicts。
```

### 3.2 复用 SAT harness

- `solve_sat_with_decider(clauses, atoms, decider_factory)` 直接用（理论原子）。
- `build_bool_snapshot(clauses)`：`collect_atoms` 已识别比较原子；子句共现图 + 度/极性特征。
- `lookahead_scores(clauses, atoms)`：`consequences` 对理论原子做理论传播计数（比 SAT 更丰富）。
- `build_lookahead_examples_sat([(atoms, clauses), ...])`、`DecideRLTrainer.train_sat`（reward=−conflicts）。

### 3.3 verify-then-extend（可学习性护栏 + 理论特征应急）

- **先验证**：SMT look-ahead imitation 的 branch 损失是否下降（护栏测试）。子句图 = boolean 传播结构。
- **若冻结**（theory 传播依赖**共享变量**而非仅共享子句——LIA 教训）：给快照加 `TheoryAtomInfo`
  （`var_coeffs`/`rhs`）+ 数值变量节点 + 原子↔变量边（复刻 `extractor.py` 的线性分解），使 GNN 见
  "哪些原子共享变量"（理论传播结构）。**仅在冻结时加**（YAGNI；SAT 用子句图已足）。

### 3.4 实验（`examples/smt_branch.py`，新）

困难 SMT(LIA) → imitation 冷启动 + RL(−conflicts) → learned-decide vs VSIDS conflicts，多 seed mean±std。
（可与 `sat_branch.py` 合并为 `--theory sat|smt` 一个脚本；本 spec 用独立脚本，YAGNI 后合。）

## 4. 关键风险

- **可学习性（LIA 教训）**：理论传播依赖共享变量；子句图或不足 → verify-then-extend（§3.3）。
- **紧性 vs 可解**：太紧 → UNSAT（无 imitation 标签，如 PHP）；太松 → 太易（无 headroom）。生成期
  经验校验 conflicts 落在数百（有 headroom 且 SAT）。
- **look-ahead 开销**：`consequences` 对理论原子含理论推理，较贵 → 子采样/top-K（`LookaheadConfig.max_atoms`）。

## 5. 复用 vs 新建

- **复用**：`solve_sat_with_decider`、`build_bool_snapshot`、`lookahead_scores`、
  `build_lookahead_examples_sat`、`DecideRLTrainer.collect_sat/train_sat`、`LearnedDecidePropagator`、
  `PolicyDecider`、GNN。
- **新建**：`generate_hard_smt_lia`、SMT 可学习性护栏、`examples/smt_branch.py`；**应急**：理论原子
  快照结构（仅冻结时）。

## 6. 非目标

- 不改 z3；v1 用 LIA（已 probe）；LRA/mixed 为后续（机制相同）。
- 紧随机 SMT 用 SAT 实例（有 imitation 标签）；UNSAT SMT 为后续（需 UNSAT-capable 教师）。
