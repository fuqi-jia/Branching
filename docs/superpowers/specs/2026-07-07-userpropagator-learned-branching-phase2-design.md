# UserPropagator 学习分支 Phase 2（学习）—— 设计

## 0. 背景

Phase 1（PR #5）已打通管道:未训练 GNN 经 `LearnedDecidePropagator`/`PolicyDecider` 接管
z3 内部布尔决策,`solve_omt_with_decider` 达 `== native`,可测 rlimit/conflicts/decisions。
未训练 GNN 劣于 VSIDS(conflicts 40.5 vs 4.8)。**Phase 2 目标:训练使 learned-decide 优于
VSIDS-decide(等 `== native` 最优下,更低 rlimit/conflicts)。**

**可学习性已验证**:look-ahead 分数 = "假设某原子会强制多少其他原子"(经 `s.consequences`),
这是**子句共现图的直接函数**——正是 GNN 已见的特征(`build_bool_snapshot` 的 clause 图)。
故不同于 LIA(分离度需缺失的 LP 特征),此处标签与特征同构,imitation 应能收敛。

## 1. 目标与成功标准

- **成功**:trained learned-decide 在 rlimit/conflicts 上 **< VSIDS-decide**,且 `match=1`
  (`== native`)。多 seed 带误差棒(sweep 教训)。
- 不改 z3;复用 Phase 1 全部管道 + GNN(bool/phase head)+ ImitationTrainer + rl.py。

## 2. 组件

### 2.1 look-ahead 教师(`omt_branching/solver/lookahead.py`)

对候选原子 `a`(根状态 + 若干采样部分赋值),经 `z3.Solver.consequences`:

- `prop_T = |{被 a=true 强制的其他原子}|`、`prop_F = |{被 a=false 强制的其他原子}|`
  (减去平凡自蕴含 `a→a` / `a→¬¬a`)。
- **failed literal**:`consequences([a], atoms).result == unsat` 表示 a=true 不可行 → a 被强制为
  假(反之亦然)→ 记大哨兵分,phase = 可行侧。
- **打分(march 风格 product)**:`score(a) = (prop_T+1) * (prop_F+1)`;failed literal → 大哨兵。
  奖励**两侧都传播多**的原子(有效切分搜索)。
- **phase 目标**:传播更多的一侧(failed literal 时为可行侧)。
- 代价控制:按 clause-degree 预筛 top-K 原子;实例/状态子采样。

签名:
```python
def lookahead_scores(assertions, atoms=None, assignment=None, config=LookaheadConfig()
    ) -> tuple[dict[str, float], dict[str, bool]]   # (score_by_atomkey, phase_by_atomkey)
```

### 2.2 imitation 冷启动(`training_data.py` 扩展)

`build_lookahead_examples(instances, config) -> list[RankingExample]`:每实例根状态
`build_bool_snapshot` 建图,`lookahead_scores` 打标签,映射 atom_key → 图内 BOOL_VAR 局部索引,
填 `bool_target_scores` + `phase_targets`。复用 `ImitationTrainer`(已消费 bool ranking + phase)。

### 2.3 RL 微调(`rl.py` 扩展或新 `rl_decide.py`)

- **动作**:每次 refocus,GNN bool-head 分数 → softmax;每个 `decide` 从**缓存 softmax 采样**
  一个原子(记录 log-prob),而非 argmax。
- **奖励**:整体求解的 **−rlimit**(或 −conflicts);per-instance EMA baseline(已验证有效)。
- **更新**:REINFORCE,重跑记录的 refocus 图前向(同 `SolverInLoopRLTrainer` 结构)。
- imitation 冷启动后微调。

采样版 decider:`SamplingPolicyDecider`(PolicyDecider 的采样变体,记录 (refocus_graph,
sampled_key) 供 RL)。

### 2.4 实验(`examples/decide_branch.py` 扩展)

加 `--train/--iters/--epochs`:生成布尔结构整数 OMT → look-ahead imitation 冷启动 → RL 微调 →
三臂对比(**trained** policy)。多 seed 聚合 mean±std。断言 `match=1`。

## 3. 数据流

```
实例 → build_bool_snapshot(clause 图) → look-ahead 标签(consequences) → imitation(bool+phase head)
     → RL(采样 decide,reward=−rlimit) → trained policy → PolicyDecider → 三臂对比 vs VSIDS
```

## 4. 关键风险

- **look-ahead 开销**:`consequences` 是完整蕴含推理(非纯 unit-prop),每原子一次,贵 → top-K
  预筛 + 实例子采样;离线。
- **RL 噪声**:per-decision 采样 + 整体 reward,credit assignment 粗 → imitation 冷启动 +
  per-instance baseline 降险;RL 为次要,imitation 是主力(LIA 教训:特征可学时 imitation 有效)。
- **atom_key 稳定**:`str(atom)`,与 Phase 1 一致。

## 5. 复用 vs 新建

- **复用**:`build_bool_snapshot`、`PolicyDecider`、`solve_omt_with_decider`、`ImitationTrainer`、
  `SolverInLoopRLTrainer` 结构、`generate_bool_lia_dataset`。
- **新建**:`lookahead.py`、`build_lookahead_examples`、`SamplingPolicyDecider` + RL decide 适配。

## 6. 非目标

- 不改 z3;v1 不追求超越 native `Optimize`(比 learned vs VSIDS 同回路分支质量)。
- 不做纯 unit-prop 的自定义传播(用 `consequences` 作 look-ahead 代理即可)。
