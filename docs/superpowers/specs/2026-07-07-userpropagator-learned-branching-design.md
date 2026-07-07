# GNN 驱动 z3 内部布尔分支（UserPropagator）—— 设计

## 0. 背景与方向修正

原始目标(a):**改进 z3 内部的 CDCL 布尔文字/子句分支决策**。此前的 GOMT 外层 F-Split 与
LIA B&B 是**外层**分支,够不到 z3 内部决策(且 LIA B&B 与布尔分支关联弱)。

**关键发现(已用 spike 验证,`docs/ref/spike_userpropagator_decide.py`)**:z3 4.15.4 的
`UserPropagateBase.add_decide` + `next_split` 允许**外部接管 z3 内部布尔决策,且无需改 z3**。
证据:decide 回调被触发、next_split 被接受、正确性不变(baseline/ASC/DESC 均 sat),两个不同
策略在同一实例上 rlimit 不同(4694 vs 7880)——**我们真的控制了内部分支**。

**又一关键约束(已验证)**:`z3.Optimize` **不支持** propagator
(`'Optimize' object has no attribute 'solver'`)。故 OMT 的学习分支**只能**走
**Solver 回路**:这恰是 GOMT calculus 的退化形态(无 F-Split,纯 Better-cut 线性搜索),
于是 **GOMT 回路 = OMT 骨架(复用),UserPropagator decide = 内部布尔分支(新贡献)**,两层统一。

## 1. 目标与成功标准

- **v1 目标**:OMT(带目标)。学习分支在 z3 **内部布尔决策**层生效。
- **诚实、可隔离贡献的对比**:同一 OMT Solver 回路下三臂——
  - **learned-decide**(GNN 驱动 decide)
  - **VSIDS-decide**(不挂 propagator,z3 自身决策)——隔离"分支质量"的基线
  - **native `Optimize`**(不可控 skyline)
- **成功**:learned-decide 在 **rlimit/conflicts/decisions** 上优于 VSIDS-decide,且最优值
  `== native`(正确性)。native `Optimize` 更快是已知的(finding ⑤),但三臂中 learned vs
  VSIDS 用**同一回路**,差异即分支质量。
- **不改 z3**;复用 GNN(bool head + phase head)、抽取器、推理门控(`use_gnn` 回退)。

## 2. 架构

```
OMT(φ, obj):单个 z3.Solver 上的线性搜索回路
  I0 = Solve(φ);  循环: 加 Better-cut(obj≻I); Solve; 直到 UNSAT ⇒ I 为最优
                        └── 挂 LearnedDecidePropagator,接管内部布尔决策 ──┘
```

回路复用 `GOMTSolver` + `BaselineStrategy`(全 resolve,无 F-Split),后端换成**挂了
propagator 的 Solver**。GNN 不再做外层 F-Split,而是经 decide 在**内部**分支。

## 3. 组件(各自独立、可测)

1. **`propagator.py` — `LearnedDecidePropagator(z3.UserPropagateBase)`**
   - `add(atom)` 注册所有布尔原子;`add_fixed`/`push`/`pop` 维护当前部分赋值 trail。
   - `decide(t, idx, phase)`:从**缓存的优先级表**取最高优先级的**未定**原子,
     `next_split(atom, 0, phase)`;策略不自信(`use_gnn=False`)时**直接 return → 退回 VSIDS**。
   - **周期性 refocus**:每 N 次 conflict 重算优先级(调用 GNN)。计数经 `add_fixed`/statistics。
2. **快照构建(新 `omt_branching/solver/propagator_snapshot.py`,复用抽取器的 AST 遍历/线性
   分解辅助)**:从 **φ 的布尔骨架**(原子 + **子句/共现结构**,一次性算)+ propagator 的**动态
   状态**(当前赋值、conflict/decision 计数,每次 refocus 刷新)构造 `SolverSnapshot`。
   独立于 `extractor.py`(后者面向 GOMTState,保持不耦合)。
3. **优先级策略**:复用 `BranchingPolicyService.advise` → `activity_priors`(每原子分)+
   `phase_suggestions`。以稳定原子键映射 z3 atom ↔ `bool_var_id` ↔ 图内索引。
4. **`lookahead.py` — SAT look-ahead 教师**:对每个候选原子试探性赋值,按引发的单元传播/冲突
   数打分,产出 imitation 标签(布尔 head + phase head)。
5. **imitation 训练**:复用 `ImitationTrainer`,喂 look-ahead 标签。
6. **RL 训练器(改 `rl.py`)**:动作 = 每次 refocus 的优先级排序;奖励 = −整体求解 rlimit;
   REINFORCE(imitation 冷启动后微调)。
7. **`examples/decide_branch.py` + 指标**:rlimit/conflicts/decisions,learned vs VSIDS vs
   native,断言 `== native`。

## 4. 数据流

```
z3 内部搜索 → decide 回调 →(周期 refocus 时)从 propagator 状态建 SolverSnapshot
           → GNN → 每原子优先级 + 相位 → 缓存 → decide 取最高未定原子 next_split
```

## 5. 关键风险(诚实,源自 LIA 教训)

- **快照特征必须能预测"好决策"**:SAT 分支质量的关键结构是**子句图**(原子共现、子句状态)+
  赋值。若只用"原子无子句"的图,极可能学不动(=LIA 缺 LP 特征的翻版)。故**子句/共现图是必须的**,
  非可选。子句结构从 φ 的布尔骨架(自算 CNF / Tseitin skeleton)一次性构建。
- **原子跨 refocus 的身份稳定**:以 z3 `get_id()` / 结构键为准。
- **look-ahead 开销**:每原子传播计数,离线、可子采样。
- **RL 噪声**:per-refocus 排序为动作,imitation 冷启动降风险。

## 6. 分阶段(供 plan 拆解)

- **Phase 1 —— 管道 + 测量(不学习)**:propagator + 快照 + 策略(哪怕未训练)+ harness。
  成功 = learned-decide 的 OMT 求解达到 `== native` 最优,且能**测量** rlimit/decisions vs
  VSIDS。**先证明管道正确,再谈训练。**
- **Phase 2 —— 学习**:look-ahead imitation 冷启动 → RL 微调。成功 = learned-decide 在 rlimit
  上优于 VSIDS-decide(等最优)。

## 7. 复用 vs 新建

- **复用**:GNN(`policy.py` bool/phase head)、`InferenceEngine` 门控、
  `BranchingPolicyService`、`ImitationTrainer`、`GOMTSolver`+`BaselineStrategy`(退化 OMT 回路)、
  `SolverSnapshot` 契约、抽取器的 AST 遍历/线性分解。
- **新建**:`propagator.py`、look-ahead 教师、propagator→snapshot 构建(含子句图)、
  RL 动作/奖励适配、`decide_branch.py` 实验。
- **降级为消融/基础设施**:GOMT 外层 F-Split、LIA B&B 实验(作对照)。

## 8. 非目标

- 不改、不重编译 z3;不试图控制 `z3.Optimize` 内部(其不支持 propagator)。
- 不在 wall-clock 上正面超越 native `Optimize`(比的是同回路内 learned vs VSIDS 分支质量)。
- v1 不做 LRA 无界/epsilon 特殊处理(聚焦 LIA/有界 OMT,分支质量研究不依赖它)。
