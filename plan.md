# 使用图神经网络训练 OMT 分支选择策略方案

## 1. 背景与目标

OMT（Optimization Modulo Theories）在 SMT 的可满足性搜索之上增加目标函数优化。典型 lazy OMT 求解器由两个部分交替工作：

- CDCL/DPLL(T) 布尔搜索枚举满足理论一致性的 truth assignment。
- 理论优化器（例如 LRA/LIRA minimizer、MILP/ILP branch-and-bound、MaxSMT engine）在当前 assignment 下优化目标值，并用新的 bound/cut 继续收缩搜索空间。

本方案目标是训练一个图神经网络（GNN）分支选择策略，用来在 OMT 搜索中替代或增强现有 branching heuristic，使求解器在保持完备性的前提下减少搜索节点、冲突数和 wall-clock time。

优先研究对象：

- OMT(LRA/LIA/LIRA) 或 OMT(LRA/LIA + Boolean structure)。
- 单目标优化优先，后续扩展到 lexicographic、Pareto、box objectives。
- 分支决策优先从“选择哪个布尔抽象变量/理论原子做 SAT decision”做起，再扩展到“整数变量 branch-and-bound 变量选择”和“phase/value 选择”。

## 2. 相关论文与可借鉴结论

### 2.1 OMT / SMT 优化

1. Sebastiani 与 Tomasi，Optimization Modulo Theories with Linear Rational Costs  
   早期系统化 OMT(LRA) lazy 架构：SMT 搜索产生 assignment，理论 minimizer 优化 objective，并通过 bound tightening 迭代。

2. Sebastiani 与 Trentin，Pushing the Envelope of Optimization Modulo Theories with Linear-Arithmetic Cost Functions，TACAS 2015  
   将 OptiMathSAT 扩展到 LIRA、混合整数/有理优化、多目标和增量 OMT。重要启发是 OMT(LIRA) 内部会复用 LP/ILP minimization 与 branch-and-bound，因此 MILP branching 的经验可迁移。

3. Sebastiani、Tomasi、Trentin，OptiMathSAT: A Tool for Optimization Modulo Theories，JAR 2018  
   说明 OptiMathSAT 支持 linear search、binary search、adaptive search、incremental OMT、多目标优化等。对本课题最重要的是：OMT 搜索有多个可控 branching 层次，包括 SAT decision、优化 bound search、LIRA branch-and-bound。

4. Bjørner、Phan、Fleckenstein，νZ - An Optimizing SMT Solver  
   Z3 的优化模块 νZ/Z3Opt 将 MaxSMT、OptSMT、线性算术优化组合在一个 portfolio 中。Z3 内部对整数变量会在 Simplex 得到非整数值时创建 bound split，例如 `x <= floor(v)` 与 `x >= ceil(v)`。

5. On Optimization Modulo Theories, MaxSMT and Sorting Networks  
   指出 OMT+PB/MaxSMT 中相同权重 soft clauses 会导致 truth assignment 对称性和组合爆炸，sorting networks 能缓解。对 GNN 的启发是：图表示需要显式编码 soft clause 权重、目标贡献、对称结构。

6. Exploiting Partial-Assignment Enumeration in Optimization Modulo Theories，2025  
   指出 OMT 通常在 total assignment 上调用理论优化器，但 partial assignment 可扩大理论优化器一次看到的可行区域，提高 anytime 解质量和最终效率。对 GNN 的启发是：分支策略不应只模仿“尽快补全 assignment”，也要学习保留有价值的自由度。

7. Ashera: Neural Guided Optimization Modulo Theory，2023  
   使用神经引导的 warm start 与 logical neighborhood search 加速 OMT。虽然重点不是 branching，但说明 OMT 中“逻辑骨架 + 理论优化”的结构适合神经方法介入。

### 2.2 MILP 分支选择

1. Gasse et al., Exact Combinatorial Optimization with Graph Convolutional Neural Networks，NeurIPS 2019  
   将 MILP 表示为变量-约束二部图，用 imitation learning 近似 strong branching。核心可迁移结论：GNN 能处理 variable-size instance，并减少人工特征工程。

2. Gupta et al., Hybrid Models for Learning to Branch，NeurIPS 2020  
   GNN 每次 branch 都调用成本较高，因此可在 root 或少数关键节点运行 GNN，再用轻量 MLP/缓存 embedding 处理深层节点。对 OMT 很重要，因为 SMT branching 频率远高于 MILP B&B。

3. Learning to Branch in Combinatorial Optimization with Graph Pointer Networks，2023  
   在图表示上结合 pointer mechanism、global features、historical features，预测候选变量。对 OMT 的启发是：候选分支集合动态变化，pointer/ranking 比固定维度分类更合适。

### 2.3 SAT / SMT 分支选择

1. VSIDS / EVSIDS  
   CDCL 求解器的经典 branching heuristic，对冲突中出现的变量 bump activity，并随时间 decay。经验上能聚焦公式社区结构中的高中心性 bridge variables。

2. CHB 与 LRB  
   将 SAT branching 视为在线优化或 multi-armed bandit，基于变量带来冲突/学习子句的能力更新分数。对 OMT 的启发是：GNN 可以与在线反馈结合，而不是完全离线固定。

3. NeuroSAT  
   使用 literal-clause message passing 学习 SAT 表示。作为端到端求解器不适合直接替代 CDCL，但其图结构是后续神经分支方法基础。

4. NeuroCore  
   用 GNN 预测 UNSAT core 变量，并周期性 refocus MiniSat/Glucose/Z3 的 VSIDS activity。对 OMT 最实用的启发是：不要每个 decision 都调用 GNN，可周期性重置或融合 activity，以降低开销并保留求解器完备性。

5. Graph-Q-SAT  
   用 value-based RL 训练 GNN 分支策略，尤其改善 solver warm-up 阶段。对 OMT 的启发是：先在早期分支或 root neighborhood 使用 RL/GNN，避免全程高频神经调用。

6. SMT theory-aware branching / lookahead  
   SMT 求解器可用 theory solver 信息偏置 SAT branching。OMT 是更强的 theory-aware 场景，因为 objective bound、reduced cost、LP relaxation、unsat core、theory propagation 都能成为分支特征。

## 3. OMT 分支选择的建模

### 3.1 可学习的分支对象

建议按风险分三阶段推进。

阶段 A：布尔抽象层 decision variable 选择  
动作是在当前 CDCL trail 下选择一个未赋值布尔变量，即某个理论原子、soft constraint 或 CNF 变量。该阶段最容易接入 VSIDS，与 NeuroCore 类似，可用 GNN 分数周期性 refocus activity。

阶段 B：phase/value 选择  
对选中的布尔变量预测 polarity，例如理论原子 `x <= c` 优先取真还是假，soft clause 优先满足还是放弃。这里可借鉴 phase saving、LSIDS 和 OMT bound tightening，把“更可能改善 objective 或导致快速冲突”的方向作为标签。

阶段 C：理论整数变量 B&B 选择  
当 LIA/LIRA solver 发现整数变量取分数值时，选择哪个整数变量 split，以及 split direction。这里可借鉴 MILP strong branching、pseudo-cost、reliability branching，用 GNN 近似昂贵的 strong branching。

### 3.2 保持完备性的原则

GNN 只改变分支顺序和 activity/phase bias，不删除分支、不跳过理论一致性检查、不替代理论优化器的正确性判断。若模型超时、置信度低或输入超出训练分布，回退到原生 VSIDS/LRB/SMT/MILP heuristic。

## 4. 图表示设计

### 4.1 异构图节点

使用异构图比单一 CNF 图更适合 OMT：

- Literal/Boolean variable 节点：CNF literal、理论原子抽象变量、soft indicator。
- Clause 节点：原始 clause、learned clause、blocking clause、bound tightening clause。
- Theory atom 节点：例如 `a^T x <= b`、`x = y + c`、PB constraint、array/string/bit-vector 原子。
- Numeric variable 节点：LRA/LIA/MILP 中的实数/整数变量。
- Objective 节点：目标函数、当前 incumbent、当前 lower/upper bound、lexicographic priority。
- Constraint row 节点：线性约束行，用于 MILP 风格二部图。
- Search state 节点：当前 depth、restart count、conflict count、best objective、gap、time budget。

### 4.2 边类型

- literal-in-clause：literal 与 clause 的正/负出现。
- atom-abstracted-by：theory atom 与对应 Boolean variable。
- variable-in-atom：数值变量出现在理论原子中，边特征为 coefficient、sign、normalized coefficient。
- variable-in-objective：数值变量在目标函数中的 coefficient。
- clause-derived-from：learned/bound clause 与相关 conflict/core 的弱关联。
- bound-relates-variable：变量当前 lower/upper bound、是否由 incumbent 或 theory propagation 产生。
- soft-weight：soft clause/indicator 与 objective 的权重边。

### 4.3 节点与全局特征

布尔/文字特征：

- 当前赋值状态、decision level、是否 candidate、VSIDS/EVSIDS activity、LRB/CHB score、phase saved value。
- 出现次数、正负极性比例、是否来自 soft constraint、是否属于 recent learned clause。
- 距离 objective/bound clauses 的图距离或消息传递后隐式表示。

理论原子/数值特征：

- 原子类型（<=、>=、=、PB、BV compare 等）。
- 当前 slack、violation、LP relaxation value、reduced cost、basis status。
- atom 对 objective bound 的潜在影响，例如取真/假后是否收紧 `obj < incumbent`。

OMT 全局特征：

- 当前 incumbent objective、best bound、relative gap、是否 unbounded 检测阶段。
- linear/binary/adaptive search mode。
- 当前 trail 长度、decision level、conflict rate、restart 周期。
- 最近一次 theory conflict、unsat core size、bound tightening 成功幅度。

## 5. 标签与训练数据

### 5.1 数据采集

在 Z3/νZ 或 OptiMathSAT 类架构中增加日志插桩：

- 每次需要 SAT decision 时，导出当前候选变量集合、局部图快照、solver state。
- 每次 LIA/LIRA branch-and-bound 需要整数 split 时，导出候选整数变量和 LP relaxation 状态。
- 记录分支后短期反馈：是否快速产生 conflict、传播数量、learned clause LBD、objective 改善、bound 改善、子树节点数、最终 solve time。
- 对小中型 instance 可运行强专家生成标签；对大型 instance 使用在线反馈或离线 replay。

建议数据源：

- SMT-LIB optimization benchmarks、OptiMathSAT/νZ 示例与 formal verification OMT benchmarks。
- MaxSMT/OMT+PB benchmarks，覆盖 soft clause 权重和对称性。
- 从 MILP benchmark 转写为 OMT(LIRA)，保留线性整数结构。
- 随机生成 scheduling、resource allocation、multi-agent routing、TSP-with-logic 等 OMT family，方便控制训练/测试分布。

### 5.2 专家标签

阶段 A 可使用多种标签混合：

- Strong branching 标签：对 top-k 候选变量临时试探两侧分支，估计节点数、冲突数或 bound improvement，选择综合收益最高者。
- Oracle subtree 标签：对小 instance 离线求解完整树，标注能最小化剩余 solve time/节点数的变量。
- Heuristic distillation：蒸馏 VSIDS/LRB/CHB 与 theory-aware heuristic 的选择，用作冷启动。
- OMT objective-aware 标签：优先选择能产生更好 incumbent、缩小 objective gap、或更快证明 `obj < incumbent` 不可行的变量。

阶段 B 标签：

- 使用 phase saving 作为弱标签。
- 用双分支 probing 比较真/假方向的短期 objective improvement、propagation、conflict quality。
- 对 soft clauses，可标注“满足高权重 soft clause”或“快速证明无法满足”的方向。

阶段 C 标签：

- MILP strong branching：试探每个 fractional integer variable 的上下分支，依据 LP bound improvement、infeasibility、objective gap 缩小打分。
- pseudo-cost/reliability branching 作为低成本标签。

### 5.3 训练目标

建议采用 ranking 而不是硬分类：

- 候选变量 ranking loss：top-k cross entropy、pairwise hinge、ListNet/ListMLE。
- 分支方向 binary cross entropy。
- 辅助任务：预测 conflict probability、objective improvement、bound tightening、是否属于 unsat core、subtree size。
- 多目标损失：`L = L_branch + λ1 L_phase + λ2 L_value + λ3 L_core + λ4 L_gap`。

## 6. 模型结构

### 6.1 主模型

采用异构 message passing GNN：

- 输入为 OMT 异构图，边类型使用 R-GCN、HGT 或 Graph Transformer 编码。
- 对 literal/Boolean candidate 输出 branching score。
- 对 literal polarity 输出 phase score。
- 对整数变量候选输出 B&B score。
- 全局 state embedding 通过 attention/readout 注入每个候选节点。

候选选择：

- 使用 pointer/ranking head，只在当前未赋值候选变量上 softmax。
- 支持 mask：已赋值、eliminated、非 decision candidate 的变量不可选。

### 6.2 低开销部署模型

参考 NeuroCore 与 Hybrid Models：

- Root/重启后/固定冲突间隔调用完整 GNN，生成全局 activity prior。
- 普通 decision 使用融合分数：
  `score(v) = alpha * normalized_solver_activity(v) + beta * gnn_score(v) + gamma * theory_score(v)`。
- 深层节点使用轻量 MLP，输入为缓存的 root embedding + 当前动态 solver features。
- 当图太大时只抽取候选变量 k-hop 子图、recent conflict 子图、objective-relevant 子图。

## 7. 与求解器集成

### 7.1 最小可行集成

首选做 VSIDS refocus，而不是直接替换 decision procedure：

1. 在 solver 初始化、restart 后或每 N 次 conflict 后构建图。
2. GNN 给所有候选布尔变量输出 prior。
3. 将 prior 写入或混合进 SAT activity。
4. 后续若干 decision 仍由原生 priority queue/VSIDS 选择。

优点：

- 改动小，保持 CDCL 数据结构和性能。
- 即使 GNN 预测一般，也会被后续冲突 bump/decay 修正。
- 可直接对照 NeuroCore 的周期性 refocusing 经验。

### 7.2 OMT-aware 增强

在基础 refocus 上增加 OMT 特征：

- objective proximity：变量/原子到 objective 和 incumbent cut 的结构距离。
- bound sensitivity：变量分支后对 lower bound/upper bound 的历史影响。
- theory conflict participation：变量是否频繁出现在 theory lemma 或 unsat core。
- partial assignment preservation：惩罚过早固定与 objective 优化无关、但会缩小理论 minimizer 搜索空间的变量。

### 7.3 LIA/LIRA B&B 集成

对整数变量分支：

1. 从 LP relaxation 获取 fractional integer variables。
2. 构建变量-约束二部图，复用 MILP GNN。
3. 输出 split variable 分数。
4. direction 可由 pseudo-cost、objective coefficient 或 phase head 决定。
5. 若 GNN 推理超过预算，回退到原生 `int_branch` / pseudo-cost。

## 8. 训练流程

### 8.1 阶段一：离线 imitation learning

1. 收集小中型 OMT instances。
2. 用原生 solver + probing/strong branching 生成专家标签。
3. 训练 GNN ranking policy。
4. 在同分布和放大规模 instance 上测 generalization。

成功标准：

- top-1/top-5 imitation accuracy 高于 heuristic baseline。
- 在 solver replay 中减少 branch count/conflict count。

### 8.2 阶段二：solver-in-the-loop fine-tuning

1. 将 GNN 集成到 solver 的 refocus 模式。
2. 使用真实求解反馈收集轨迹。
3. 用 DAgger 或 off-policy RL 修正分布偏移。
4. 奖励函数以 wall-clock time、节点数、objective gap area-under-curve 为主。

推荐 reward：

- 完整求解：`-log(1 + solve_time)` 或 `-nodes`。
- anytime OMT：固定时间内 incumbent improvement 与 final gap。
- 局部 step：bound improvement、conflict quality、propagation count、subtree prune。

### 8.3 阶段三：混合策略自动调参

训练或搜索 `alpha/beta/gamma` 融合权重，并按 instance family 或 solver phase 自适应：

- warm-up 阶段提高 GNN 权重。
- 冲突学习稳定后提高 VSIDS/LRB 权重。
- objective gap 停滞时提高 theory/objective-aware 权重。

## 9. 评测设计

### 9.1 Baseline

- 原生 Z3/νZ 或 OptiMathSAT。
- VSIDS/EVSIDS、phase saving。
- LRB/CHB，如可在目标 solver 中实现。
- MILP B&B baseline：pseudo-cost、strong branching、reliability branching。
- NeuroCore 风格 SAT-only GNN，不含 OMT 特征，用于证明 OMT-aware 特征的增益。

### 9.2 指标

- solved instances 数量。
- wall-clock time，PAR-2/PAR-10。
- SAT decisions、conflicts、restarts、learned clauses。
- theory conflicts、theory propagations、unsat core size。
- OMT iterations、incumbent 更新次数、objective bound tightening 次数。
- B&B nodes、LP solves、integrality gap。
- anytime 指标：objective gap over time 的面积。
- GNN overhead：推理次数、平均推理耗时、占总时间比例。

### 9.3 消融实验

- SAT CNF 图 vs OMT 异构图。
- 无 objective 特征 vs 有 objective 特征。
- 每次 decision 调用 GNN vs 周期性 refocus。
- GNN-only vs GNN + VSIDS 混合。
- imitation only vs solver-in-the-loop fine-tuning。
- total assignment 导向标签 vs partial assignment/objective-aware 标签。

## 10. 工程实现路线

### Milestone 1：日志与数据集

- 在 solver decision 点记录候选变量、trail、activity、phase、clause graph。
- 在 OMT optimizer 记录 objective、incumbent、bound、theory minimizer 结果。
- 导出统一 protobuf/jsonl 格式。
- 先离线构图，避免侵入 solver 主流程。

### Milestone 2：SAT-level GNN refocus 原型

- 实现 literal-clause GNN。
- 训练预测 strong branching/VSIDS oracle。
- 集成周期性 activity refocus。
- 在 MaxSMT/OMT+PB 上验证是否优于 SAT-only baseline。

### Milestone 3：加入理论与目标函数节点

- 扩展为异构图。
- 加入 numeric variable、linear constraint、objective、bound cut。
- 训练 objective-aware branching。
- 验证 OMT(LRA/LIRA) 上的 objective gap 与 solve time。

### Milestone 4：整数 B&B 分支策略

- 对 LIA/LIRA fractional integer variables 训练 MILP 风格 GNN。
- 与原生 integer branch heuristic 混合。
- 评测 B&B nodes、LP solves、solve time。

### Milestone 5：在线微调与稳定性

- solver-in-the-loop 采集轨迹。
- DAgger/RL fine-tuning。
- 加入 OOD 检测、超时回退、推理预算控制。

## 11. 主要风险与缓解

- GNN 推理开销过高：采用周期性 refocus、root embedding 缓存、k-hop 子图、轻量 MLP fallback。
- 离线标签与真实求解目标不一致：用 DAgger/RL 或 solver replay 修正。
- OMT benchmark 稀缺：从 MILP、MaxSMT、SMT-LIB、调度/资源分配领域生成合成数据，并做跨分布测试。
- 图规模过大：限制 learned clauses 数量，只保留 recent/glue/objective-relevant clauses。
- 模型破坏求解器鲁棒性：只作为 activity/phase bias，保留原生 heuristic 和完整回退。
- 泛化不足：按 theory/domain 分桶训练，同时保留 portfolio 策略，按 instance embedding 选择是否启用 GNN。

## 12. 推荐的最小实验闭环

第一版不要直接替换所有 branching。建议实现：

1. 选定 Z3/νZ 或 OptiMathSAT 作为平台。
2. 只做 Boolean decision 的周期性 GNN refocus。
3. 图先用 literal-clause + soft/objective/bound cut 节点。
4. 标签先用 strong branching top-k + 原生 solver trajectory 混合。
5. 在 MaxSMT/OMT+PB 和 OMT(LIA/LRA) 小中型 benchmark 上比较：
   - 原生 solver。
   - SAT-only GNN refocus。
   - OMT-aware GNN refocus。

如果 OMT-aware GNN 能在相同推理预算下减少 PAR-2、conflicts 或 objective gap area，再继续扩展到 phase selection 与 LIRA integer branching。

## 13. 参考文献清单

- Roberto Sebastiani, Silvia Tomasi. Optimization Modulo Theories with Linear Rational Costs.
- Roberto Sebastiani, Patrick Trentin. Pushing the Envelope of Optimization Modulo Theories with Linear-Arithmetic Cost Functions. TACAS 2015.
- Roberto Sebastiani, Silvia Tomasi, Patrick Trentin. OptiMathSAT: A Tool for Optimization Modulo Theories. JAR 2018.
- Nikolaj Bjørner, Anh-Dung Phan, Lars Fleckenstein. νZ - An Optimizing SMT Solver.
- Roberto Sebastiani, Patrick Trentin. On Optimization Modulo Theories, MaxSMT and Sorting Networks.
- Exploiting Partial-Assignment Enumeration in Optimization Modulo Theories. 2025.
- Ashera: Neural Guided Optimization Modulo Theory. UC Berkeley Technical Report, 2023.
- Maxime Gasse et al. Exact Combinatorial Optimization with Graph Convolutional Neural Networks. NeurIPS 2019.
- Prateek Gupta et al. Hybrid Models for Learning to Branch. NeurIPS 2020.
- Learning to Branch in Combinatorial Optimization with Graph Pointer Networks. 2023.
- Matthew Selsam et al. Learning a SAT Solver from Single-Bit Supervision. NeuroSAT, 2018.
- Matthew Selsam, Nikolaj Bjørner. Guiding High-Performance SAT Solvers with Unsat-Core Predictions. NeuroCore, 2019.
- Can Q-Learning with Graph Networks Learn a Generalizable Branching Heuristic for a SAT Solver? Graph-Q-SAT, NeurIPS 2020.
- Jia Hui Liang et al. Exponential Recency Weighted Average Branching Heuristic for SAT Solvers. CHB, AAAI 2016.
- Jia Hui Liang et al. Learning Rate Based Branching Heuristic for SAT Solvers. SAT 2016.
- Understanding VSIDS Branching Heuristics in Conflict-Driven Clause-Learning SAT Solvers.
- Designing New Phase Selection Heuristics. SAT 2020.
