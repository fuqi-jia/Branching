# 学习布尔分支在困难 SMT(LIA) 理论原子上的负结果与根因

> 结果文档（供论文）。日期 2026-07-09。承 SAT 正结果
> （[`findings-sat-learned-branching.md`](findings-sat-learned-branching.md)）向理论原子的扩展。
> 代码：`omt_branching/solver/{sat_instances,propagator_snapshot,lookahead,rl_decide,sat_solve}.py`
> + `examples/smt_branch.py`；全套测试绿（128 passed）。

## 1. 设定

把 SAT 管道原样扩到**理论原子**：原子改为**线性算术比较原子**（LIA）。困难 SMT(LIA) 由
`generate_hard_smt_lia` 生成（`n_vars` 整型变量 + 盒约束 `[0,ub]`；`n_disj` 个 `k` 元析取，系数
`[-chi,chi]`，紧到布尔搜索成瓶颈）。两臂均附 `LearnedDecidePropagator`（关预处理→纯 CDCL），
`conflicts` 为度量，与 SAT 完全同构。

**理论原子结构特征**（`build_bool_snapshot` 新增 `TheoryAtomInfo` var_coeffs/rhs + 数值变量节点 +
`atom↔var` / `atom↔bool` 边，复用既有 hetero-graph）**打破了** look-ahead imitation 的 exact-zero 分支
梯度冻结——即 look-ahead 标签在理论原子上**可被 GNN 拟合**（branch loss 从精确不变到单调下降）。

## 2. 结果（负）

困难 SMT(LIA)（`n_vars=8, n_disj=30`，多 seed，成对同实例）：

| 臂 | conflicts (mean±std) | vs VSIDS |
|---|---|---|
| VSIDS-decide | 108 ± 66 | — |
| learned-decide（trained, imitation+RL） | 269 ± 213 | **输** |
| learned-decide（untrained 对照） | 196 ± 160 | 输 |

- **learned 输 VSIDS**（约 2.5×）。$\sigma\approx\pm200$（10 实例）巨大，故「trained 比 untrained 还差」
  多半是噪声；**稳健的结论是 learned < VSIDS**。

## 3. 根因（系统调试 / oracle 探针）

用 **look-ahead 教师 oracle**（每决策点直接按教师 argmax 分支、无 GNN）作判别探针，vs VSIDS，在
**SMT 与 SAT（正对照）**上比 conflicts：

| 域 | VSIDS | 教师 oracle（dyn, max_atoms=32） | GNN（trained） |
|---|---|---|---|
| **SAT** 3-SAT n=70 | 123 | 127（≈VSIDS，成对 2/8） | **101（胜 VSIDS，且胜 oracle）** |
| **SMT(LIA)** | 120 | 112（≈VSIDS，成对 5/8） | 269（远输） |

去掉 `max_atoms` 截断后的**全覆盖教师 oracle**（纯教师序、无 defer 稀释）在 SMT 上 VSIDS 成对胜 2/3
（73/44/156 vs 148/118/45），即教师全序对 SMT ≈ 或略差于 VSIDS。

**结论链**：
1. **不是 teacher-mismatch**：教师 oracle 在 SAT 与 SMT 上**都 ≈ VSIDS**（教师信号本身中性）。
2. **SAT 上 GNN 既胜 VSIDS 又胜教师 oracle** → GNN 从弱 imitation **泛化出超过教师的序**——这才是 SAT
   正结果（~28%）的真正来源，而非「教师好」。
3. **SMT 上 GNN 远差于 VSIDS 与教师 oracle** → **GNN 在理论原子表示下泛化失败**：它总是**全覆盖硬
   override、几乎不 defer**，学到的准静态全序打不过 VSIDS 的**冲突自适应动态**。

**机理**：DPLL(T) 中 VSIDS 的逐冲突自适应很强。在纯布尔（SAT，子句共现图可学）上，GNN 的准静态
（周期 refocus）全覆盖序尚能超过它；但在理论原子上学到的序不够好，且 GNN 几乎不 defer（总 override）
→ 劣于 VSIDS。契合 native-z3 天花板与 OMT 负结果的「静态覆盖 < 动态 VSIDS」墙。

## 4. 诚实边界与后续

- **可学 ≠ 有用**：理论特征让 look-ahead 标签**可拟合**（护栏通过），但拟合出的全序**不构成**胜过 VSIDS
  的分支策略。这是「可学习性」与「求解收益」之间的清晰分离。
- **方差**：$\sigma$ 大，需更多 seed / 成对差收紧；但 learned<VSIDS 方向稳健。
- **后续方向**：
  1. **选择性 defer** —— GNN 输出置信度，只在高置信理论原子 override、其余交回 VSIDS（软混入 activity），
     直击「全覆盖硬 override 打不过动态 VSIDS」的根因。
  2. 更强理论感知特征（LP 松弛 / 理论求解器反馈）/ 更大训练预算。
  3. UNSAT-capable 教师（冲突/证明驱动），替代在矛盾基础上无标签的 `consequences`。

## 5. 结论

**在困难 SMT(LIA) 理论原子上，同一套（在 SAT 上取胜的）学习布尔分支方法不敌 z3 VSIDS。** 根因不在
look-ahead 教师（其 oracle 两域都 ≈ VSIDS），而在 **GNN 未能像在 SAT 上那样、在理论原子表示下泛化出
超越 VSIDS 动态的全覆盖序**。这是一条诚实的边界结果：学习分支的收益依赖「可学结构 × 泛化出超越动态
基线的策略」，SAT 满足、SMT(LIA)（本预算/表示下）不满足。
