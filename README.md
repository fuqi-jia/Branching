# omt_branching — 学习驱动的 OMT/SMT/SAT 分支选择

用**图神经网络 (GNN) 学习布尔分支决策**，并通过 **z3 `UserPropagator`**（`add_decide` / `next_split`）
把学到的策略注入 z3 **内部的 CDCL 决策**——**全程不修改、不重编译 z3**（仅用 pip 的 `z3-solver` 4.15.4）。

本仓库是 AAAI2027 论文的研究原型。它记录了一条**诚实的科学弧线**：在 OMT 优化回路中学习分支
**无效**（负结果），转向**单次可满足性搜索**后在困难 SAT 上**有效**（正结果，learned < VSIDS），
再扩展到 **SMT(LIA) 理论原子**时**又无效**（负结果，并给出了经系统调试确认的根因）。

---

## 目录

- [1. 主要结果（先看结论）](#1-主要结果先看结论)
- [2. 核心机制](#2-核心机制)
- [3. 总体架构与数据流](#3-总体架构与数据流)
- [4. 形式化（公式级）](#4-形式化公式级)
  - [4.1 问题设定与度量](#41-问题设定与度量)
  - [4.2 look-ahead 教师（imitation 监督信号）](#42-look-ahead-教师imitation-监督信号)
  - [4.3 图构造与特征](#43-图构造与特征)
  - [4.4 GNN 编码器（R-GCN 风格）](#44-gnn-编码器r-gcn-风格)
  - [4.5 任务头](#45-任务头)
  - [4.6 imitation 损失（ListNet 风格）](#46-imitation-损失listnet-风格)
  - [4.7 RL 目标（REINFORCE + 逐实例基线）](#47-rl-目标reinforce--逐实例基线)
- [5. 实验流程（一步步跑通）](#5-实验流程一步步跑通)
- [6. 结果详解与根因](#6-结果详解与根因)
- [7. 目录结构](#7-目录结构)
- [8. 安装 / 运行 / 测试](#8-安装--运行--测试)
- [9. 开发与贡献流程](#9-开发与贡献流程)
- [10. 已知边界与后续方向](#10-已知边界与后续方向)

---

## 1. 主要结果（先看结论）

三臂对比均为**成对同实例**、多 seed；两臂在**同一未预处理搜索**下只有决策启发式不同，因此
`conflicts` 差异**隔离了分支质量**（详见 §4.1）。

| 设定 | 度量 | VSIDS | learned | 结论 |
|---|---|---|---|---|
| **困难 3-SAT**（相变点 ratio≈4.26, n=70） | conflicts | 111 | **80** | **learned 胜 ~28%**（本会话复现 116→101，方向一致） |
| **PHP**（pigeonhole, UNSAT） | conflicts | 3372 | 3725 | learned 输（UNSAT 无 imitation 标签，见 §6） |
| **困难 SMT(LIA)**（`n_vars=8, n_disj=30`） | conflicts | 108±66 | 269±213 | **learned 输**（根因见 §6） |
| **OMT 优化回路** | rlimit | ≈ | ≈ | 无差异（分支非瓶颈，三堵墙，见 §6） |

**一句话结论**：学习布尔分支的收益，取决于「结构是否可学」**且**「GNN 是否能泛化出**超过 VSIDS
动态**的全序」。这在困难 SAT 上成立；在 OMT 回路（分支非瓶颈）与 SMT(LIA) 理论原子（GNN 全覆盖
override 打不过 VSIDS 冲突自适应动态）上不成立。完整分析见
[`docs/findings-sat-learned-branching.md`](docs/findings-sat-learned-branching.md)、
[`docs/findings-userpropagator-learned-branching.md`](docs/findings-userpropagator-learned-branching.md)。

---

## 2. 核心机制

**为什么用 `UserPropagator`。** z3 不暴露「注入 VSIDS 活跃度」的软接口，但暴露了 `UserPropagateBase`：
- `add_decide(callback)` / `next_split(atom, phase, ...)`：在 z3 **每个内部决策点**回调外部策略，让它
  指定下一个要分裂的布尔文字与相位——**这就是接管 z3 内部 CDCL 决策、且不改 z3 源码的载体**。
- 回调返回 `None` 即**退回 z3 原生 VSIDS**（保持完备性）。

**关键副作用（务必知道）：附加一个 propagator 会让 z3 关闭其预处理/化简，退化为纯 CDCL 搜索。**
这会「制造」出大量 conflicts（否则困难实例经预处理后 conflicts≈0，无信号）。因此**两臂都必须附
propagator**才公平——见 §4.1。

**两套 harness**（注意其对称性差异）：
- `solver/sat_solve.py`（**单次可满足性检查**，SAT/SMT）：**两臂都附** propagator，VSIDS 臂的 decider 恒返回 `None`。
- `solver/decide_omt.py`（**OMT 线性搜索回路**）：**仅 learned 臂附** propagator，VSIDS 基线臂不附。
  论文中引用某张表时，请核对它出自哪套 harness。

---

## 3. 总体架构与数据流

代码按**三段解耦 + 两个 dataclass 契约**组织；求解器只接触契约，不接触模型内部。

```
                      ┌────────────── 训练期（离线）──────────────┐
 困难实例生成器 ─────▶ look-ahead 教师 ─────▶ imitation 样本 ─────▶ ImitationTrainer（冷启动）
 (sat_instances.py)   (lookahead.py)          (training_data.py)     ─▶ DecideRLTrainer（RL 微调）
                                                                              │ 训练出 policy
                      ┌──────────────── 部署/评测期（求解中）──────────────────┘
                      ▼
 z3.Solver.check() ──每个决策点──▶ LearnedDecidePropagator._on_decide
                                        │  build_bool_snapshot(当前赋值)
                                        ▼
   SolverSnapshot ──GraphBuilder──▶ HeteroGraph ──HeteroEncoder+Heads──▶ PolicyOutput
        (input/)                      (graph/)          (model/)              │
                                                                              ▼
   z3.next_split(atom,phase) ◀──PolicyDecider(argmax/defer)◀── BranchingAdvice ◀─AdviceDecoder(output/)
```

- **输入契约** `SolverSnapshot`（`input/solver_state.py`）：求解器在决策点按 dataclass 填一份快照。
- **输出契约** `BranchingAdvice`（`output/advice.py`）：activity 先验、候选排序、phase、整数 split、置信度、`fallback` 标记。
- **门面** `BranchingPolicyService.advise(snapshot)`（`service.py`）：唯一入口，串起 建图→推理→解码。
- **决策器** `PolicyDecider`（`solver/policy_decider.py`）：把 policy 包成 propagator 的 decide 回调，
  **每 `refocus_every` 次决策重算一次**原子优先级（调 GNN，代价可控），decide 时 O(1) 取最高优先级
  的未定原子；`use_gnn=False` 或无候选时返回 `None` → 退回 VSIDS。

---

## 4. 形式化（公式级）

### 4.1 问题设定与度量

给定断言集合 $\Phi$（CNF 子句 / 理论子句）与候选布尔原子集合 $\mathcal{A}$，跑一次
`z3.Solver().check()`。**两臂都附** `LearnedDecidePropagator`（关预处理 → 纯 CDCL）：

- **VSIDS 臂**：decider $\equiv \texttt{None}$（每次都退回 z3 原生 VSIDS）。
- **learned 臂**：decider = GNN 策略（look-ahead imitation 冷启动 + $-\log$ conflicts 的 RL 微调）。

`solve_sat_with_decider` 返回三元度量（`solver/sat_solve.py`）：

$$
\texttt{conflicts}=\text{z3 stat ``conflicts''},\quad
\texttt{rlimit}=\text{z3 stat ``rlimit count''},\quad
\texttt{decisions}=\texttt{prop.n\_decisions}.
$$

- **`conflicts`** = 分支质量的直接度量（越少越好），是本项目主指标。
- **`rlimit`** = 诚实的总工作量（含理论求解开销）。
- **`decisions`** = **仅统计 learned 决策器实际 override 的次数**（不含退回 VSIDS 的原生决策）。

两臂共享同一未预处理搜索、仅决策启发式不同 ⇒ conflicts 差异**隔离分支质量**。

### 4.2 look-ahead 教师（imitation 监督信号）

对每个候选原子 $a$，用 z3 的 `consequences` 计其**边际传播强度**（march / look-ahead 风格）。
令 $\mathrm{Imp}(L)$ 为在假设集 $L$ 下被强制取值的原子键集合（`_strip_not` 剥离否定）：

$$
B=\mathrm{Imp}(\varnothing)\ \text{(仅 hard 约束就蕴含的原子, 基线)},\qquad
p_t(a)=\bigl|\mathrm{Imp}(\{a\})\setminus B\setminus\{a\}\bigr|,\quad
p_f(a)=\bigl|\mathrm{Imp}(\{\lnot a\})\setminus B\setminus\{a\}\bigr|.
$$

**教师分数与相位**（`solver/lookahead.py:70-71`）：

$$
\boxed{\ \mathrm{score}(a)=(p_t(a)+1)\,(p_f(a)+1)\ },\qquad
\mathrm{phase}(a)=\mathbb{1}[\,p_t(a)\ge p_f(a)\,].
$$

- $+1$ 偏置：两侧都不传播的原子得分 $1$（而非 $0$），可比较。
- **减去基线 $B$**：否则 box/被蕴含原子在每个假设下都「蕴含」同一批，计数恒等 → 标签 uniform 无法学。
- **排除非决策点**：若某侧在根部即 `unsat`（$a$ 被 hard 蕴含或矛盾），跳过该原子——它由 z3 传播而非分支。
- 配置 `LookaheadConfig(max_atoms=32)`：只给前 32 个原子打分（`eps` 字段声明但**未使用**）。

> **注意（可学习性教训）**：`consequences` 在 SMT 上包含**理论传播**。教师分数是「传播计数」，在纯 SAT
> 单元传播上是好的分支启发；在 DPLL(T) 上它是否是好的**全序**，是本项目的核心问题（见 §6）。

### 4.3 图构造与特征

`SolverSnapshot` → `HeteroGraph`（`input/graph_builder.py`）。异构节点/边与其特征维度：

**节点特征维度**（`_NODE_DIMS`，缺失值约定见下）：

| 节点类型 | 维度 | 主要特征 |
|---|---|---|
| `BOOL_VAR` | **19** | 赋值 one-hot、decision level、候选/消去标记、VSIDS/LRB/CHB 活跃度、phase saving、出现/正负极性计数、soft 标记 |
| `CLAUSE` | **11** | 子句类型 one-hot、LBD、activity、文字数、是否已满足 |
| `THEORY_ATOM` | **22** | 原子类型 one-hot(LE/GE/EQ/…)、`signed_log(rhs)`、slack/violation/lp_value/reduced_cost、basis 状态、变量数 |
| `NUMERIC_VAR` | **17** | 是否整型、LP 值/上下界、分数性、目标系数、reduced cost、pseudocost |
| `OBJECTIVE` | **10** | 极性、incumbent/best_bound/gap、lex 优先级 |
| `SEARCH_STATE` | **16** | 深度/决策层/trail、restart/conflict 计数、conflict rate、search mode |

**缺失值编码约定**（`graph_builder.py`）：可选标量 $x\mapsto[x,1]$（有）或 $[\text{default},0]$（无）；
大尺度数值过 $\mathrm{signed\_log}(x)=\operatorname{sign}(x)\log(1+|x|)$；三态布尔 → one-hot $[none,true,false]$。

**边类型与边特征**（`_EDGE_DIMS` / `EDGE_SCHEMA`，**有向**）：

| 边类型 | $(\text{src}\to\text{dst})$ | 边特征 |
|---|---|---|
| `LITERAL_IN_CLAUSE` | `BOOL_VAR → CLAUSE` | `[is_positive]` |
| `ATOM_ABSTRACTED_BY` | `THEORY_ATOM → BOOL_VAR` | 无 |
| `VARIABLE_IN_ATOM` | `NUMERIC_VAR → THEORY_ATOM` | $[\mathrm{signed\_log}(c),\ \operatorname{sign}(c),\ \log(1+\lvert c\rvert)]$ |
| `VARIABLE_IN_OBJECTIVE` | `NUMERIC_VAR → OBJECTIVE` | 同上（目标系数） |
| `SOFT_WEIGHT` | `BOOL_VAR → OBJECTIVE` | $[w,\ \log(1+\lvert w\rvert)]$ |
| `BOUND_RELATES_VARIABLE` | `OBJECTIVE → NUMERIC_VAR` | 上下界 |
| `STATE_TO_BOOL` / `STATE_TO_OBJECTIVE` | `SEARCH_STATE → …` | 无（全局广播边） |

> **实现现状更正**：`interfaces.py` 注释称「所有边补充反向边」，但**代码并未实现反向边**——消息传递
> 严格按 `EDGE_SCHEMA` 的**有向**边进行（例如只有 `THEORY_ATOM→BOOL_VAR`，无反向）。

**理论原子的线性分解**（`solver/propagator_snapshot.py::_linear`）。把比较原子 $\sum_i c_i x_i\ \mathrm{OP}\ b$
统一到一侧。设 $\text{lhs}=(\mathbf{c}^{L},k^{L})$、$\text{rhs}=(\mathbf{c}^{R},k^{R})$ 为两侧线性分解，则

$$
\mathbf{c}=\mathbf{c}^{L}-\mathbf{c}^{R}\ (\text{丢弃 }0\text{ 系数}),\qquad
b=k^{R}-k^{L},\qquad
\texttt{kind}=\_\mathrm{map\_kind}(\mathrm{OP}).
$$

存为 `TheoryAtomInfo(atom_id=k, bool_var_id=k, var_coeffs=`$\mathbf{c}$`, rhs=`$b$`)`——**原子即其抽象布尔
变量（同一键 $k$）**，从而 `ATOM_ABSTRACTED_BY` 边把理论原子节点连到同键 bool 节点；每个数值变量存为
`NumericVarInfo(is_integer=True)`（LIA），`VARIABLE_IN_ATOM` 边由 `var_coeffs` 连出。**纯 SAT 布尔常量
（非算术）被跳过**，`theory_atoms`/`numeric_vars` 保持空——保证纯布尔图不受影响。

### 4.4 GNN 编码器（R-GCN 风格）

`model/gnn.py`。输入投影 $h_v^{(0)}=\mathrm{Linear}_{t(v)}(x_v)$，堆叠 `num_layers` 层（默认 **3**，
`hidden`=**64**）关系式消息传递。第 $l$ 层对节点 $v$（类型 $t(v)$）：

$$
\boxed{\;
h_v^{(l+1)}=\mathrm{LN}_{t(v)}\!\Bigl(\,h_v^{(l)}+\mathrm{ReLU}\bigl(\,
W^{\text{self}}_{t(v)}\,h_v^{(l)}
+\underbrace{\operatorname*{mean}_{r,\;u\in\mathcal N_r(v)} W_r\,[\,h_u^{(l)}\,\Vert\,e_{u\to v}\,]}_{\text{按目标节点对所有关系}r\text{求和后取均值}}
\bigr)\Bigr)\;}
$$

- $W_r=\mathrm{Linear}(\text{hidden}+\dim(e_r),\ \text{hidden})$：**每种边类型一套**消息权重，输入拼接边特征 $e_{u\to v}$。
- 聚合：对**所有以 $v$ 为目标**的关系，消息用 `index_add_` **求和到同一累加器**，再除以度数（度下限 $1$，防除零）= **均值聚合**。
- 更新：自变换 + 均值消息 → ReLU → **残差** → **LayerNorm**（均按节点类型独立参数）。

### 4.5 任务头

`model/heads.py`。公共块 $\mathrm{mlp}(x)=\mathrm{Linear}\to\mathrm{ReLU}\to\mathrm{Linear}$，
所有头输出**原始未归一化**分数（softmax/mask 在 `policy.py` 后置）：

| 头 | 输入 | 输出 |
|---|---|---|
| `BranchingHead` | `BOOL_VAR` 嵌入 | 每原子标量分数 $s_a=\mathrm{mlp}(h_a)$（**分支主头**） |
| `PhaseHead` | `BOOL_VAR` 嵌入 | 每原子「取真」logit（独立权重） |
| `IntegerBranchHead` | `NUMERIC_VAR` 嵌入 | 整数 B&B 的 split 分数 + 向上取整方向 logit |
| `AuxiliaryHeads` | `BOOL_VAR` 嵌入 | 4 个辅助：conflict 概率、unsat-core 归属、目标改进（回归）、子树规模（回归） |

### 4.6 imitation 损失（ListNet 风格）

`model/trainer.py`。总损失（各任务加权）：

$$
L=L_{\text{branch}}+\lambda_{\text{phase}}L_{\text{phase}}
+\lambda_{\text{int}}\,(L_{\text{int}}+L_{\text{int\_dir}})
+\lambda_{\text{aux}}L_{\text{aux}},
$$

默认 $\lambda_{\text{phase}}=0.5,\ \lambda_{\text{int}}=1.0,\ \lambda_{\text{aux}}=0.3$。分支/整数排序损失为
**ListNet 风格软交叉熵**（教师分数经 softmax 作软标签）：

$$
\boxed{\;
L_{\text{rank}}=-\sum_{a\in\text{cand}} \mathrm{softmax}(\text{target})_a\;\log\mathrm{softmax}(s)_a
\;=\;H\!\bigl(\mathrm{softmax}(\text{target}),\ \mathrm{softmax}(s)\bigr)\;}
$$

（等价于 $\mathrm{KL}$ 到常数；$s$ 为 `BranchingHead` 分数，target 为 §4.2 教师分数。）相位/方向用带 logits
的 BCE；辅助头用 BCE（conflict/core）+ MSE（目标改进/子树）。优化器 `Adam(lr=1e-3, weight_decay=1e-5)`，
`grad_clip=5.0`。`fit()` 返回逐 epoch 的 `dict`，键含 `"loss"`（恒有）与按样本标签出现的 `"branch"/"phase"/"int"/"aux"`。

> **可学习性护栏**：SMT 理论原子上 `"branch"` 损失是否随 epoch 下降，是判断「标签能否被特征预测」的关键
> 探针（若冻结在 $\log N$ 的 uniform floor，说明特征不足；本项目正是靠 §4.3 的理论原子结构特征打破了
> exact-zero 分支梯度冻结）。

### 4.7 RL 目标（REINFORCE + 逐实例基线）

`solver/rl_decide.py`。采样决策器每步在动作空间 $[\texttt{defer}, a_1, a_2,\dots]$（仅当前未定原子）上采样：

$$
\pi(\cdot\mid s)=\mathrm{softmax}\bigl([\,\ell_{\text{defer}},\ s_{a_1},\dots,s_{a_m}\,]\bigr),\qquad
a=\begin{cases}\text{multinomial}(\pi)&\text{采样}\\ \arg\max\pi&\text{贪心}\end{cases}
$$

其中 $\ell_{\text{defer}}$ 是 trainer 持有的共享标量 `nn.Parameter`（占索引 0）。采样到 $\texttt{defer}$ → 返回
`None`（退回 VSIDS）；否则选中该原子并**强制相位为真**（相位不采样）。

**奖励**（log 压缩，非原始 conflicts）：SAT 臂 $R=-\log(1+\texttt{conflicts})$；OMT 臂 $R=-\log(1+\texttt{rlimit})$。
**逐实例键的 EMA 基线**（动量 $0.9$）：$b_{\text{key}}\leftarrow 0.9\,b_{\text{key}}+0.1\,R$。
**策略梯度**（整条轨迹共享一个标量优势 $A=R-b_{\text{key}}$，无逐步 credit、无折扣）：

$$
\boxed{\;
L_{\text{RL}}=-\frac{1}{n}\sum_{t=1}^{n}\log\pi(a_t\mid s_t)\cdot(R-b_{\text{key}})\;}
$$

`refocus_every` 默认 **50**（RL）/ 由 CLI 指定（部署），首次调用即 refocus。

---

## 5. 实验流程（一步步跑通）

以 SMT(LIA) 为例（SAT 完全同构，只换生成器）。全部在 `examples/smt_branch.py`：

**① 生成困难实例**（`solver/sat_instances.py::generate_hard_smt_lia`）。构造紧随机 SMT(LIA)，使布尔搜索
成瓶颈（附 propagator 后数百 conflicts）：

$$
x_i\in[0,\text{ub}]\ (i<\text{n\_vars}),\qquad
\bigwedge_{j=1}^{\text{n\_disj}}\ \bigvee_{l=1}^{k}\Bigl(\textstyle\sum_i c^{(j,l)}_i x_i\ \{\le\,\text{或}\,\ge\}\ b^{(j,l)}\Bigr),
$$

系数 $c\in[-\text{chi},\text{chi}]$、$b\in[-\text{ub},\text{ub}\cdot\text{chi}]$、每文字 $\le/\ge$ 各半；默认
`n_vars=8, n_disj=30, k=3, ub=6, chi=4`。返回 `(atoms, clauses)`。

**② imitation 冷启动**：`build_lookahead_examples_sat`（§4.2 教师打标）→ `ImitationTrainer.fit`（§4.6）。

**③ RL 微调**：`DecideRLTrainer.train_sat`（§4.7，reward=$-\log(1+$conflicts$)$）。

**④ 两臂对比**：对每个测试实例跑
`solve_sat_with_decider(clauses, atoms, decider_factory)`——VSIDS 臂 `None`，learned 臂
`PolicyDecider`——比 `conflicts` 均值±标准差、成对胜率。**训练/测试 seed 不相交**（防泄漏）。

```bash
# 困难 SMT(LIA)：imitation(+RL) vs VSIDS
conda run -n omt python -m examples.smt_branch \
    --n-vars 8 --n-disj 30 --test 12 --train 30 --epochs 25 --rl-iters 2

# 对照：不训练（--train 0 --rl-iters 0）验证 trained ≫ untrained
# 困难 3-SAT（正结果）
conda run -n omt python -m examples.sat_branch --sat-n 70 --test 20 --train 30 --epochs 25
```

---

## 6. 结果详解与根因

**困难 3-SAT（正结果）**：learned-decide conflicts < VSIDS（~28%，成对同实例、两次运行方向一致、
trained≫untrained）。**PHP（UNSAT）输**：`consequences` look-ahead 教师在 UNSAT 基础上无标签（基础即矛盾），
只能靠 RL，提升有限——诚实边界。

**OMT 优化回路（负结果，三堵墙）**：(1) rlimit 被回路开销主导 → 布尔分支非瓶颈、$-$rlimit 无 RL 信号；
(2) 无 headroom（更难实例 VSIDS 仍极少 conflicts）；(3) z3 只暴露硬 `next_split`、不暴露活跃度软注入 →
静态覆盖 < 动态 VSIDS。

**SMT(LIA) 理论原子（负结果 + 经系统调试确认的根因）**。理论原子结构特征（§4.3）**打破了** look-ahead
imitation 的 exact-zero 分支梯度冻结（标签可拟合），但 learned 仍输 VSIDS（269 vs 108，$\sigma\approx\pm200$）。
用 **oracle 探针**（直接按教师 argmax 分支、无 GNN）定位根因（详见
[`docs/findings-smt-lia-learned-branching.md`](docs/findings-smt-lia-learned-branching.md)）：

- **不是 teacher-mismatch**：教师 oracle 在 SAT 与 SMT 上**都 ≈ VSIDS**。
- **SAT 上 GNN 既胜 VSIDS 又胜教师 oracle** → GNN 从弱 imitation **泛化出超过教师的序**（这才是 SAT 正结果来源）。
- **SMT 上 GNN 远差于 VSIDS 与教师 oracle** → **GNN 在理论原子表示下泛化失败**：它总是**全覆盖硬 override、
  几乎不 defer**，学到的准静态全序打不过 VSIDS 的**冲突自适应动态**。$\sigma$ 巨大说明「trained<untrained」是噪声。

**教训**：收益 = 可学结构 × GNN 泛化出超越 VSIDS 动态的全序。SAT 满足，SMT(LIA) 在本预算/表示下不满足。
后续方向：让 GNN 学会**选择性 defer**（高置信才 override、余交回 VSIDS，即 CLAUDE.md 的 VSIDS-refocus 软混入），
而非全覆盖硬覆盖。

---

## 7. 目录结构

```
omt_branching/
├── interfaces.py              # 节点/边/原子/搜索枚举 + EDGE_SCHEMA（一处改，多处联动）
├── graph/hetero_graph.py      # 轻量异构图容器（不依赖 PyG/DGL，手写 index_add_ 聚合）
├── input/
│   ├── solver_state.py        # 【输入契约】SolverSnapshot + 各 *Info dataclass
│   └── graph_builder.py       # SolverSnapshot → HeteroGraph（节点/边特征编码，_NODE_DIMS/_EDGE_DIMS）
├── model/
│   ├── gnn.py                 # HeteroEncoder（R-GCN 风格关系消息传递，§4.4）
│   ├── heads.py               # 分支/相位/整数/辅助任务头（§4.5）
│   ├── policy.py              # 编码器 + 多头 = BranchingPolicy → PolicyOutput
│   ├── trainer.py             # 阶段一：离线 imitation（ListNet 排序损失，§4.6）
│   ├── finetune.py            # 阶段二：solver-in-the-loop（DAgger/REINFORCE）
│   └── inference.py           # 部署期推理（预算门控 + OOD 回退）
├── output/{advice.py,decoder.py}   # 【输出契约】BranchingAdvice + 解码
├── service.py                 # BranchingPolicyService.advise —— 唯一门面
└── solver/                    # ★ z3 集成与实验核心
    ├── propagator.py          # LearnedDecidePropagator（add_decide/next_split 接管决策）
    ├── propagator_snapshot.py # build_bool_snapshot：断言 → SolverSnapshot（含理论原子分解 §4.3）
    ├── policy_decider.py      # PolicyDecider：policy → decide 回调（周期 refocus）
    ├── lookahead.py           # look-ahead 教师（§4.2）
    ├── sat_solve.py           # solve_sat_with_decider（单次 check，两臂均附 propagator，§4.1）
    ├── decide_omt.py          # OMT 线性搜索回路 harness（仅 learned 臂附 propagator）
    ├── rl_decide.py           # DecideRLTrainer / SamplingPolicyDecider（REINFORCE，§4.7）
    ├── sat_instances.py       # generate_php / generate_rand_3sat / generate_hard_smt_lia
    ├── training_data.py       # build_lookahead_examples(_sat)：教师 → imitation 样本
    └── extractor.py, strong_branch.py, …   # LP 松弛 / 强分支等基础设施

examples/    demo.py, z3_demo.py, sat_branch.py, smt_branch.py, decide_branch.py, lia_branch.py, rl_LRA.py
docs/        findings-*.md（SAT 正结果 / OMT 负结果）
tests/solver/    pytest 套件（128 passed）
```

---

## 8. 安装 / 运行 / 测试

PyTorch **不在** base 环境，需 `omt` conda 环境（或 `pip install -r requirements.txt`，torch≥2.0）。

```bash
conda activate omt                                  # 或 pip install -r requirements.txt

# 冒烟：建图 → 推理 → 解码 → 一步训练
python -m examples.demo

# 核心实验（见 §5）
python -m examples.sat_branch --sat-n 70 --test 20 --train 30 --epochs 25
python -m examples.smt_branch --n-vars 8 --n-disj 30 --test 12 --train 30 --epochs 25 --rl-iters 2

# 测试
conda run -n omt python -m pytest tests/solver -q       # 全套（含 GNN 训练测试，较慢）
conda run -n omt python -m pytest tests/solver/test_sat_solve.py -q   # 单文件
```

---

## 9. 开发与贡献流程

- **约定**：工作语言中文——docstring/注释/`README.md`/commit message 用中文；`__all__`/标识符/类型注解用英文。
  遵 YAGNI；优先**正确性与可扩展性**，控制参数尽量收敛到统一的 config-dataclass（`LookaheadConfig` /
  `TrainConfig` / `DecideRLConfig`）而非散落 flag。
- **跨文件不变量**（改动前必读，详见 `CLAUDE.md`）：`FeatureSpec` / `_NODE_DIMS` / `_EDGE_DIMS` / `_encode_*`
  四处**维度联动**（`GraphBuilder._check` 运行时断言）；枚举与 `EDGE_SCHEMA` 联动；训练标签与 `PolicyOutput`
  用**图局部索引**，只有 `AdviceDecoder` 还原 solver id。
- **Git 工作流**：`main` 上提交 → `git push fork main`（fork = `fuqi-jia/Branching`）→ PR 到
  `Electroplating/Branching`（origin）。commit message 结尾附
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`。

---

## 10. 已知边界与后续方向

- **实现现状更正**：`interfaces.py` 注释所称的「反向边」**未实现**，消息传递严格有向（§4.3）。
  `LookaheadConfig.eps` 声明但未用。RL 奖励为 $-\log(1+\cdot)$ 压缩、优势为**整条轨迹单标量**（无逐步 credit/折扣）；
  RL 只采样「选哪个原子/是否 defer」，**相位恒为真**。两套 harness 的 propagator 附加**不对称**（§2）。
- **后续（若继续）**：
  1. **选择性 defer** —— GNN 输出置信度，只在高置信原子 override、其余交回 VSIDS（软混入 activity），
     直击 SMT 根因（§6）。
  2. **UNSAT-capable 教师** —— 冲突/证明驱动的监督信号，替代在 UNSAT 上无标签的 `consequences`（PHP 边界）。
  3. 更强理论感知特征 / 更大训练预算；真实 solver 轨迹上验证 wall-clock / PAR-2。
  4. 效率：look-ahead 的 `consequences` 开销较大，可子采样 / 缓存 root embedding（当前阶段以正确性优先）。
```
