# 为什么"挂 UserPropagator 就变慢"：归因与修复

> 诊断文档。日期 2026-07-17。针对学长反馈："即使 `on_decide` 恒直接返回（不跑 GNN、
> 只用 z3 默认选择），挂 `UserPropagatorBase` 仍远慢于 `decider_factory=None`，单次采样 2–3min"。
> 涉及 `omt_branching/solver/{propagator,decide_omt,rl_decide,propagator_snapshot}.py`。
> 可复现诊断脚本：`docs/ref/bench_prop_overhead.py`（从仓库根目录 `python docs/ref/bench_prop_overhead.py`）。

## 结论（先说要点）

**Cursor 的假设"挂 propagator → z3 关预处理 → CDCL 过重"与实测不符。** z3 的求解工作量
（`rlimit` / `conflicts` / `decisions`）在"仅挂 propagator"时几乎不变，甚至更低；轻量预处理
（`simplify` + `propagate-values`）对耗时**零影响**。真正的主因是 **Python 回调开销**，其中
**`_on_fixed` 每次回调对原子做一次 `str()`** 占了总开销的约 **65%**。

已修复（`propagator.py`），行为完全不变（同 `value`/`rlimit`/`conflicts`），诊断实例上
**18.7× → 7.2×**（wall-time 相对无 propagator 基线），即绝对耗时 **7.9s → 3.3s（2.4×）**。

## 1. 诊断方法：分离"求解工作量"与"Python 开销"

关键是**同时**测 wall-time 与 z3 自报的 `rlimit`/`conflicts`。若挂 propagator 后耗时暴涨但
`rlimit` 不变 → 是 Python 开销；若 `rlimit` 同步暴涨 → 才是 z3 少做了预处理/多做了搜索。

在同一"较难布尔结构整数 OMT"实例（12 vars, 90 析取, k=5；注册 12 个析取子句原子）上，把
`solve_omt_with_decider` 的线性搜索回路在多种 propagator 配置下各跑一遍：

| 臂 | 含义 | 耗时 | rlimit | conflicts | 说明 |
|---|---|---|---|---|---|
| `none_raw` | 无 propagator，原始 hard（真基线） | 0.30s | 347k | 57 | 基线 |
| `none_pp` | 无 propagator，但求解**预处理后**公式 | 0.25s | 315k | 15 | 预处理**不是**慢因（反而略快） |
| `attach0` | 挂 prop，注册 0 原子，无 add_fixed，decide 立即返回 | 0.42s | 327k | 143 | 仅"挂上"几乎无损 |
| `reg_nofix` | 挂 prop，注册全部原子，无 add_fixed，decide 立即返回 | 2.08s | 538k | 727 | +注册 add_decide → 搜索被扰动 |
| `fix_empty` | 同上 + add_fixed（回调体只计数） | 2.26s | 538k | 727 | 回调**调用**本身很便宜（+0.2s） |
| `fix_idkey` | 同上 + add_fixed（get_id() 查表） | 2.26s | 538k | 727 | 用 get_id 代替 str：**免费** |
| `full_noop` | 同上 + add_fixed（`atom_key`=`str(t)`，**现状**） | 6.40s | 538k | 727 | `str()` 独占 **4.1s（65%）** |

**读法**：`fix_idkey` 与 `full_noop` 的 `rlimit`/`conflicts` **完全相同**（538k / 727），唯一
差别是 `_on_fixed` 里 `str(t)` vs `get_id()` 查表 → 4.1s 差额是**纯 `str()` 序列化**，约
**2.2ms/次 × 1853 次**。且 z3 每次回调都新建 `t` 的 Python 包装（`distinct_ids == n_fixed`），
使 `atom_key` 的 `id()` 缓存**永远 miss**，每次都重新 `str()`。

## 2. 三层开销归因

1. **`_on_fixed` 的 `str(t)`（主因，~65%，纯 Python，零收益）** —— 已修复。
2. **注册 `add_decide` 扰动 z3 搜索（真实求解代价，~2×）** —— `reg_nofix` vs `attach0`：
   decisions 7041→18164、rlimit ×1.55。**接管分支必然改变搜索**，这是"接管 z3 内部决策"的
   固有代价；只要 GNN 的 `next_split` 真能降冲突，这 2× 就是"入场费"。**不可无损消除。**
3. **每次 decide 的 `undecided` 列表推导 + `dict(self._val)` 拷贝（次要 Python 开销）** ——
   `dict` 拷贝已去除。

## 3. 修复（`propagator.py`，行为不变）

- `_on_fixed`：改用**注册时一次性建的 `get_id()→key` 表** O(1) 命中，替代每次 `str(t)`。
  安全性：注册原子被 `self.atoms` 钉住，存活整个求解，其 z3 AST id 不会被回收复用；z3 只对
  `add` 过的项回调 fixed，故 `t.get_id()` 必命中表。（`atom_key` 的注释警告的是**跨求解**的
  长期缓存里 id 复用，本表仅活在单次求解内，不受影响。）
- `_on_decide`：`self._val` **只读**传给 decider（各 decider 仅在 refocus 即时读取、不留引用、
  不修改），省去每次 decide 一次 `dict` 拷贝。

验证（真实 `solve_omt_with_decider`，decider 恒 `None` = 学长的实验）：修复前 7.9s、修复后
3.3s，`value`/`rlimit`/`conflicts` 三者逐位一致 → **纯提速、零行为改变**。

## 4. 给学长的进一步建议（按性价比排序）

1. **已落地的 `get_id` 修复**是最大且最安全的一笔（2.4×），直接解决"no-GNN 也慢"。
2. **剩余 ~2× 来自 add_decide 扰动搜索**，属固有代价。想再压：可让 policy 只在"够自信"时才
   `next_split`、否则整段放行 VSIDS（现有 defer 机制已是此思路）——但这不改变 z3 每次 decision
   仍会**调用** decide 回调的事实（回调调用本身便宜，见 `fix_empty`）。
3. **单次采样 2–3min 的另一半**可能在 GNN 侧：每 `refocus_every` 次决策要 `build_bool_snapshot`
   + `GraphBuilder.build` + `policy.infer` 一次。若仍偏慢，优先量它（本文附诊断脚本
   `verify_full_collect.py`），再决定是否加大 `refocus_every` 或缓存图骨架。
4. **不要**再往"关预处理"方向调 z3 参数——实测证明那不是瓶颈。

## 5. 验证时另发现的两个训练路径 bug（与本问题独立，均已修）

排查过程中跑 `tests/solver`（`git stash` 隔离证明与本次改动无关，是 pull 带入）暴露两处：

- **`decide_omt.py`：`sample=True` 且 `ref_rlimit=None` 时 `2 * ref_rlimit` 崩溃**
  （`TypeError: int * NoneType`）。`collect()`/`train()` 未传 `ref_rlimit` 即触发。已加
  `ref_rlimit is not None` 守卫（无参考时不做预算剪枝）。
- **`rl_decide.py::decide_rl_reward` 键名笔误 `weighted_rlimit`（应为 `"weighted rlimit"`，含空格）**：
  `solve_omt_with_decider` 产出的是带空格的键，消费端取下划线键 → 恒 `None` → **reward 恒 `-2.0`**
  → REINFORCE 优势塌缩、**策略无法学习**。这是比"慢"更严重的训练问题。已改为正确键，并把
  `assert ref_rlimit is not None` 换成"无参考则返回 `-2.0`"的守卫（与 docstring 描述一致）。
  端到端验证：修复后同一实例 `reward=0.994`（非 `-2.0`），`value==native 最优`。

三处修复后 `tests/solver/test_rl_decide.py`（含此前 3 个失败）+ `test_decide_omt.py` 全绿。

## 6. "单次采样 2–3min" 的另一半：GNN refocus

本文修复的是**无 GNN 也慢**的传播开销。剩余耗时的大头在 GNN 侧：每 `refocus_every`
次决策要 `build_bool_snapshot` + `GraphBuilder.build` + `policy.infer` 一次；难实例上决策数
上万 → 数百次 refocus × 每次 GNN 推理，单实例 collect 可达分钟级（实测一较难实例 collect
> 4min）。这属**预期的 GNN 成本**，非 bug。想压：加大 `refocus_every`、缓存图骨架
（`_STATIC_CACHE` 已缓存静态结构，可进一步复用图张量）、或 batch 推理。
