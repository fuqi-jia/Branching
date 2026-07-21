# 分支敏感 OMT 实例：CSL≈VSIDS 且 oracle 可降 rlimit

> 日期 2026-07-19。代码：`omt_branching/solver/instance_gen.py::generate_branch_focus_lia_*`，
> 探针 `examples/probe_branch_focus.py`；落盘目录
> `examples/artifacts/dataset_branch_focus/`。

## 目标

构造使**布尔分支选择**成为可测量瓶颈的 OMT 实例，满足：

1. **check-sat-loop ≈ 公平 VSIDS**：`|rlimit_csl - rlimit_vsids| / rlimit_csl` 较小
   （挂 propagator 恒 defer 不显著扰动搜索）；
2. **理论 headroom**：种植守卫上的 oracle 优先分支使 `rlimit_oracle / rlimit_vsids` 明显小于 1
   （优化分支选择后 learned 臂*有空间*压低 rlimit）。

## 构造

`branch_focus` 族（witness 驱动、保证 SAT）：

- 选择变量 `x0` 切成多个互斥模式（`Or` of `And(lo≤x0≤hi, …)`）；
- **最优模式**约束松、目标可达高值；**次优模式**紧 packing + 大量析取，且压低高目标变量；
- 顶层再挂大量**不含模式守卫**的干扰析取，抬高 VSIDS 对干扰原子的活跃度；
- oracle：优先以正极性分支模式 0 守卫（`x0` 上下界原子）。

原始随机候选约 15–25% 同时满足 gap / ratio 阈值；探针后验筛选落盘。

## 默认筛选阈值

| 条件 | 默认 |
|---|---|
| CSL–VSIDS 相对 gap | ≤ 0.30 |
| oracle / VSIDS rlimit | ≤ 0.70（至少省 30%） |
| VSIDS rlimit 下限 | ≥ 2000（排除过易） |
| 三臂最优 value | 一致 |

## 实测（seed=0 / 200）

| 划分 | 候选 | 合格 | gap mean | oracle/vsids mean |
|---|---|---|---|---|
| test（seed=0） | 40 | **8** | 0.154 | **0.462**（约 −54%） |
| train（seed=200） | 60 | **7** | 0.197 | **0.416**（约 −58%） |

代表实例 `bfocus11`：CSL=762k，VSIDS=776k（gap=1.8%），oracle=197k（ratio=0.25）。

## 用法

```bash
# 探针 + 落盘（test）
python -m examples.probe_branch_focus --candidates 40 --need 8 --save

# 训练集（换 seed / split）
python -m examples.probe_branch_focus --seed 200 --split train --candidates 60 --need 12 --save \
  --dataset-dir examples/artifacts/dataset_branch_focus

# 随后写 ref 缓存，再跑 decide_branch
python -m examples.solve_dataset_binary --dataset-dir examples/artifacts/dataset_branch_focus
```

## 诚实边界

- **后验筛选**：生成器提高「可出现 headroom」的先验概率，不保证每个样本都合格；必须经探针。
- **oracle ≠ learned**：oracle 用种植守卫序；GNN 仍需从特征学到等价策略，本结果只证明
  **分支 headroom 存在**且与 CSL/VSIDS 公平口径相容。
- CSL–VSIDS 仍偶发大 gap（`add_decide` 固有扰动）；放宽 gap 阈值会提高产量但削弱「两臂接近」。
