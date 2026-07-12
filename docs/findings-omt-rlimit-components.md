# 对OMT求解中rlimit cost来源的分析

> 运行配置：
> - 测试入口：examples/decide_branch.py
> - 规模参数：
>   - train = 60
>   - test = 15
>   - min_vars = 5
>   - max_vars = 7
>   - rl_iters = 3
>   - 其余为默认值

## 1. 在solve_omt_with_decider中对rlimit进行分类

solve_omt_with_decider中与z3.Solver交互可构建为如下流程：

```

初始化Solver
    |
    ▼
根据decider_factory
构建Propagator
    |
    ▼
添加原公式约束
    |
    ▼
check()◀───────────┐
    |               │
    ▼               │
eval()              │
    |               │
    ▼               │
根据eval            │
添加better_cut──────┘

```

在各部分提取rlimit读数，分类为以下几部分：

- solver_rlimit: 初始化消耗
- decider_factory_rlimit: 构建Propagator消耗；对于VSIDS应为0
- model_base_rlimit: 添加原公式约束消耗
- check_rlimit: check()消耗
- eval_rlimit: eval()消耗
- model_cut_rlimit: 添加better_cut消耗
- weighted_rlimit: 根据check_rlimit和eval_rlimit对迭代次数加权计算得到，代替总rlimit用于reward

## 2. 测试结果

|类别|VSIDS|learned|+/-|
|---|---|---|---|
| solver_rlimit | 3725227928 | 3725583716 | +355788 |
| decider_factory_rlimit | 0 | 1610 | +1610 |
| model_base_rlimit | 2144 | 2144 | 0 |
| check_rlimit | 306713 | 188100 | -118,613 |
| eval_rlimit | 22087 | 4169 | -17,918 |
| model_cut_rlimit | 20933 | 4008 | -16,925 |
| weighted_rlimit | 14393481 | 1780964 | - |

## 3. 结论

此前 learned 不及 VSIDS 的结论来源于**初始化Solver时**rlimit消耗量的巨大差异，在对rlimit详细分析后可以发现RL微调能够大幅降低OMT求解消耗。