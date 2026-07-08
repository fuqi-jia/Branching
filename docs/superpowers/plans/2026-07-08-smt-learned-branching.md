# 困难 SMT(LIA) theory-atom 学习分支 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 SAT 正结果扩展到 SMT theory-atom：在困难 SMT(LIA) 单次可满足性检查上，让训练后的
learned-decide 在 conflicts 上优于 VSIDS-decide（分支于线性算术原子）。

**Architecture:** 几乎全复用 SAT 管道（`solve_sat_with_decider`/`build_bool_snapshot`/
`lookahead_scores`/`DecideRLTrainer.collect_sat` 均 atom-type-agnostic）。仅新增紧 SMT(LIA) 生成器
+ 实验；理论原子快照特征为应急（仅 imitation 冻结时加）。

**Tech Stack:** Python 3.10 / z3-solver 4.15.4 / PyTorch；pytest。

## Global Constraints

- 运行/测试：`conda run -n omt python -m pytest tests/solver`。
- 不改 z3；两臂均附 propagator（关 z3 预处理→纯 CDCL；已验证紧 SMT(LIA) 附 prop 得 610–786 conflicts）。
- docstring/注释中文；标识符/类型英文。
- 复用（原样）：`solve_sat_with_decider(assertions, atoms, decider_factory=None) -> {result,conflicts,decisions,rlimit}`；`build_bool_snapshot`；`collect_atoms(assertions)`（识别比较原子）；`lookahead_scores(assertions, atoms, config)`；`build_lookahead_examples_sat(problems=list[(atoms,clauses)], config)`；`DecideRLTrainer(policy, DecideRLConfig).train_sat(problems=list[(atoms,clauses)], iterations)`；`PolicyDecider(service, assertions, refocus_every)`；`ImitationTrainer`。

---

### Task 1: 困难 SMT(LIA) 生成器

**Files:**
- Modify: `omt_branching/solver/sat_instances.py`
- Modify: `omt_branching/solver/__init__.py`
- Test: `tests/solver/test_sat_instances.py`（追加）

**Interfaces:**
- Consumes: `collect_atoms`（`propagator_snapshot`）。
- Produces: `generate_hard_smt_lia(n_vars=8, n_disj=30, k=3, ub=6, chi=4, seed=0) -> tuple[list, list]`（(atoms, clauses)；atoms=理论原子）。

- [ ] **Step 1: 写失败测试**

```python
# tests/solver/test_sat_instances.py 追加
def test_hard_smt_lia_theory_atoms_and_conflicts():
    from omt_branching.solver.sat_instances import generate_hard_smt_lia
    from omt_branching.solver.sat_solve import solve_sat_with_decider
    atoms, clauses = generate_hard_smt_lia(n_vars=8, n_disj=30, k=3, ub=6, chi=4, seed=1)
    # 原子是线性算术比较（含加法/整数变量），非纯布尔常量
    assert len(atoms) > 20
    assert all(z3.is_bool(a) and a.num_args() >= 1 for a in atoms)   # 比较原子有参数
    # 附 propagator 关预处理 -> 大量 conflicts（headroom）
    r = solve_sat_with_decider(clauses, atoms, decider_factory=None)
    assert r["result"] in ("sat", "unsat")
    assert r["conflicts"] > 100                                       # 数百 conflicts
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_instances.py::test_hard_smt_lia_theory_atoms_and_conflicts -q`
Expected: FAIL（`ImportError: generate_hard_smt_lia`）

- [ ] **Step 3: 写实现**

在 `sat_instances.py` 末尾（`__all__` 之前）新增：

```python
def generate_hard_smt_lia(n_vars: int = 8, n_disj: int = 30, k: int = 3,
                          ub: int = 6, chi: int = 4, seed: int = 0):
    """紧随机 SMT(LIA)：``n_vars`` 整数变量 + 盒约束；``n_disj`` 个 ``k`` 元析取(线性原子)。
    系数 [-chi,chi]、小域 [0,ub] 使布尔搜索成瓶颈（附 propagator 数百 conflicts）。返回
    (atoms, clauses)，``atoms``=出现的理论原子（供 propagator 分支）。
    """
    from omt_branching.solver.propagator_snapshot import collect_atoms

    rng = random.Random(seed)
    xs = [z3.Int(f"y{i}") for i in range(n_vars)]
    clauses = []
    for x in xs:
        clauses.append(z3.Or(x >= 0))          # 盒下界（原子形式，便于 collect_atoms 收录）
        clauses.append(z3.Or(x <= ub))
    for _ in range(n_disj):
        lits = []
        for _ in range(k):
            c = [rng.randint(-chi, chi) for _ in range(n_vars)]
            if all(v == 0 for v in c):
                c[rng.randrange(n_vars)] = 1
            lhs = z3.Sum([cc * x for cc, x in zip(c, xs)])
            b = rng.randint(-ub, ub * chi)
            lits.append(lhs <= b if rng.random() < 0.5 else lhs >= b)
        clauses.append(z3.Or(*lits))
    atoms = collect_atoms(clauses)
    return atoms, clauses
```

在 `__all__` 加入 `"generate_hard_smt_lia"`；在 `omt_branching/solver/__init__.py` 的 sat_instances
import 与 `__all__` 加入它（与 `generate_php`/`generate_rand_3sat` 并列）。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_instances.py -q`
Expected: PASS（3 passed）

- [ ] **Step 5: 提交**

```bash
git add omt_branching/solver/sat_instances.py omt_branching/solver/__init__.py tests/solver/test_sat_instances.py
git commit -m "feat: generate_hard_smt_lia（紧随机 SMT(LIA)，理论原子 + 数百 conflicts）"
```

---

### Task 2: SMT look-ahead imitation 可学习性护栏（verify step）

**Files:**
- Test: `tests/solver/test_sat_solve.py`（追加）

**Interfaces:**
- Consumes: `generate_hard_smt_lia`（Task 1）、`build_lookahead_examples_sat`、`ImitationTrainer`。
- Produces: 无新代码——**LIA 教训的 verify 步**：确认理论原子 look-ahead 标签可被 GNN 学（branch 损失下降）。若冻结 -> 触发应急（§3.3：加理论原子结构特征）。

- [ ] **Step 1: 写测试**

```python
# tests/solver/test_sat_solve.py 追加
def test_smt_lia_lookahead_imitation_learnable():
    import torch
    from omt_branching.solver.sat_instances import generate_hard_smt_lia
    from omt_branching.solver.training_data import build_lookahead_examples_sat
    from omt_branching.model.policy import BranchingPolicy
    from omt_branching.model.trainer import ImitationTrainer, TrainConfig

    torch.manual_seed(0)
    problems = [generate_hard_smt_lia(8, 30, 3, 6, 4, seed=s) for s in range(6)]
    exs = [e for e in build_lookahead_examples_sat(problems) if e.bool_target_scores]
    assert exs, "应有带 bool 标签的样本（理论原子）"
    policy = BranchingPolicy()
    h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=12)
    assert "branch" in h[0]
    assert h[-1]["branch"] < h[0]["branch"]    # 理论原子 look-ahead 可学（否则触发理论特征应急）
```

- [ ] **Step 2: 运行**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_solve.py::test_smt_lia_lookahead_imitation_learnable -q`
Expected: PASS。**若 FAIL（branch 不降）**：说明子句图不足以预测理论传播（LIA 教训重演）——
应急：改 `build_bool_snapshot` 增 `TheoryAtomInfo(var_coeffs, rhs)` + 数值变量节点 + 原子↔变量边
（复刻 `extractor.py` 的 `_linear` 分解），使 GNN 见共享变量结构；然后重跑本护栏。**先跑，按结果决定。**

- [ ] **Step 3: （视结果——通过则无代码；冻结则加理论特征）**

若通过：跳过。若冻结：按 Step 2 应急改 `build_bool_snapshot`（新增理论原子/变量结构），使本测试通过。

- [ ] **Step 4: 重跑确认**

Run: `conda run -n omt python -m pytest tests/solver/test_sat_solve.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add tests/solver/test_sat_solve.py
git commit -m "test: SMT(LIA) 理论原子 look-ahead imitation 可学护栏(verify step)"
# 若触发应急改了 build_bool_snapshot，一并 add omt_branching/solver/propagator_snapshot.py
```

---

### Task 3: SMT 实验 + 冒烟 + 关键测量

**Files:**
- Create: `examples/smt_branch.py`
- Test: `tests/solver/test_demo_smoke.py`（追加）

**Interfaces:**
- Consumes: `generate_hard_smt_lia`、`solve_sat_with_decider`、`build_lookahead_examples_sat`、`DecideRLTrainer`、`PolicyDecider`。
- Produces: `examples/smt_branch.py`（困难 SMT(LIA)，imitation+RL，learned-decide vs VSIDS conflicts，多 seed）。

- [ ] **Step 1: 写实验脚本**

```python
# examples/smt_branch.py
"""困难 SMT(LIA) theory-atom 学习分支：两臂均附 propagator（关预处理→纯 CDCL），比
VSIDS-decide vs (imitation+RL) learned-decide 的 conflicts。原子=线性算术比较。多 seed mean±std。
"""
from __future__ import annotations

import argparse
import statistics

import torch

from omt_branching.model.policy import BranchingPolicy
from omt_branching.model.trainer import ImitationTrainer, TrainConfig
from omt_branching.service import BranchingPolicyService
from omt_branching.solver import solve_sat_with_decider
from omt_branching.solver.sat_instances import generate_hard_smt_lia
from omt_branching.solver.policy_decider import PolicyDecider
from omt_branching.solver.rl_decide import DecideRLConfig, DecideRLTrainer
from omt_branching.solver.training_data import build_lookahead_examples_sat


def _bench(problems, decider_factory):
    return [solve_sat_with_decider(clauses, atoms, decider_factory=decider_factory)["conflicts"]
            for atoms, clauses in problems]


def main() -> None:
    ap = argparse.ArgumentParser(description="困难 SMT(LIA) 学习分支：VSIDS vs learned")
    ap.add_argument("--n-vars", type=int, default=8)
    ap.add_argument("--n-disj", type=int, default=30)
    ap.add_argument("--test", type=int, default=12)
    ap.add_argument("--train", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--rl-iters", type=int, default=0)
    ap.add_argument("--refocus", type=int, default=100)
    args = ap.parse_args()
    torch.manual_seed(0)

    test = [generate_hard_smt_lia(args.n_vars, args.n_disj, 3, 6, 4, 1000 + s)
            for s in range(args.test)]

    policy = BranchingPolicy()
    if args.train > 0:
        tr = [generate_hard_smt_lia(args.n_vars, args.n_disj, 3, 6, 4, s) for s in range(args.train)]
        exs = [e for e in build_lookahead_examples_sat(tr) if e.bool_target_scores]
        h = ImitationTrainer(policy, TrainConfig(lr=5e-3)).fit(exs, epochs=args.epochs)
        print(f"imitation: {len(exs)} 样本, branch {h[0].get('branch',0):.3f}->{h[-1].get('branch',0):.3f}")
    if args.rl_iters > 0:
        rl = [generate_hard_smt_lia(args.n_vars, args.n_disj, 3, 6, 4, s) for s in range(max(args.train, 20))]
        rlt = DecideRLTrainer(policy, DecideRLConfig(refocus_every=args.refocus))
        hh = rlt.train_sat(rl, iterations=args.rl_iters, log=False)
        if hh:
            print(f"RL: {len(hh)} 步, 末条 conflicts={hh[-1]['conflicts']}, defer_logit={float(rlt.defer_logit):.3f}")
    svc = BranchingPolicyService(policy=policy)

    v = _bench(test, None)
    ln = _bench(test, lambda assertions: PolicyDecider(svc, assertions, args.refocus))
    print(f"[SMT-LIA] VSIDS conflicts={statistics.fmean(v):.0f}±{statistics.pstdev(v):.0f} | "
          f"learned={statistics.fmean(ln):.0f}±{statistics.pstdev(ln):.0f} | "
          f"胜={'是' if statistics.fmean(ln) < statistics.fmean(v) else '否'}")
    print("两臂均附 propagator（关预处理→纯 CDCL）；分支于线性算术原子；conflicts 越少越好。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 追加冒烟测试**

在 `tests/solver/test_demo_smoke.py` 追加：
```python
def test_smt_branch_smoke():
    out = subprocess.run(
        [sys.executable, "-m", "examples.smt_branch",
         "--n-vars", "6", "--n-disj", "15", "--test", "3", "--train", "4", "--epochs", "3"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr
    assert "SMT-LIA" in out.stdout
```

- [ ] **Step 3: 跑关键测量 + 全量**

Run: `conda run -n omt python -m examples.smt_branch --n-vars 8 --n-disj 30 --test 12 --train 30 --epochs 25 --rl-iters 2`
Expected: 打印 imitation loss、RL、SMT-LIA VSIDS vs learned conflicts + 胜负。**关键看**：trained
learned<VSIDS（且需 trained≫untrained：另跑 --train 0 --rl-iters 0 对照）。诚实报告胜/平/负。

Run: `conda run -n omt python -m pytest tests/solver -q`
Expected: 全绿。

- [ ] **Step 4: 提交**

```bash
git add examples/smt_branch.py tests/solver/test_demo_smoke.py
git commit -m "feat: SMT(LIA) 学习分支实验(理论原子, VSIDS vs learned conflicts) + 冒烟"
```

---

## Self-Review

**Spec 覆盖：** §3.1 生成器→Task 1；§3.3 verify-then-extend→Task 2；§3.4 实验→Task 3。§3.2 复用
（无需新任务）。

**Placeholder scan：** 无 TBD/TODO；每步含完整代码。Task 2 Step 3 的"应急"是条件分支（按 Step 2
结果），非 placeholder——先跑护栏，冻结才加理论特征（YAGNI）。

**Type consistency：** `generate_hard_smt_lia(...) -> (atoms, clauses)` Task 1 定义、Task 2/3 一致；
`build_lookahead_examples_sat(problems=list[(atoms,clauses)])`、`train_sat(problems=list[(atoms,clauses)])`、
`solve_sat_with_decider(clauses, atoms, decider_factory)` 均与既有一致（生成器返回序 (atoms,clauses)）。

**风险提示：** Task 2 若 branch 不降=子句图不足预测理论传播 → 加理论原子结构（LIA 教训）。Task 3 若
learned 未胜=如实报告；查 trained vs untrained、refocus 频率、RL 收敛。
