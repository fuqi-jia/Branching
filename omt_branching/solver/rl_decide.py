"""Decide 层 RL：采样式 decider（含 defer-to-VSIDS 动作）+ REINFORCE 训练器。

每个 decide 对 ``[defer_logit, 未定原子 bool 分数]`` softmax 采样：采到 defer -> return None
(退回 VSIDS)，否则覆盖采样原子。记录 (refocus 图, 未定局部索引, 采样索引) 供 REINFORCE 重算
log-prob。奖励 = −log1p(rlimit)，per-instance EMA baseline。
"""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import torch

from omt_branching.model.device import gnn_device
from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import build_bool_snapshot

from tqdm import tqdm

DEFAULT_RL_COLLECT_WORKERS = 12


def _policy_state_cpu(policy: BranchingPolicy) -> dict:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def _steps_to_cpu(steps) -> list:
    """进程间回传前把图张量落到 CPU。"""
    out = []
    for g, locs, idx in steps:
        if g.node_features:
            dev = next(iter(g.node_features.values())).device
            g = g.to("cpu") if dev.type != "cpu" else g
        out.append((g, locs, idx))
    return out


class SamplingPolicyDecider:
    def __init__(self, policy: BranchingPolicy, defer_logit, assertions,
                 refocus_every: int = 50, sample: bool = True,
                 device: str | torch.device = "cpu"):
        self.policy = policy
        self.defer_logit = defer_logit          # torch 标量（trainer 持有的可学参数）
        self.device = device
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self.sample = sample
        self._graph = None
        self._scores = None                      # detached bool_branch_scores（CPU 采样）
        self._bmap: dict = {}
        self._since = self.refocus_every
        self.steps: list = []

    def _refocus(self, assignment):
        snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
        g = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        g = g.to(self.device)
        out = self.policy.infer(g)
        self._graph = g
        # z3 回调内用 CPU 分数采样，避免 GPU .item() 同步
        self._scores = out.bool_branch_scores.detach().cpu()
        self._bmap = g.id_maps.get(NodeType.BOOL_VAR, {})

    def __call__(self, undecided_keys, assignment) -> Optional[tuple]:
        if self._since >= self.refocus_every:
            self._refocus(assignment)
            self._since = 0
        self._since += 1
        if self._graph is None or self._scores is None or self._scores.numel() == 0:
            return None
        pairs = [(k, self._bmap.get(k)) for k in undecided_keys]
        pairs = [(k, l) for k, l in pairs if l is not None and l < self._scores.numel()]
        if not pairs:
            return None
        keys = [k for k, _ in pairs]
        locs = [l for _, l in pairs]
        defer = self.defer_logit.detach().cpu().reshape(1)
        logits = torch.cat([defer, self._scores[locs]])
        probs = torch.softmax(logits, dim=0)
        idx = int(torch.multinomial(probs, 1).item()) if self.sample else int(torch.argmax(probs).item())
        self.steps.append((self._graph, locs, idx))
        if idx == 0:
            return None                          # defer -> VSIDS
        return keys[idx - 1], True               # 覆盖采样原子（相位取真）


@dataclass
class DecideRLConfig:
    lr: float = 1e-3
    refocus_every: int = 50
    max_iters: int = 100000
    baseline_momentum: float = 0.9
    grad_clip: float = 5.0
    device: str = field(default_factory=gnn_device)
    workers: int = 1


def _rl_collect_worker(task: tuple) -> tuple:
    """ProcessPool worker：按 index/seed 重建实例并 collect（独立 z3 Context）。"""
    (
        inst_idx,
        seed,
        hard,
        min_vars,
        max_vars,
        policy_state,
        defer_val,
        refocus_every,
        max_iters,
        device,
    ) = task
    from omt_branching.solver.instance_gen import bool_lia_instance_at

    inst = bool_lia_instance_at(
        inst_idx, seed, hard=hard, min_vars=min_vars, max_vars=max_vars
    )
    hard_exprs, obj, sense = inst.as_tuple()

    policy = BranchingPolicy()
    policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()
    defer = torch.nn.Parameter(torch.tensor(defer_val, dtype=torch.float32))

    holder: dict = {}

    def factory(assertions):
        d = SamplingPolicyDecider(
            policy,
            defer,
            assertions,
            refocus_every,
            sample=True,
            device=device,
        )
        holder["d"] = d
        return d

    res = solve_omt_with_decider(
        hard_exprs, obj, sense, decider_factory=factory, max_iters=max_iters
    )
    steps = holder["d"].steps if "d" in holder else []
    reward = -math.log1p(res["weighted rlimit"])
    return inst_idx, _steps_to_cpu(steps), reward, res


class DecideRLTrainer:
    def __init__(self, policy: BranchingPolicy, config: DecideRLConfig = DecideRLConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.defer_logit = torch.nn.Parameter(torch.zeros((), device=config.device))
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + [self.defer_logit], lr=config.lr
        )
        self._baselines: dict = {}
        self._baseline = 0.0

    def _baseline_for(self, key):
        return self._baselines.get(key, self._baseline)

    def _update_baseline_for(self, key, value):
        m = self.config.baseline_momentum
        if key in self._baselines:
            self._baselines[key] = m * self._baselines[key] + (1 - m) * value
        else:
            self._baselines[key] = value

    def collect(self, hard, objective, sense: Sense):
        holder: dict = {}

        def factory(assertions):
            d = SamplingPolicyDecider(
                self.policy,
                self.defer_logit,
                assertions,
                self.config.refocus_every,
                sample=True,
                device=self.config.device,
            )
            holder["d"] = d
            return d

        res = solve_omt_with_decider(
            hard,
            objective,
            sense,
            decider_factory=factory,
            max_iters=self.config.max_iters,
        )
        steps = holder["d"].steps if "d" in holder else []
        reward = -math.log1p(res["weighted rlimit"])
        return steps, reward, res

    def _collect_parallel(
        self,
        count: int,
        *,
        seed: int,
        hard: bool,
        min_vars: int,
        max_vars: int,
        workers: int,
    ) -> list[tuple[int, list, float, dict]]:
        """多进程 collect；返回按 inst_idx 排序的 (idx, steps, reward, res)。"""
        n_workers = max(1, min(workers, count))
        worker_device = self.config.device if n_workers == 1 else "cpu"
        policy_state = _policy_state_cpu(self.policy)
        defer_val = float(self.defer_logit.detach().cpu())
        tasks = [
            (
                j,
                seed,
                hard,
                min_vars,
                max_vars,
                policy_state,
                defer_val,
                self.config.refocus_every,
                self.config.max_iters,
                worker_device,
            )
            for j in range(count)
        ]
        results: list[tuple[int, list, float, dict]] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_rl_collect_worker, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda x: x[0])
        return results

    def collect_sat(self, assertions, atoms):
        from omt_branching.solver.sat_solve import solve_sat_with_decider

        holder: dict = {}

        def factory(asserts):
            d = SamplingPolicyDecider(
                self.policy,
                self.defer_logit,
                asserts,
                self.config.refocus_every,
                sample=True,
                device=self.config.device,
            )
            holder["d"] = d
            return d

        res = solve_sat_with_decider(list(assertions), list(atoms),
                                     decider_factory=lambda a: factory(a))
        steps = holder["d"].steps if "d" in holder else []
        reward = -math.log1p(res["conflicts"])
        return steps, reward, res

    def train_sat(self, problems, iterations: int = 1, log: bool = False):
        """``problems = list[(atoms, clauses)]``（同生成器返回序）。"""
        problems = list(problems)
        history = []
        for it in range(iterations):
            for j, (atoms, assertions) in enumerate(problems):
                steps, reward, res = self.collect_sat(assertions, atoms)
                stats = self.update(steps, reward, key=j)
                stats.update({"iter": it, "instance": j, "conflicts": res["conflicts"]})
                history.append(stats)
                if log:
                    print(f"[it {it} inst {j}] loss={stats['loss']:.4f} reward={reward:.3f} "
                          f"conflicts={res['conflicts']} steps={stats['steps']}")
        return history

    def update(self, steps, reward, key) -> dict:
        if not steps:
            self._update_baseline_for(key, reward)
            return {"loss": 0.0, "reward": reward, "steps": 0}
        adv = reward - self._baseline_for(key)
        cache: dict = {}
        loss = torch.zeros((), device=self.config.device)
        n = 0
        target = torch.device(self.config.device)
        for g, locs, idx in steps:
            gid = id(g)
            if gid not in cache:
                if g.node_features:
                    dev = next(iter(g.node_features.values())).device
                    g_dev = g if dev == target else g.to(target)
                else:
                    g_dev = g.to(target)
                cache[gid] = self.policy(g_dev).bool_branch_scores
            scores = cache[gid]
            logits = torch.cat([self.defer_logit.reshape(1), scores[locs]])
            logp = torch.log_softmax(logits, dim=0)[idx]
            loss = loss - logp * adv
            n += 1
        loss = loss / n
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + [self.defer_logit],
                                       self.config.grad_clip)
        self.opt.step()
        self._update_baseline_for(key, reward)
        return {"loss": float(loss), "reward": reward, "steps": n}

    def train(
        self,
        instances,
        iterations: int = 1,
        log: bool = False,
        *,
        workers: int | None = None,
        collect_seed: int = 1,
        collect_hard: bool = False,
        collect_min_vars: int = 5,
        collect_max_vars: int = 7,
    ):
        """REINFORCE 训练。``workers>1`` 时用进程池并行 collect（按 seed/index 重建实例）。"""
        instances = list(instances)
        count = len(instances)
        n_workers = workers if workers is not None else self.config.workers
        history = []
        with tqdm(total=iterations * count, desc="rl_train") as pbar:
            for it in range(iterations):
                if n_workers > 1:
                    batch = self._collect_parallel(
                        count,
                        seed=collect_seed,
                        hard=collect_hard,
                        min_vars=collect_min_vars,
                        max_vars=collect_max_vars,
                        workers=n_workers,
                    )
                    for j, steps, reward, res in batch:
                        stats = self.update(steps, reward, key=j)
                        stats.update({
                            "iter": it,
                            "instance": j,
                            "rlimit": res["rlimit"],
                            "conflicts": res["conflicts"],
                        })
                        history.append(stats)
                        if log:
                            print(
                                f"[it {it} inst {j}] loss={stats['loss']:.4f} "
                                f"reward={reward:.3f} rlimit={res['rlimit']} "
                                f"conflicts={res['conflicts']} steps={stats['steps']}"
                            )
                        pbar.update(1)
                else:
                    for j, (hard, obj, sense) in enumerate(instances):
                        steps, reward, res = self.collect(hard, obj, sense)
                        stats = self.update(steps, reward, key=j)
                        stats.update({
                            "iter": it,
                            "instance": j,
                            "rlimit": res["rlimit"],
                            "conflicts": res["conflicts"],
                        })
                        history.append(stats)
                        if log:
                            print(
                                f"[it {it} inst {j}] loss={stats['loss']:.4f} "
                                f"reward={reward:.3f} rlimit={res['rlimit']} "
                                f"conflicts={res['conflicts']} steps={stats['steps']}"
                            )
                        pbar.update(1)
        return history


__all__ = [
    "DEFAULT_RL_COLLECT_WORKERS",
    "SamplingPolicyDecider",
    "DecideRLConfig",
    "DecideRLTrainer",
]
