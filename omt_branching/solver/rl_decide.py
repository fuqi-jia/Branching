"""Decide 层 RL：采样式 decider（含 defer-to-VSIDS 动作）+ REINFORCE 训练器。

每个 decide 对 ``[defer_logit, 未定原子 bool 分数]`` softmax 采样：采到 defer -> return None
(退回 VSIDS)，否则覆盖采样原子。记录 (refocus 图, 未定局部索引, 采样索引) 供 REINFORCE 重算
log-prob。奖励 = −log1p(rlimit)，per-instance EMA baseline。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import build_bool_snapshot


class SamplingPolicyDecider:
    def __init__(self, policy: BranchingPolicy, defer_logit, assertions,
                 refocus_every: int = 50, sample: bool = True):
        self.policy = policy
        self.defer_logit = defer_logit          # torch 标量（trainer 持有的可学参数）
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self.sample = sample
        self._graph = None
        self._scores = None                      # detached bool_branch_scores
        self._bmap: dict = {}
        self._since = self.refocus_every
        self.steps: list = []

    def _refocus(self, assignment):
        snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
        g = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        out = self.policy.infer(g)
        self._graph = g
        self._scores = out.bool_branch_scores.detach()
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
        logits = torch.cat([self.defer_logit.detach().reshape(1), self._scores[locs]])
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
    device: str = "cpu"


class DecideRLTrainer:
    def __init__(self, policy: BranchingPolicy, config: DecideRLConfig = DecideRLConfig()):
        self.policy = policy.to(config.device)
        self.config = config
        self.defer_logit = torch.nn.Parameter(torch.zeros((), device=config.device))
        self.opt = torch.optim.Adam(list(policy.parameters()) + [self.defer_logit], lr=config.lr)
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
            d = SamplingPolicyDecider(self.policy, self.defer_logit, assertions,
                                      self.config.refocus_every, sample=True)
            holder["d"] = d
            return d

        res = solve_omt_with_decider(hard, objective, sense,
                                     decider_factory=factory, max_iters=self.config.max_iters)
        steps = holder["d"].steps if "d" in holder else []
        reward = -math.log1p(res["rlimit"])
        return steps, reward, res

    def collect_sat(self, assertions, atoms):
        from omt_branching.solver.sat_solve import solve_sat_with_decider

        holder: dict = {}

        def factory(asserts):
            d = SamplingPolicyDecider(self.policy, self.defer_logit, asserts,
                                      self.config.refocus_every, sample=True)
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
        for g, locs, idx in steps:
            gid = id(g)
            if gid not in cache:
                cache[gid] = self.policy(g).bool_branch_scores
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

    def train(self, instances, iterations: int = 1, log: bool = False):
        instances = list(instances)
        history = []
        for it in range(iterations):
            for j, (hard, obj, sense) in enumerate(instances):
                steps, reward, res = self.collect(hard, obj, sense)
                stats = self.update(steps, reward, key=j)
                stats.update({"iter": it, "instance": j, "rlimit": res["rlimit"],
                              "conflicts": res["conflicts"]})
                history.append(stats)
                if log:
                    print(f"[it {it} inst {j}] loss={stats['loss']:.4f} reward={reward:.3f} "
                          f"rlimit={res['rlimit']} conflicts={res['conflicts']} steps={stats['steps']}")
        return history


__all__ = ["SamplingPolicyDecider", "DecideRLConfig", "DecideRLTrainer"]
