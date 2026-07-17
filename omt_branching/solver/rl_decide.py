"""Decide 层 RL：采样式 decider（含 defer-to-VSIDS 动作）+ REINFORCE 训练器。

**窗口粘性**（``sticky_window=True``，训练默认）：每个 ``refocus_every`` 窗口只跑一次
GNN，对 ``[defer, 当时未定原子]`` 采样一次并记一条 REINFORCE step；窗口内后续 decide
不再采样/进 torch——defer 则整窗放行 VSIDS，否则粘性返回采样原子，已定后按缓存分数贪心。

``sticky_window=False`` 恢复旧行为（每次 decide 都采样并记 step）。
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
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

DEFAULT_RL_COLLECT_WORKERS = 4
MIN_INSTANCES_FOR_RL_PARALLEL = 8


def _policy_state_cpu(policy: BranchingPolicy) -> dict:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def decide_rl_reward(res: dict, ref_val, ref_rlimit) -> float:
    """由 ``solve_omt_with_decider`` 返回值计算 REINFORCE 奖励。
    若目标值不一致或weighted未生成，说明迭代超过上限，惩罚设置为 -2.0
    否则根据weighted / ref 归一化给出 [-1, 1] 的奖励
    """
    if ref_val is not None and (res.get("value") is None or res["value"] != ref_val):
        return -2.0
    # 键名须与 solve_omt_with_decider 的输出一致（"weighted rlimit"，含空格）；
    # 早前用 "weighted_rlimit" 恒取到 None -> reward 恒 -2.0 -> REINFORCE 无学习信号。
    weighted = res.get("weighted rlimit")
    if weighted is None or ref_rlimit is None or ref_rlimit <= 0:
        return -2.0
    ratio = min(1.0 * weighted / ref_rlimit, 2.0)
    return 1.0 - ratio


def effective_rl_workers(
    count: int,
    requested: int,
    *,
    min_instances: int = MIN_INSTANCES_FOR_RL_PARALLEL,
) -> int:
    """小批量或 requested<=1 时退回串行，避免进程启动开销超过 z3 收益。"""
    if requested <= 1 or count < min_instances:
        return 1
    cpus = os.cpu_count() or 1
    return max(1, min(requested, count, cpus))


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
    def __init__(
        self,
        policy: BranchingPolicy,
        defer_logit,
        assertions,
        refocus_every: int = 50,
        sample: bool = True,
        device: str | torch.device = "cpu",
        *,
        sticky_window: bool = True,
    ):
        self.policy = policy
        self.defer_logit = defer_logit  # torch 标量（trainer 持有的可学参数）
        self.device = device
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self.sample = sample
        # True：每窗只采样/记 step 一次，其后按缓存分数立即返回（训练默认）
        self.sticky_window = sticky_window
        self._graph = None
        self._scores = None  # detached bool_branch_scores（CPU）
        self._phases = None  # detached phase_logits（CPU），>0 → True
        self._bmap: dict = {}
        self._score_by_key: dict[str, float] = {}
        self._phase_by_key: dict[str, bool] = {}
        self._window_defer: bool = False
        self._window_choice: Optional[str] = None
        self._window_phase: bool = True
        self._since = self.refocus_every
        self.steps: list = []

    def _undecided_pairs(self, undecided_keys):
        if self._scores is None or self._scores.numel() == 0:
            return [], []
        keys, locs = [], []
        n = self._scores.numel()
        for k in undecided_keys:
            loc = self._bmap.get(k)
            if loc is not None and loc < n:
                keys.append(k)
                locs.append(loc)
        return keys, locs

    def _rebuild_key_tables(self) -> None:
        """从当前 ``_scores``/``_phases``/``_bmap`` 建 key→分数表（供窗口内贪心）。"""
        self._score_by_key = {}
        self._phase_by_key = {}
        if self._scores is None:
            return
        score_list = self._scores.tolist()
        phase_list = self._phases.tolist() if self._phases is not None else None
        n = len(score_list)
        for k, loc in self._bmap.items():
            if loc < n:
                self._score_by_key[k] = score_list[loc]
                if phase_list is not None and loc < len(phase_list):
                    self._phase_by_key[k] = phase_list[loc] > 0
                else:
                    self._phase_by_key[k] = True

    def _greedy_from_cache(self, undecided_keys) -> Optional[tuple]:
        """窗口内快路径：无 torch，按缓存 GNN 分数取最高未定原子。"""
        best_k = None
        best_s = float("-inf")
        for k in undecided_keys:
            s = self._score_by_key.get(k)
            if s is not None and s > best_s:
                best_s = s
                best_k = k
        if best_k is None:
            return None
        return best_k, self._phase_by_key.get(best_k, True)

    def _sample_window_action(self, undecided_keys) -> Optional[tuple]:
        """在当前缓存分数上采样/argmax 一次，写入 window 状态并 ``steps.append``。"""
        keys, locs = self._undecided_pairs(undecided_keys)
        self._window_defer = False
        self._window_choice = None
        self._window_phase = True
        if not keys or self._graph is None:
            self._window_defer = True
            return None
        defer = self.defer_logit.detach().cpu().reshape(1)
        logits = torch.cat([defer, self._scores[locs]])
        if self.sample:
            idx = int(torch.multinomial(torch.softmax(logits, dim=0), 1).item())
        else:
            idx = int(torch.argmax(logits).item())
        self.steps.append((self._graph, locs, idx))
        if idx == 0:
            self._window_defer = True
            return None
        key = keys[idx - 1]
        phase = self._phase_by_key.get(key, True)
        self._window_choice = key
        self._window_phase = phase
        return key, phase

    def _refocus(self, assignment) -> None:
        snap, _ = build_bool_snapshot(self.assertions, assignment=assignment)
        g = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        g = g.to(self.device)
        out = self.policy.infer(g)
        self._graph = g
        # z3 回调内用 CPU 分数，避免 GPU .item() 同步
        self._scores = out.bool_branch_scores.detach().cpu()
        self._phases = out.phase_logits.detach().cpu()
        self._bmap = g.id_maps.get(NodeType.BOOL_VAR, {})
        self._rebuild_key_tables()

    def __call__(self, undecided_keys, assignment) -> Optional[tuple]:
        if self._since >= self.refocus_every:
            self._refocus(assignment)
            self._since = 0
            if self.sticky_window:
                # 本窗唯一一次采样 / 记 step；返回值即本次回调结果
                self._since += 1
                return self._sample_window_action(undecided_keys)
        self._since += 1

        if self._graph is None or self._scores is None or self._scores.numel() == 0:
            return None

        if self.sticky_window:
            if self._window_defer:
                return None
            # 粘性原子仍未定：直接返回（无采样）
            if self._window_choice is not None and self._window_choice in undecided_keys:
                return self._window_choice, self._window_phase
            # 粘性原子已定：按缓存分数贪心覆盖其余未定
            return self._greedy_from_cache(undecided_keys)

        # 旧行为：每次 decide 都采样并记 step（消融 / 对比用）
        keys, locs = self._undecided_pairs(undecided_keys)
        if not keys:
            return None
        defer = self.defer_logit.detach().cpu().reshape(1)
        logits = torch.cat([defer, self._scores[locs]])
        probs = torch.softmax(logits, dim=0)
        idx = (
            int(torch.multinomial(probs, 1).item())
            if self.sample
            else int(torch.argmax(probs).item())
        )
        self.steps.append((self._graph, locs, idx))
        if idx == 0:
            return None
        return keys[idx - 1], self._phase_by_key.get(keys[idx - 1], True)


@dataclass
class DecideRLConfig:
    lr: float = 1e-3
    refocus_every: int = 50
    max_iters: int = 100000
    baseline_momentum: float = 0.9
    grad_clip: float = 5.0
    device: str = field(default_factory=gnn_device)
    workers: int = 1
    min_instances_for_parallel: int = MIN_INSTANCES_FOR_RL_PARALLEL


def _make_rl_process_pool(workers: int) -> ProcessPoolExecutor:
    """Linux 上必须用 spawn：父进程已 init CUDA 时 fork 会导致子进程卡死。"""
    ctx = mp.get_context("spawn")
    return ProcessPoolExecutor(max_workers=workers, mp_context=ctx)


def _rl_collect_worker(task: tuple) -> tuple:
    """ProcessPool worker：从 smt2 或按 index/seed 重建实例并 collect。"""
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
        smt2_path,
        instance_id,
        ref_value,
        ref_rlimit,
    ) = task
    if smt2_path:
        from omt_branching.solver.decide_omt import smt2_to_instance

        inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    else:
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
        hard_exprs,
        obj,
        sense,
        decider_factory=factory,
        max_iters=max_iters,
        ref_rlimit=ref_rlimit,
        sample=True,
    )
    steps = holder["d"].steps if "d" in holder else []
    reward = decide_rl_reward(res, ref_value, ref_rlimit)
    return inst_idx, _steps_to_cpu(steps), reward, res


class DecideRLTrainer:
    def __init__(
        self, policy: BranchingPolicy, config: DecideRLConfig = DecideRLConfig()
    ):
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

    def collect(
        self,
        hard,
        objective,
        sense: Sense,
        *,
        ref_value=None,
        ref_rlimit: int | None = None,
    ):
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
            ref_rlimit=ref_rlimit,
            sample=True,
        )
        steps = holder["d"].steps if "d" in holder else []
        reward = decide_rl_reward(res, ref_value, ref_rlimit)
        return steps, reward, res

    def _collect_parallel(
        self,
        count: int,
        *,
        seed: int,
        hard: bool,
        min_vars: int,
        max_vars: int,
        pool: ProcessPoolExecutor,
        pbar=None,
        iter_idx: int = 0,
        smt2_paths: list[str] | None = None,
        instance_ids: list[str] | None = None,
        ref_values: list | None = None,
        ref_rlimits: list[int | None] | None = None,
    ) -> list[tuple[int, list, float, dict]]:
        """多进程 collect；worker 固定 CPU，主进程 update 阶段再用 GPU。"""
        policy_state = _policy_state_cpu(self.policy)
        defer_val = float(self.defer_logit.detach().cpu())
        worker_device = "cpu"
        paths = smt2_paths or [None] * count
        ids = instance_ids or [None] * count
        vals = ref_values if ref_values is not None else [None] * count
        rls = ref_rlimits if ref_rlimits is not None else [None] * count
        if not (len(paths) == len(ids) == len(vals) == len(rls) == count):
            raise ValueError("smt2_paths / instance_ids / ref_* 长度必须等于实例数")
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
                paths[j],
                ids[j],
                vals[j],
                rls[j],
            )
            for j in range(count)
        ]
        results: list[tuple[int, list, float, dict]] = []
        futures = [pool.submit(_rl_collect_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(iter=iter_idx, phase="collect", inst=f"{done}/{count}")
        results.sort(key=lambda x: x[0])
        return results

    def save_checkpoint(self, path, *, meta: dict | None = None) -> None:
        """保存策略权重 + ``defer_logit``（写入 ``meta``）。"""
        from omt_branching.model.persistence import save_policy

        payload_meta = {
            "defer_logit": float(self.defer_logit.detach().cpu()),
            **(meta or {}),
        }
        save_policy(self.policy, path, meta=payload_meta)

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

        res = solve_sat_with_decider(
            list(assertions), list(atoms), decider_factory=lambda a: factory(a)
        )
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
                    print(
                        f"[it {it} inst {j}] loss={stats['loss']:.4f} reward={reward:.3f} "
                        f"conflicts={res['conflicts']} steps={stats['steps']}"
                    )
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
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.parameters()) + [self.defer_logit], self.config.grad_clip
        )
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
        smt2_paths: list[str] | None = None,
        instance_ids: list[str] | None = None,
        ref_values: list | None = None,
        ref_rlimits: list[int | None] | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 1,
    ):
        """REINFORCE 训练。``workers>1`` 且实例数足够时用 spawn 进程池并行 collect。

        ``smt2_paths`` 非空时 worker 从落盘文件读实例（与已有 dataset 对齐），否则按
        seed 重建。``ref_values`` / ``ref_rlimits`` 为各实例的 binary 参考（传入
        ``solve_omt_with_decider`` 供 reward）。``checkpoint_dir`` 非空时每隔
        ``checkpoint_every`` 轮保存中间权重。
        """
        instances = list(instances)
        count = len(instances)
        if smt2_paths is not None and len(smt2_paths) != count:
            raise ValueError("smt2_paths 长度必须等于 instances")
        if instance_ids is not None and len(instance_ids) != count:
            raise ValueError("instance_ids 长度必须等于 instances")
        if ref_values is not None and len(ref_values) != count:
            raise ValueError("ref_values 长度必须等于 instances")
        if ref_rlimits is not None and len(ref_rlimits) != count:
            raise ValueError("ref_rlimits 长度必须等于 instances")
        vals = ref_values if ref_values is not None else [None] * count
        rls = ref_rlimits if ref_rlimits is not None else [None] * count
        requested = workers if workers is not None else self.config.workers
        n_workers = effective_rl_workers(
            count,
            requested,
            min_instances=self.config.min_instances_for_parallel,
        )
        use_parallel = n_workers > 1
        history = []
        pool: ProcessPoolExecutor | None = None
        if use_parallel:
            pool = _make_rl_process_pool(n_workers)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        try:
            with tqdm(total=iterations * count, desc="rl_train") as pbar:
                for it in range(iterations):
                    if use_parallel:
                        assert pool is not None
                        batch = self._collect_parallel(
                            count,
                            seed=collect_seed,
                            hard=collect_hard,
                            min_vars=collect_min_vars,
                            max_vars=collect_max_vars,
                            pool=pool,
                            pbar=pbar,
                            iter_idx=it,
                            smt2_paths=smt2_paths,
                            instance_ids=instance_ids,
                            ref_values=vals,
                            ref_rlimits=rls,
                        )
                        for upd_i, (j, steps, reward, res) in enumerate(batch):
                            stats = self.update(steps, reward, key=j)
                            stats.update(
                                {
                                    "iter": it,
                                    "instance": j,
                                    "rlimit": res["rlimit"],
                                    "conflicts": res["conflicts"],
                                    "ref_rlimit": res.get("ref_rlimit"),
                                    "match": (
                                        res.get("ref_value") is None
                                        or res.get("value") == res.get("ref_value")
                                    ),
                                }
                            )
                            history.append(stats)
                            if log:
                                print(
                                    f"[it {it} inst {j}] loss={stats['loss']:.4f} "
                                    f"reward={reward:.3f} rlimit={res['rlimit']} "
                                    f"conflicts={res['conflicts']} steps={stats['steps']}"
                                )
                            pbar.set_postfix(
                                iter=it,
                                phase="update",
                                inst=f"{upd_i + 1}/{count}",
                            )
                    else:
                        for j, (hard, obj, sense) in enumerate(instances):
                            steps, reward, res = self.collect(
                                hard,
                                obj,
                                sense,
                                ref_value=vals[j],
                                ref_rlimit=rls[j],
                            )
                            stats = self.update(steps, reward, key=j)
                            stats.update(
                                {
                                    "iter": it,
                                    "instance": j,
                                    "rlimit": res["rlimit"],
                                    "conflicts": res["conflicts"],
                                    "ref_rlimit": res.get("ref_rlimit"),
                                    "match": (
                                        res.get("ref_value") is None
                                        or res.get("value") == res.get("ref_value")
                                    ),
                                }
                            )
                            history.append(stats)
                            if log:
                                print(
                                    f"[it {it} inst {j}] loss={stats['loss']:.4f} "
                                    f"reward={reward:.3f} rlimit={res['rlimit']} "
                                    f"conflicts={res['conflicts']} steps={stats['steps']}"
                                )
                            pbar.update(1)
                    if (
                        checkpoint_dir
                        and checkpoint_every > 0
                        and (it + 1) % checkpoint_every == 0
                    ):
                        ckpt = os.path.join(checkpoint_dir, f"iter_{it + 1:04d}.pt")
                        self.save_checkpoint(
                            ckpt,
                            meta={"iter": it + 1, "iterations": iterations},
                        )
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        return history


__all__ = [
    "DEFAULT_RL_COLLECT_WORKERS",
    "MIN_INSTANCES_FOR_RL_PARALLEL",
    "effective_rl_workers",
    "decide_rl_reward",
    "SamplingPolicyDecider",
    "DecideRLConfig",
    "DecideRLTrainer",
]
