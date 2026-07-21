"""Decide 层 RL：采样式 decider（含 defer-to-VSIDS 动作）+ REINFORCE 训练器。

**窗口粘性**（``sticky_window=True``）：每个 ``refocus_every`` 窗口只跑一次 GNN，对
``[defer, 当时未定原子]`` 采样一次并记一条 REINFORCE step；窗口内后续 decide 不再
采样/进 torch——defer 则整窗放行 VSIDS，否则粘性返回采样原子，已定后按缓存分数贪心。

``sticky_window=False``（``DecideRLConfig`` / ``decide_branch`` 训练默认）：每次
decide 都采样并记 step。

冲突回退（propagator ``pop`` → :meth:`SamplingPolicyDecider.on_backtrack`，默认开启）
会清空粘性窗并强制下次 decide 立刻 refocus。

**并行 collect**（``workers>1``，方案 C）：

- Z3 在 ``spawn`` :class:`ProcessPoolExecutor` 中并行（独立进程，避免共享 Context）；
- GNN 由主进程 :class:`GpuInferService` 后台线程内的 :class:`GpuInferPool` 排队占用全部
  GPU；worker 经 ``Manager.Queue`` 提交 CPU 图并取回分数（实例间 Z3 与 infer 可重叠）。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import threading
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import torch

from omt_branching.graph.hetero_graph import HeteroGraph
from omt_branching.model.device import gnn_device, resolve_infer_devices
from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.policy import BranchingPolicy
from omt_branching.solver.decide_omt import solve_omt_with_decider
from omt_branching.solver.interfaces import Sense
from omt_branching.solver.propagator_snapshot import (
    build_bool_snapshot,
    merge_root_assignment,
    root_forced_assignment,
)

from tqdm import tqdm

DEFAULT_RL_COLLECT_WORKERS = 4
MIN_INSTANCES_FOR_RL_PARALLEL = 8

# ProcessPool worker 全局（initializer 注入主进程 GpuInferService 的请求队列）
_INFER_REQ = None
_MP_CTX = None


def _policy_state_cpu(policy: BranchingPolicy) -> dict:
    return {k: v.detach().cpu() for k, v in policy.state_dict().items()}


def decide_rl_reward(res: dict, ref_val, ref_rlimit) -> float:
    """由 ``solve_omt_with_decider`` 返回值计算 REINFORCE 奖励。
    若目标值不一致或weighted未生成，说明迭代超过上限，惩罚设置为 -1.0
    否则根据weighted / ref 归一化给出 (-1, 1) 的奖励
    """
    if ref_val is not None and (res.get("value") is None or res["value"] != ref_val):
        return -1.0
    # 键名须与 solve_omt_with_decider 的输出一致（"weighted rlimit"，含空格）；
    # 早前用 "weighted_rlimit" 恒取到 None -> reward 恒 -2.0 -> REINFORCE 无学习信号。
    if ref_rlimit is None or ref_rlimit <= 0:
        return 0.0
    weighted = res.get("weighted rlimit")
    if weighted is None:
        return -1.0
    # ratio = min(1.0 * weighted / ref_rlimit, 2.0)
    # return 1.0 - ratio
    ratio = 1.0 * weighted / ref_rlimit

    def _smooth(x: float) -> float:
        return (1 - x * x) / (1 + x * x)

    return _smooth(ratio)


def sat_conflict_reward(conflicts, ref_conflicts, cap: float = 2.0) -> float:
    """单次 SAT/SMT 可满足性检查的**归一化** conflicts 奖励,恒落在 ``[-1, 0]``。

    锚点:``0`` = 零冲突(理论最优);``-1/cap``(默认 ``-0.5``)= 与参考(VSIDS)持平;
    ``-1`` = 冲突数 ≥ ``cap×`` 参考。单调随 conflicts 递减,且**有界** —— 原
    ``-log1p(conflicts)`` 无界且重尾,离群 episode 会主导 REINFORCE 梯度、训练不稳;有界
    奖励 + 每实例 baseline 才稳定。锚在零冲突(而非 VSIDS 持平)可保留反超 VSIDS 的活梯度
    (胜过 VSIDS 映射到 ``(-0.5, 0)``)。``ref_conflicts`` 缺失/非正时退回 ``1``(保守惩罚)。
    """
    ref = max(int(ref_conflicts), 1) if ref_conflicts else 1
    ratio = min(conflicts / ref, cap)
    return -ratio / cap


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


def _clone_policy(policy: BranchingPolicy, device: str) -> BranchingPolicy:
    """复制架构与权重到 ``device``，供推理副本使用。"""
    spec = getattr(policy.encoder, "spec", DEFAULT_FEATURE_SPEC)
    replica = BranchingPolicy(feature_spec=spec, config=policy.config)
    replica.load_state_dict(policy.state_dict())
    return replica.to(device).eval()


class InferBackend(Protocol):
    """本地 :class:`GpuInferPool` 与远程 :class:`RemoteInferClient` 的共同接口。"""

    def infer(self, g: HeteroGraph) -> tuple[torch.Tensor, torch.Tensor]: ...


class GpuInferPool:
    """同进程多设备排队推理：线程安全。

    空闲设备索引放在 ``queue.Queue`` 中；线程 ``get`` 占用一张卡做 ``infer``，
    ``finally`` 归还。多卡时可并行占用不同 GPU，同卡请求自动串行。
    """

    def __init__(self, policy: BranchingPolicy, devices: list[str]):
        if not devices:
            raise ValueError("GpuInferPool 需要至少一个 device")
        self.devices = list(devices)
        self._replicas = [_clone_policy(policy, d) for d in self.devices]
        self._free: queue.Queue[int] = queue.Queue()
        for i in range(len(self.devices)):
            self._free.put(i)
        self._sync_lock = threading.Lock()

    @classmethod
    def from_policy(
        cls,
        policy: BranchingPolicy,
        *,
        device: str = "cpu",
        use_all_gpus: bool = True,
    ) -> "GpuInferPool":
        return cls(policy, resolve_infer_devices(device, use_all_gpus=use_all_gpus))

    def sync_from(self, policy: BranchingPolicy) -> None:
        """从训练用策略刷新全部推理副本权重。"""
        self.sync_state(_policy_state_cpu(policy))

    def sync_state(self, state: dict) -> None:
        with self._sync_lock:
            for rep in self._replicas:
                rep.load_state_dict(state)
                rep.eval()

    def infer(self, g: HeteroGraph) -> tuple[torch.Tensor, torch.Tensor]:
        """在空闲 GPU/CPU 上推理；返回 CPU 上的 ``(bool_scores, phase_logits)``。"""
        slot = self._free.get()
        try:
            rep = self._replicas[slot]
            dev = self.devices[slot]
            g_dev = g.copy_to(dev)
            with torch.inference_mode():
                out = rep.infer(g_dev)
            scores = out.bool_branch_scores.detach().cpu()
            phases = out.phase_logits.detach().cpu()
            return scores, phases
        finally:
            self._free.put(slot)


class GpuInferService:
    """主进程内 GPU 推理服务：经 Queue 接请求，线程池并发调用 :class:`GpuInferPool`。

    Z3 worker 进程只发送 CPU 图；CUDA 仅在主进程使用，与训练 ``update`` 分阶段共享设备。
    每条 infer 请求携带单向 ``Pipe`` 发送端；调度线程取请求后丢给大小为 ``#devices``
    的线程池，从而多卡可真正并行占用。
    """

    def __init__(self, pool: GpuInferPool, req_queue):
        self.pool = pool
        self.devices = pool.devices
        self._req = req_queue
        self._thread: threading.Thread | None = None
        self._workers: ThreadPoolExecutor | None = None

    @classmethod
    def from_policy(
        cls,
        policy: BranchingPolicy,
        req_queue,
        *,
        device: str = "cpu",
        use_all_gpus: bool = True,
    ) -> "GpuInferService":
        pool = GpuInferPool.from_policy(
            policy, device=device, use_all_gpus=use_all_gpus
        )
        return cls(pool, req_queue)

    def start(self) -> None:
        if self._thread is not None:
            return
        # 与 GPU 槽数一致：多请求可并行占满各卡
        self._workers = ThreadPoolExecutor(
            max_workers=max(1, len(self.devices)),
            thread_name_prefix="GpuInferSlot",
        )
        self._thread = threading.Thread(
            target=self._loop, name="GpuInferService", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 60.0) -> None:
        if self._thread is None:
            return
        self._req.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None
        if self._workers is not None:
            self._workers.shutdown(wait=True)
            self._workers = None

    def sync_from(self, policy: BranchingPolicy) -> None:
        """批次 collect 前在主线程刷新权重（此时无 worker 在飞）。"""
        self.pool.sync_from(policy)

    def _handle_infer(self, send_conn, g: HeteroGraph) -> None:
        try:
            result = self.pool.infer(g)
            send_conn.send(("ok", result))
        except Exception as exc:
            try:
                send_conn.send(
                    (
                        "err",
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                    )
                )
            except Exception:
                pass
        finally:
            try:
                send_conn.close()
            except Exception:
                pass

    def _loop(self) -> None:
        assert self._workers is not None
        while True:
            msg = self._req.get()
            if msg is None:
                break
            kind = msg[0]
            if kind != "infer":
                continue
            _, send_conn, g = msg
            self._workers.submit(self._handle_infer, send_conn, g)


class RemoteInferClient:
    """进程池 worker 侧：经 ``mp.Queue`` + ``Pipe`` 调用主进程 :class:`GpuInferService`。"""

    def __init__(self, req_queue, ctx=None):
        self._req = req_queue
        self._ctx = ctx or mp.get_context("spawn")

    def infer(self, g: HeteroGraph) -> tuple[torch.Tensor, torch.Tensor]:
        if g.node_features:
            dev = next(iter(g.node_features.values())).device
            if dev.type != "cpu":
                g = g.copy_to("cpu")
        # 注意：勿在 Queue.put 返回后立刻 close(send_conn)。
        # put 由后台 feeder 异步 pickle，过早关闭会导致 DupFd 失败。
        recv_conn, send_conn = self._ctx.Pipe(duplex=False)
        try:
            self._req.put(("infer", send_conn, g))
            tag, payload = recv_conn.recv()
        finally:
            try:
                recv_conn.close()
            except Exception:
                pass
        if tag == "ok":
            return payload
        raise RuntimeError(f"远程 GPU infer 失败:\n{payload}")


def _rl_worker_init(req_queue, ctx_name: str = "spawn") -> None:
    global _INFER_REQ, _MP_CTX
    _INFER_REQ = req_queue
    _MP_CTX = mp.get_context(ctx_name)


def _make_rl_process_pool(workers: int, req_queue, ctx) -> ProcessPoolExecutor:
    """Linux 上必须用 spawn：父进程已 init CUDA 时 fork 会导致子进程卡死。"""
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_rl_worker_init,
        initargs=(req_queue, ctx.get_start_method()),
    )


def _rl_collect_worker(task: tuple) -> tuple:
    """ProcessPool worker：独立进程跑 z3，GNN 经 RemoteInferClient 排队到主进程 GPU。"""
    (
        inst_idx,
        seed,
        hard,
        min_vars,
        max_vars,
        defer_val,
        refocus_every,
        max_iters,
        sticky_window,
        smt2_path,
        instance_id,
        ref_value,
        ref_rlimit,
    ) = task
    if _INFER_REQ is None:
        raise RuntimeError("RL worker 未初始化远程 infer（缺少 initializer）")

    if smt2_path:
        from omt_branching.solver.decide_omt import smt2_to_instance

        inst = smt2_to_instance(smt2_path, instance_id=instance_id)
    else:
        from omt_branching.solver.instance_gen import bool_lia_instance_at

        inst = bool_lia_instance_at(
            inst_idx, seed, hard=hard, min_vars=min_vars, max_vars=max_vars
        )
    hard_exprs, obj, sense = inst.as_tuple()

    # worker 不加载策略权重；infer 全部走主进程 GpuInferService
    policy = BranchingPolicy()
    policy.eval()
    defer = torch.nn.Parameter(torch.tensor(float(defer_val), dtype=torch.float32))
    client = RemoteInferClient(_INFER_REQ, _MP_CTX)

    holder: dict = {}

    def factory(assertions):
        d = SamplingPolicyDecider(
            policy,
            defer,
            assertions,
            refocus_every,
            sample=True,
            device="cpu",
            sticky_window=bool(sticky_window),
            infer_pool=client,
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
    res = dict(res)
    res["ref_value"] = ref_value
    res["ref_rlimit"] = ref_rlimit
    steps = holder["d"].steps if "d" in holder else []
    reward = decide_rl_reward(res, ref_value, ref_rlimit)
    return inst_idx, _steps_to_cpu(steps), reward, res


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
        sticky_window: bool = False,
        refocus_on_backtrack: bool = True,
        infer_pool: InferBackend | None = None,
    ):
        self.policy = policy
        self.defer_logit = defer_logit  # torch 标量（trainer 持有的可学参数）
        self.device = device
        self.infer_pool = infer_pool
        self.assertions = list(assertions)
        self.refocus_every = max(1, refocus_every)
        self.sample = sample
        # True：每窗只采样/记 step 一次，其后按缓存分数立即返回
        self.sticky_window = sticky_window
        self.refocus_on_backtrack = refocus_on_backtrack
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
        # 跨 cut 根级强制赋值（add_hard 后由 consequences 刷新）
        self._root_fixed: dict[str, bool] = {}

    def force_refocus(self) -> None:
        """清空图/分数/粘性窗，使下次 decide 立刻跑 GNN。"""
        self._graph = None
        self._scores = None
        self._phases = None
        self._bmap = {}
        self._score_by_key = {}
        self._phase_by_key = {}
        self._window_defer = False
        self._window_choice = None
        self._window_phase = True
        self._since = self.refocus_every

    def add_hard(self, *exprs) -> None:
        """并入硬约束（如 better-cut），刷新根级 forced，并强制下次 decide refocus。

        用 :func:`root_forced_assignment` 在新断言上求根级强制原子，供后续建图投影。
        """
        if not exprs:
            return
        self.assertions.extend(exprs)
        self._root_fixed = root_forced_assignment(self.assertions)
        self.force_refocus()

    def on_backtrack(self, num_scopes: int = 1) -> None:
        """propagator ``pop`` 回调：冲突回退后强制下次 decide refocus。"""
        if self.refocus_on_backtrack:
            self.force_refocus()

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
        asg = merge_root_assignment(self._root_fixed, assignment)
        snap, _ = build_bool_snapshot(self.assertions, assignment=asg)
        # 建图始终在 CPU；steps 存 CPU 图，update 时再搬到训练设备
        g = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
        if self.infer_pool is not None:
            self._scores, self._phases = self.infer_pool.infer(g)
        else:
            g_dev = g.copy_to(self.device)
            out = self.policy.infer(g_dev)
            self._scores = out.bool_branch_scores.detach().cpu()
            self._phases = out.phase_logits.detach().cpu()
        self._graph = g
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
            if (
                self._window_choice is not None
                and self._window_choice in undecided_keys
            ):
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
    #: 并行 collect 时是否把全部 CUDA 设备纳入 :class:`GpuInferPool`
    use_all_gpus: bool = True
    #: 窗口粘性；False 时每次 decide 采样并记 step（训练默认）
    sticky_window: bool = False
    #: 每轮 collect 的实例数；None / ≤0 / ≥训练集大小则整集 collect
    collect_batch_size: int | None = None


@dataclass
class EarlyStopConfig:
    """基于验证集指标的早停。

    ``maximize=True`` 时指标越大越好（如 mean_reward）；否则越小越好
    （如 mean_weighted_rlimit）。相对提升不足 ``tol`` 则计入 patience。
    """

    patience: int = 3
    tol: float = 0.02
    maximize: bool = True
    metric_key: str = "mean_reward"
    min_iters: int = 1
    max_iters: int = 10_000
    eval_every: int = 1


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
        self._infer_service: GpuInferService | None = None

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
        infer_pool: InferBackend | None = None,
    ):
        holder: dict = {}
        pool = infer_pool

        def factory(assertions):
            d = SamplingPolicyDecider(
                self.policy,
                self.defer_logit,
                assertions,
                self.config.refocus_every,
                sample=True,
                device=self.config.device,
                sticky_window=self.config.sticky_window,
                infer_pool=pool,
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
        # solve_omt_with_decider 不回传参考字段；写入 history / match 前补上。
        res = dict(res)
        res["ref_value"] = ref_value
        res["ref_rlimit"] = ref_rlimit
        steps = holder["d"].steps if "d" in holder else []
        reward = decide_rl_reward(res, ref_value, ref_rlimit)
        return steps, reward, res

    def _collect_parallel(
        self,
        indices: Sequence[int],
        *,
        seed: int,
        hard: bool,
        min_vars: int,
        max_vars: int,
        pool: ProcessPoolExecutor,
        infer_service: GpuInferService,
        pbar=None,
        iter_idx: int = 0,
        smt2_paths: list[str] | None = None,
        instance_ids: list[str] | None = None,
        ref_values: list | None = None,
        ref_rlimits: list[int | None] | None = None,
    ) -> list[tuple[int, list, float, dict]]:
        """多进程 collect：仅对 ``indices`` 中的实例；GNN 经主进程 GpuInferService。"""
        infer_service.sync_from(self.policy)
        defer_val = float(self.defer_logit.detach().cpu())
        sticky = self.config.sticky_window
        paths = smt2_paths
        ids = instance_ids
        vals = ref_values
        rls = ref_rlimits
        tasks = [
            (
                j,
                seed,
                hard,
                min_vars,
                max_vars,
                defer_val,
                self.config.refocus_every,
                self.config.max_iters,
                sticky,
                None if paths is None else paths[j],
                None if ids is None else ids[j],
                None if vals is None else vals[j],
                None if rls is None else rls[j],
            )
            for j in indices
        ]
        results: list[tuple[int, list, float, dict]] = []
        futures = [pool.submit(_rl_collect_worker, t) for t in tasks]
        done = 0
        total = len(indices)
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    iter=iter_idx, phase="collect", inst=f"{done}/{total}"
                )
        results.sort(key=lambda x: x[0])
        return results

    @staticmethod
    def _iter_indices(
        count: int,
        batch_size: int | None,
        *,
        seed: int,
        iter_idx: int,
    ) -> list[int]:
        """本轮要 collect 的实例下标；``batch_size`` 空/过大则整集。"""
        if batch_size is None or batch_size <= 0 or batch_size >= count:
            return list(range(count))
        rng = random.Random(seed + iter_idx * 100_003)
        return rng.sample(range(count), batch_size)

    def save_checkpoint(self, path, *, meta: dict | None = None) -> None:
        """保存策略权重 + ``defer_logit``（写入 ``meta``）。"""
        from omt_branching.model.persistence import save_policy

        payload_meta = {
            "defer_logit": float(self.defer_logit.detach().cpu()),
            **(meta or {}),
        }
        save_policy(self.policy, path, meta=payload_meta)

    def collect_sat(self, assertions, atoms, ref_conflicts: int | None = None):
        """采一次学习臂;reward = 归一化 ``[-1,0]`` 的 conflicts(见 :func:`sat_conflict_reward`)。

        ``ref_conflicts`` 为同实例 VSIDS 臂的 conflicts(归一化参考);缺省时内部跑一次 VSIDS
        求出(单实例调试方便,但训练应由 :meth:`train_sat` 预算好复用,避免每轮重算)。
        """
        from omt_branching.solver.sat_solve import solve_sat_with_decider

        if ref_conflicts is None:
            ref_conflicts = solve_sat_with_decider(
                list(assertions), list(atoms), None
            )["conflicts"]

        holder: dict = {}

        def factory(asserts):
            d = SamplingPolicyDecider(
                self.policy,
                self.defer_logit,
                asserts,
                self.config.refocus_every,
                sample=True,
                device=self.config.device,
                sticky_window=self.config.sticky_window,
            )
            holder["d"] = d
            return d

        res = solve_sat_with_decider(
            list(assertions), list(atoms), decider_factory=lambda a: factory(a)
        )
        steps = holder["d"].steps if "d" in holder else []
        reward = sat_conflict_reward(res["conflicts"], ref_conflicts, self.config.reward_cap)
        return steps, reward, res

    def train_sat(self, problems, iterations: int = 1, log: bool = False):
        """``problems = list[(atoms, clauses)]``（同生成器返回序）。

        每实例**一次性**算出 VSIDS 参考 conflicts(跨 iteration 复用),供 reward 归一化。
        """
        from omt_branching.solver.sat_solve import solve_sat_with_decider

        problems = list(problems)
        refs = [
            solve_sat_with_decider(list(clauses), list(atoms), None)["conflicts"]
            for atoms, clauses in problems
        ]
        history = []
        for it in range(iterations):
            for j, (atoms, assertions) in enumerate(problems):
                steps, reward, res = self.collect_sat(
                    assertions, atoms, ref_conflicts=refs[j]
                )
                stats = self.update(steps, reward, key=j)
                stats.update({
                    "iter": it, "instance": j, "conflicts": res["conflicts"],
                    "ref_conflicts": refs[j], "reward": reward,
                })
                history.append(stats)
                if log:
                    print(
                        f"[it {it} inst {j}] loss={stats['loss']:.4f} reward={reward:.3f} "
                        f"conflicts={res['conflicts']} ref={refs[j]} steps={stats['steps']}"
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
        collect_batch_size: int | None = None,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 1,
        eval_callback: Callable[[int, "DecideRLTrainer"], dict] | None = None,
        early_stop: EarlyStopConfig | None = None,
    ):
        """REINFORCE 训练。``workers>1`` 且本轮实例数足够时用 spawn 进程池并行 collect。

        并行时主进程启动 :class:`GpuInferService`，经 Queue 排队占用全部 GPU 做
        GNN 推理；Z3 在子进程中运行，与其它实例的 infer 重叠。``smt2_paths`` 非空
        时 worker 从落盘文件读实例，否则按 seed 重建。``ref_values`` /
        ``ref_rlimits`` 为各实例的 ``ref/`` 缓存参考。``checkpoint_dir`` 非空时每隔
        ``checkpoint_every`` 轮保存中间权重。

        ``collect_batch_size``（或 ``config.collect_batch_size``）限制每轮 collect
        的实例数；未设则整集。``sticky_window`` 见 :class:`DecideRLConfig`。

        ``iterations=-1`` 时训练直到 ``early_stop`` 判定收敛（须提供
        ``eval_callback`` + ``early_stop``）；``iterations>0`` 时最多跑该轮数，
        若中途收敛则提前结束。
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
        if iterations == 0:
            return []
        if iterations < -1:
            raise ValueError("iterations 须为 -1（直到收敛）或正整数")
        if iterations == -1:
            if early_stop is None or eval_callback is None:
                raise ValueError(
                    "iterations=-1 需要同时提供 early_stop 与 eval_callback"
                )
            max_rounds = early_stop.max_iters
        else:
            max_rounds = iterations
            if early_stop is not None and eval_callback is None:
                raise ValueError("启用 early_stop 时必须提供 eval_callback")

        vals = ref_values if ref_values is not None else [None] * count
        rls = ref_rlimits if ref_rlimits is not None else [None] * count
        batch_cfg = (
            collect_batch_size
            if collect_batch_size is not None
            else self.config.collect_batch_size
        )
        # 用「每轮最多 collect 数」决定是否值得起进程池
        per_iter_cap = count
        if batch_cfg is not None and batch_cfg > 0:
            per_iter_cap = min(batch_cfg, count)
        requested = workers if workers is not None else self.config.workers
        n_workers = effective_rl_workers(
            per_iter_cap,
            requested,
            min_instances=self.config.min_instances_for_parallel,
        )
        use_parallel = n_workers > 1
        history = []
        eval_history: list[dict] = []
        proc_pool: ProcessPoolExecutor | None = None
        infer_service: GpuInferService | None = None
        if use_parallel:
            ctx = mp.get_context("spawn")
            req_queue = ctx.Queue()
            infer_service = GpuInferService.from_policy(
                self.policy,
                req_queue,
                device=self.config.device,
                use_all_gpus=self.config.use_all_gpus,
            )
            infer_service.start()
            self._infer_service = infer_service
            proc_pool = _make_rl_process_pool(n_workers, req_queue, ctx)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        best_metric: float | None = None
        best_state: dict | None = None
        best_defer: float | None = None
        stall = 0
        stop_reason: str | None = None
        finished_iters = 0

        def _improved(curr: float, best: float) -> bool:
            assert early_stop is not None
            scale = abs(best) + 1e-8
            if early_stop.maximize:
                return (curr - best) / scale > early_stop.tol
            return (best - curr) / scale > early_stop.tol

        try:
            pbar_total = max_rounds * per_iter_cap if iterations != -1 else None
            with tqdm(total=pbar_total, desc="rl_train") as pbar:
                for it in range(max_rounds):
                    indices = self._iter_indices(
                        count, batch_cfg, seed=collect_seed, iter_idx=it
                    )
                    if use_parallel:
                        assert proc_pool is not None and infer_service is not None
                        batch = self._collect_parallel(
                            indices,
                            seed=collect_seed,
                            hard=collect_hard,
                            min_vars=collect_min_vars,
                            max_vars=collect_max_vars,
                            pool=proc_pool,
                            infer_service=infer_service,
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
                                inst=f"{upd_i + 1}/{len(indices)}",
                            )
                    else:
                        for j in indices:
                            hard, obj, sense = instances[j]
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
                    finished_iters = it + 1
                    if (
                        checkpoint_dir
                        and checkpoint_every > 0
                        and finished_iters % checkpoint_every == 0
                    ):
                        ckpt = os.path.join(
                            checkpoint_dir, f"iter_{finished_iters:04d}.pt"
                        )
                        self.save_checkpoint(
                            ckpt,
                            meta={
                                "iter": finished_iters,
                                "iterations": iterations,
                            },
                        )

                    if (
                        early_stop is not None
                        and eval_callback is not None
                        and finished_iters % max(1, early_stop.eval_every) == 0
                    ):
                        metrics = dict(eval_callback(finished_iters, self))
                        metrics["iter"] = finished_iters
                        eval_history.append(metrics)
                        key = early_stop.metric_key
                        if key not in metrics:
                            raise KeyError(
                                f"eval_callback 未返回 early_stop.metric_key={key!r}"
                            )
                        curr = float(metrics[key])
                        if best_metric is None or _improved(curr, best_metric):
                            best_metric = curr
                            best_state = {
                                k: v.detach().cpu().clone()
                                for k, v in self.policy.state_dict().items()
                            }
                            best_defer = float(self.defer_logit.detach().cpu())
                            stall = 0
                            if checkpoint_dir:
                                self.save_checkpoint(
                                    os.path.join(checkpoint_dir, "best_eval.pt"),
                                    meta={
                                        "iter": finished_iters,
                                        "best_metric": best_metric,
                                        "metric_key": key,
                                        "best": True,
                                    },
                                )
                        else:
                            stall += 1
                        tag = (
                            f"eval[{key}]={curr:.4f} best={best_metric:.4f} "
                            f"stall={stall}/{early_stop.patience}"
                        )
                        pbar.set_postfix_str(tag)
                        if log:
                            print(f"[it {finished_iters}] {tag} | {metrics}")
                        if (
                            finished_iters >= early_stop.min_iters
                            and stall >= early_stop.patience
                        ):
                            stop_reason = "converged"
                            break
                else:
                    if iterations == -1:
                        stop_reason = "max_iters"
                    else:
                        stop_reason = "completed"
        finally:
            if proc_pool is not None:
                proc_pool.shutdown(wait=True)
            if infer_service is not None:
                infer_service.stop()
            self._infer_service = None

        # 启用早停时回滚到验证集最优权重（收敛 / 跑满上限均如此）
        if best_state is not None and early_stop is not None:
            self.policy.load_state_dict(best_state)
            with torch.no_grad():
                self.defer_logit.copy_(
                    torch.tensor(best_defer, device=self.defer_logit.device)
                )

        # 在 history 末尾附一条汇总（便于日志 / JSON）
        history.append(
            {
                "event": "train_end",
                "stop_reason": stop_reason or "completed",
                "finished_iters": finished_iters,
                "best_metric": best_metric,
                "metric_key": (
                    early_stop.metric_key if early_stop is not None else None
                ),
                "eval_history": eval_history,
                "defer_logit": float(self.defer_logit.detach().cpu()),
                "collect_batch_size": batch_cfg,
                "sticky_window": self.config.sticky_window,
            }
        )
        return history


__all__ = [
    "DEFAULT_RL_COLLECT_WORKERS",
    "MIN_INSTANCES_FOR_RL_PARALLEL",
    "effective_rl_workers",
    "decide_rl_reward",
    "GpuInferPool",
    "GpuInferService",
    "RemoteInferClient",
    "SamplingPolicyDecider",
    "DecideRLConfig",
    "DecideRLTrainer",
    "EarlyStopConfig",
]
