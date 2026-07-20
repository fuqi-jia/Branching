"""VSIDS 轨迹观察 + 模仿样本：把 z3 原生 VSIDS 的**逐步决策**作 imitation 冷启动标签。

动机：旧 look-ahead 教师(``lookahead.py``)只在**根状态**打一次静态 ``consequences`` 分数,
教出的是准静态全序 —— 在 SMT(LIA) 上输给 VSIDS 的冲突自适应动态。VSIDS 轨迹标签在**真实
搜索中间状态**采集(部分赋值经 ``build_bool_snapshot(assignment=...)`` 编码,与 RL 回路同款),
故能把 VSIDS 的动态行为作为 BC 目标 —— 但只作 **warm-start**:纯模仿至多追平 VSIDS,靠其后
的 REINFORCE(reward=归一化 −conflicts)才可能反超。

机制:z3 的 ``add_decide`` 回调既是 override 钩子也是**观察点** —— 参数 ``t`` 正是 z3 自身
(VSIDS)将要分裂的文字。观察臂在每次 decide 记录 ``(当前赋值, atom_key(t), 相位)`` 后直接
返回(不 ``next_split``),让 VSIDS 照常执行 ``t``,即免费录得 VSIDS 的完整决策轨迹,不改 z3。

已知边界:仅记录 VSIDS 落在**已注册原子**上的决策(未注册的辅助/Tseitin 变量跳过)——
部署 decider 也只在注册原子间选,故这是可获得的最贴近的克隆目标;VSIDS 选辅助变量的状态无
标签(全覆盖 override 与 VSIDS 的固有差,留给 RL 弥合)。
"""
from __future__ import annotations

from dataclasses import dataclass

import z3

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, GraphBuilder
from omt_branching.interfaces import NodeType
from omt_branching.model.trainer import RankingExample
from omt_branching.solver.propagator_snapshot import atom_key, build_bool_snapshot


@dataclass(frozen=True)
class VSIDSTraceConfig:
    """采集控制。``stride``>1 每 stride 次注册-原子决策留 1 条;``max_examples``>0 为单实例
    上限(0=不限);``weight`` 是 one-hot 目标锐度 —— ListNet 损失会对目标 softmax,故用较大值
    使 softmax 近似 one-hot(见 ``build_vsids_examples_sat``)。
    """

    stride: int = 1
    max_examples: int = 0
    weight: float = 10.0


def _stat(s, key):
    st = s.statistics()
    for k in st.keys():
        if k == key:
            return st.get_key_value(k)
    return 0


class _VSIDSTraceProp(z3.UserPropagateBase):
    """观察-only propagator:跟踪赋值(add_fixed),在每次 decide 记录 VSIDS 的选择后放行。

    结构与 :class:`LearnedDecidePropagator` 同(push/pop/fresh + get_id 表),唯一区别是
    ``_on_decide`` 不 ``next_split`` —— 即不接管、只观察。
    """

    def __init__(self, s, atoms, sink: list, config: VSIDSTraceConfig):
        super().__init__(s)
        self.atoms = list(atoms)
        self.key2atom = {atom_key(a): a for a in self.atoms}
        self._id2key = {a.get_id(): k for k, a in self.key2atom.items()}
        self.sink = sink                 # 追加 (assignment_copy, chosen_key, phase_bool)
        self.config = config
        self._val: dict = {}
        self._trail: list = []
        self._lim: list = []
        self.n_seen = 0                  # 落在注册原子上的 VSIDS 决策数
        self.n_records = 0
        self.add_fixed(self._on_fixed)
        self.add_decide(self._on_decide)
        for a in self.atoms:
            self.add(a)

    def push(self):
        self._lim.append(len(self._trail))

    def pop(self, num_scopes):
        for _ in range(num_scopes):
            lim = self._lim.pop()
            while len(self._trail) > lim:
                self._val.pop(self._trail.pop(), None)

    def fresh(self, new_ctx):
        return _VSIDSTraceProp(new_ctx, self.atoms, self.sink, self.config)

    def _on_fixed(self, t, v):
        k = self._id2key.get(t.get_id())
        if k is not None and k not in self._val:
            self._val[k] = z3.is_true(v)
            self._trail.append(k)

    def _on_decide(self, t, idx, phase):
        # t = VSIDS 将要分裂的文字。只观察落在已注册原子上的决策。
        k = self._id2key.get(t.get_id())
        if k is None or k in self._val:
            return                        # 未注册(辅助变量)或已定 -> 跳过
        self.n_seen += 1
        if self.config.stride > 1 and (self.n_seen % self.config.stride) != 0:
            return
        if self.config.max_examples and self.n_records >= self.config.max_examples:
            return
        ph = int(phase) == int(z3.Z3_L_TRUE)
        self.sink.append((dict(self._val), k, ph))   # 复制赋值:z3 会 push/pop 改动 _val
        self.n_records += 1
        # 不 next_split -> 放行 VSIDS 执行它自己的 t


def collect_vsids_trajectory(assertions, atoms, config: VSIDSTraceConfig = VSIDSTraceConfig()):
    """单实例观察 VSIDS 一遍。返回 ``(records, ref_conflicts, info)``:

    - ``records``: ``list[(assignment_copy, chosen_key, phase_bool)]``,搜索中间状态标签;
    - ``ref_conflicts``: 本次(纯 VSIDS,观察-only 未覆盖)的 conflicts,作 RL 归一化参考;
    - ``info``: 结果摘要。
    """
    s = z3.Solver()
    sink: list = []
    prop = _VSIDSTraceProp(s, list(atoms), sink, config)
    s.add(*assertions)
    res = s.check()
    ref_conflicts = _stat(s, "conflicts")
    info = {
        "result": "sat" if res == z3.sat else ("unsat" if res == z3.unsat else "unknown"),
        "conflicts": ref_conflicts,
        "decisions_registered": prop.n_seen,
        "records": len(sink),
        "rlimit": _stat(s, "rlimit count"),
    }
    return sink, ref_conflicts, info


def build_vsids_examples_sat(problems, config: VSIDSTraceConfig = VSIDSTraceConfig()):
    """VSIDS 模仿样本(与 ``build_lookahead_examples_sat`` 同签名,可直接替换教师)。

    ``problems = list[(atoms, clauses)]``。对每个 VSIDS 决策状态建图并打**近似 one-hot**标签:
    ``bool_target_scores`` 里 VSIDS 所选原子记 ``weight``、其余未定原子记 ``0`` —— 经 ListNet
    的 ``softmax(target)`` 后近似 one-hot,损失即"在全部未定原子上预测 VSIDS 之选"的交叉熵。
    """
    out: list[RankingExample] = []
    for atoms, assertions in problems:
        records, _ref, _info = collect_vsids_trajectory(list(assertions), list(atoms), config)
        for assignment, chosen_key, phase in records:
            snap, _ = build_bool_snapshot(list(assertions), assignment=assignment)
            graph = GraphBuilder(DEFAULT_FEATURE_SPEC).build(snap)
            bmap = graph.id_maps.get(NodeType.BOOL_VAR, {})
            if chosen_key not in bmap:
                continue
            undecided = [k for k in bmap if k not in assignment]
            bts = {bmap[k]: (config.weight if k == chosen_key else 0.0) for k in undecided}
            bts[bmap[chosen_key]] = config.weight   # 保证被选原子入表(即便 undecided 计算漏掉)
            pts = {bmap[chosen_key]: phase}
            out.append(RankingExample(graph=graph, bool_target_scores=bts, phase_targets=pts))
    return out


__all__ = ["VSIDSTraceConfig", "collect_vsids_trajectory", "build_vsids_examples_sat"]
