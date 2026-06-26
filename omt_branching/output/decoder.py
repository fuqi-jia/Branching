"""把 :class:`PolicyOutput` 解码成 :class:`BranchingAdvice`。

职责:

- 用图的反向 id 映射，把图内局部索引还原成求解器原始 id。
- 在候选集合上做 masked softmax，得到可混入 VSIDS 的 activity 先验。
- 生成候选排序、phase 建议、整数 split 建议。
- 读取 ``graph.meta['inference']`` 的诊断，决定 ``use_gnn``（回退）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from omt_branching.interfaces import NodeType
from omt_branching.model.policy import PolicyOutput
from omt_branching.output.advice import BranchingAdvice, IntegerSplitAdvice


@dataclass
class DecoderConfig:
    top_k: int = 0                 # 排序候选裁剪到前 k；0 = 全部
    phase_threshold: float = 0.5   # sigmoid(phase_logit) >= 阈值 -> 取真
    include_aux: bool = True       # 是否回传辅助预测


class AdviceDecoder:
    def __init__(self, config: DecoderConfig = DecoderConfig()):
        self.config = config

    def decode(self, out: PolicyOutput) -> BranchingAdvice:
        g = out.graph
        if g is None:
            raise ValueError("PolicyOutput.graph 为空，无法解码回求解器 id")

        advice = BranchingAdvice()

        diag = g.meta.get("inference", {}) if g.meta else {}
        advice.diagnostics = dict(diag)
        advice.use_gnn = not bool(diag.get("fallback", False))
        advice.confidence = float(diag.get("confidence", 0.0))

        # ---- 布尔 activity 先验 + 候选排序 ----
        bool_probs = out.masked_bool_probs()
        cand_bool = out.candidate_bool_local
        if bool_probs.numel() > 0 and cand_bool:
            phase_prob = torch.sigmoid(out.phase_logits)
            scored: list[tuple[float, object]] = []
            for local in cand_bool:
                sid = g.solver_id(NodeType.BOOL_VAR, local)
                if sid is None:
                    continue
                p = float(bool_probs[local])
                advice.activity_priors[sid] = p
                scored.append((p, sid))
                advice.phase_suggestions[sid] = (
                    float(phase_prob[local]) >= self.config.phase_threshold
                )
            scored.sort(key=lambda t: t[0], reverse=True)
            ranked = [sid for _, sid in scored]
            if self.config.top_k > 0:
                ranked = ranked[: self.config.top_k]
            advice.ranked_candidates = ranked
            if advice.confidence == 0.0 and scored:
                advice.confidence = scored[0][0]

        # ---- 整数 B&B split ----
        num_probs = out.masked_numeric_probs()
        cand_num = out.candidate_numeric_local
        if num_probs.numel() > 0 and cand_num:
            dir_prob = torch.sigmoid(out.int_dir_logits)
            splits: list[IntegerSplitAdvice] = []
            for local in cand_num:
                sid = g.solver_id(NodeType.NUMERIC_VAR, local)
                if sid is None:
                    continue
                up = float(dir_prob[local]) >= 0.5
                splits.append(
                    IntegerSplitAdvice(
                        num_var_id=sid,
                        branch_up=up,
                        score=float(num_probs[local]),
                        direction_confidence=abs(float(dir_prob[local]) - 0.5) * 2.0,
                    )
                )
            splits.sort(key=lambda s: s.score, reverse=True)
            if self.config.top_k > 0:
                splits = splits[: self.config.top_k]
            advice.ranked_integer_candidates = splits
            advice.integer_split = splits[0] if splits else None

        # ---- 辅助预测 ----
        if self.config.include_aux and out.aux:
            self._fill_aux(advice, out, g, cand_bool)

        return advice

    def _fill_aux(self, advice, out, g, cand_bool):
        aux = out.aux
        for local in cand_bool:
            sid = g.solver_id(NodeType.BOOL_VAR, local)
            if sid is None:
                continue
            rec: dict[str, float] = {}
            if "conflict_logit" in aux:
                rec["conflict_prob"] = _sig(aux["conflict_logit"], local)
            if "in_core_logit" in aux:
                rec["in_core_prob"] = _sig(aux["in_core_logit"], local)
            if "obj_improve" in aux:
                rec["obj_improve"] = float(aux["obj_improve"][local])
            if "subtree_size" in aux:
                rec["subtree_size"] = float(aux["subtree_size"][local])
            if rec:
                advice.aux_predictions[sid] = rec


def _sig(t: torch.Tensor, i: int) -> float:
    return 1.0 / (1.0 + math.exp(-float(t[i])))
