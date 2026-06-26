"""集成门面：把输入 / 模型 / 输出三部分串成一个调用入口。

求解器只需持有一个 :class:`BranchingPolicyService`，在 decision/refocus 点调用
``advise(snapshot) -> BranchingAdvice``，无需关心建图与张量细节。
"""

from __future__ import annotations

from dataclasses import dataclass

from omt_branching.input.graph_builder import DEFAULT_FEATURE_SPEC, FeatureSpec, GraphBuilder
from omt_branching.input.solver_state import SolverSnapshot
from omt_branching.model.inference import InferenceConfig, InferenceEngine
from omt_branching.model.policy import BranchingPolicy, PolicyConfig
from omt_branching.output.advice import BranchingAdvice
from omt_branching.output.decoder import AdviceDecoder, DecoderConfig


@dataclass
class ServiceConfig:
    policy: PolicyConfig = PolicyConfig()
    inference: InferenceConfig = InferenceConfig()
    decoder: DecoderConfig = DecoderConfig()


class BranchingPolicyService:
    """输入 -> 模型推理 -> 输出 的端到端封装。"""

    def __init__(self, policy: BranchingPolicy | None = None,
                 feature_spec: FeatureSpec = DEFAULT_FEATURE_SPEC,
                 config: ServiceConfig = ServiceConfig()):
        self.config = config
        self.builder = GraphBuilder(feature_spec)
        self.policy = policy or BranchingPolicy(feature_spec, config.policy)
        self.engine = InferenceEngine(self.policy, config.inference)
        self.decoder = AdviceDecoder(config.decoder)

    def advise(self, snapshot: SolverSnapshot) -> BranchingAdvice:
        """主入口：求解器快照 -> 分支建议。"""
        graph = self.builder.build(snapshot)
        out = self.engine.run(graph)
        if out is None:
            # 规模超限：返回一个仅含回退标记的空建议
            advice = BranchingAdvice(use_gnn=False)
            advice.diagnostics = dict(graph.meta.get("inference", {}))
            return advice
        return self.decoder.decode(out)
