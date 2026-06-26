"""输出部分：把模型输出解码成求解器可用的分支建议。

- ``advice``  : 定义返回给求解器的信息格式（接口契约）。
- ``decoder`` : 把 ``PolicyOutput`` 解码成 ``BranchingAdvice``。
"""

from omt_branching.output.advice import BranchingAdvice, IntegerSplitAdvice
from omt_branching.output.decoder import AdviceDecoder, DecoderConfig

__all__ = [
    "BranchingAdvice",
    "IntegerSplitAdvice",
    "AdviceDecoder",
    "DecoderConfig",
]
