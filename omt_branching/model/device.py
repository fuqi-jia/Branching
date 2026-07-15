"""GNN 推理/训练默认设备：有 CUDA 则用 GPU，否则 CPU。"""

from __future__ import annotations

import torch


def gnn_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["gnn_device"]
