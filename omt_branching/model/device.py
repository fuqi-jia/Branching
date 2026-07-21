"""GNN 推理/训练默认设备：有 CUDA 则用 GPU，否则 CPU。"""

from __future__ import annotations

import torch


def gnn_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_infer_devices(device: str = "cpu", *, use_all_gpus: bool = True) -> list[str]:
    """collect 推理设备列表。

    - 有 CUDA 且 ``use_all_gpus``：返回全部 ``cuda:0..N-1``（排队占用所有卡）；
    - 否则归一化 ``device``（``cuda`` → ``cuda:0``）；无 CUDA 时回退 ``cpu``。
    """
    if torch.cuda.is_available() and use_all_gpus:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    d = str(device)
    if d == "cuda":
        d = "cuda:0"
    if d.startswith("cuda") and not torch.cuda.is_available():
        return ["cpu"]
    return [d]


__all__ = ["gnn_device", "resolve_infer_devices"]
