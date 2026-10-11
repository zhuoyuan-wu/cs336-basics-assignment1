import torch
from typing import Any

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Any,
) -> None:
    """将模型权重、优化器状态和步数序列化到文件或路径。"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: Any,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从 Checkpoint 中恢复模型和优化器状态，并返回记录的步数。"""
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]