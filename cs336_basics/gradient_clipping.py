import torch
from typing import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """计算全局梯度的 L2 范数并执行原地裁剪。"""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return

    # 全局范数 = 所有参数梯度的二范数平方和再开方
    total_norm_sq = sum(torch.norm(p.grad.detach(), p=2).item() ** 2 for p in params_with_grad)
    total_norm = total_norm_sq ** 0.5

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coef)