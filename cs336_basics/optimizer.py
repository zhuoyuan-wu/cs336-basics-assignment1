"""cs336_basics/optimizer.py

从零手写工业级 AdamW 优化器，严格继承自 torch.optim.Optimizer。
"""

from collections.abc import Callable
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        """初始化 AdamW 优化器。

        参数:
            params: 待优化的参数可迭代对象或参数组字典列表
            lr: 学习率 alpha
            betas: 一阶矩与二阶矩的平滑系数 (beta1, beta2)
            eps: 为防止除零而加在分母上的小量 epsilon
            weight_decay: 解耦权重衰减系数 lambda
        """
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta1 参数: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta2 参数: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"无效的 epsilon 参数: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay 参数: {weight_decay}")

        # 将默认超参数打包传给父类 Base Optimizer
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        """执行单步参数更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有的参数组 (param_groups)
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # 遍历该组中的每一个参数张量 p
            for p in group["params"]:
                # 如果该参数没有梯度（比如冻结层），跳过
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 1. 状态初始化 (第一次更新该参数时)
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶矩 m 初始化为与参数同形状的全 0 张量
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # 二阶矩 v 初始化为与参数同形状的全 0 张量
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # 步数 t 从 1 开始递增
                state["step"] += 1
                t = state["step"]

                # 2. 更新一阶矩: m = beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 3. 更新二阶矩: v = beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 4. 计算偏置校正调整后的学习率: alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                step_size = (
                    lr * math.sqrt(bias_correction2) / bias_correction1
                )

                # 5. 自适应梯度更新: p = p - alpha_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                # 6. 解耦权重衰减 (Decoupled Weight Decay): p = p - alpha * weight_decay * p
                if weight_decay > 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss