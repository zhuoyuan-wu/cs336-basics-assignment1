import torch
import torch.nn as nn
from typing import Optional

from cs336_basics.model import (
    RMSNorm,
    CausalMultiHeadSelfAttention,
    PositionwiseFeedForward,
)


class ParallelTransformerBlock(nn.Module):
    """
    §7.3 并行层架构 (PaLM 风格，公式 15)
    y = x + Attn(RMSNorm(x)) + FFN(RMSNorm(x))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        # 并行层只使用一个共享的 RMSNorm
        self.ln = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.residual_dropout = nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 归一化输入
        norm_x = self.ln(x)
        # 2. 并行计算 Attention 与 FFN
        attn_out = self.residual_dropout(self.attn(norm_x))
        ffn_out = self.residual_dropout(self.ffn(norm_x))
        # 3. 双路残差聚合
        return x + attn_out + ffn_out


class NoNormTransformerBlock(nn.Module):
    """
    §7.3 拔除所有 RMSNorm 的消融积木块
    y = z + FFN(z), 其中 z = x + Attn(x)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.residual_dropout = nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 完全没有 RMSNorm
        x = x + self.residual_dropout(self.attn(x))
        x = x + self.residual_dropout(self.ffn(x))
        return x


class PostNormTransformerBlock(nn.Module):
    """
    §7.3 传统 Post-Norm 架构 (Attention is All You Need 原始方案)
    z = RMSNorm(x + Attn(x))
    y = RMSNorm(z + FFN(z))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.residual_dropout = nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：先残差后 Norm
        x = self.ln1(x + self.residual_dropout(self.attn(x)))
        x = self.ln2(x + self.residual_dropout(self.ffn(x)))
        return x