import torch
import torch.nn as nn
import math
from typing import Optional

from cs336_basics.model import RMSNorm, softmax



# 1. 纯张量手写 RoPE (Rotary Position Embedding)

def precompute_rope_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """预计算 RoPE 的旋转角频率复数表 (或正余弦张量)"""
    # dim 必须为偶数 (每两两维度一组旋转)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式 e^{i * m * theta}
    return freqs_cis


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 旋转应用到 Query 和 Key 张量上。
    xq: (batch_size, seq_len, num_heads, head_dim)
    xk: (batch_size, seq_len, num_heads, head_dim)
    freqs_cis: (seq_len, head_dim // 2) 复数
    """
    # 将最后一维视为复数 (实部与虚部两两配对)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整 freqs_cis 维度用于广播: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, -1).to(xq.device)
    
    # 复数相乘完成二维空间旋转！
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)



# 2. SwiGLU 门控前馈网络

class SwiGLUFeedForward(nn.Module):
    """
    现代 LLaMA 标配的 SwiGLU 前馈网络
    FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # 在 LLaMA 中，d_ff 通常为 int(2/3 * 4 * d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 手写 SiLU: z * sigmoid(z)，纯顶级张量运算，完全不碰违规库
        gate = self.w_gate(x)
        silu_gate = gate * torch.sigmoid(gate)
        return self.w_down(silu_gate * self.w_up(x))



# 3. 带 RoPE 的 LLaMA 自注意力

class LlamaAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: Optional[float] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(attn_pdrop if attn_pdrop is not None else 0.0)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # 核心：注入 RoPE 旋转位置编码！
        q, k = apply_rotary_emb(q, k, freqs_cis[:T])

        # 转置为 (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果遮罩
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        
        attn_probs = softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output_proj(out)



# 4. 完整的 LLaMA Block & 整机模型

class LlamaBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop=None, residual_pdrop=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = LlamaAttention(d_model, num_heads, attn_pdrop)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff)
        self.residual_dropout = nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attn(self.ln1(x), freqs_cis))
        x = x + self.residual_dropout(self.ffn(self.ln2(x)))
        return x


class LlamaLM(nn.Module):
    """
    符合现代业界最高标准的 LLaMA 架构整机模型！
    - RoPE 旋转位置编码 (无需绝对位置矩阵)
    - SwiGLU 激活前馈层
    - RMSNorm 预归一化
    - 绑定输出嵌入头
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: Optional[float] = None,
        residual_pdrop: Optional[float] = None,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(residual_pdrop if residual_pdrop is not None else 0.0)
        
        # 预计算最大 context_length 的 RoPE 复数频率表
        head_dim = d_model // num_heads
        self.register_buffer("freqs_cis", precompute_rope_cis(head_dim, context_length))

        self.layers = nn.ModuleList([
            LlamaBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        # 无需绝对位置编码相加，直接嵌入并进 dropout
        h = self.dropout(self.token_embeddings(in_indices))
        freqs_cis = self.freqs_cis[:in_indices.shape[1]]
        
        for layer in self.layers:
            h = layer(h, freqs_cis)
            
        h = self.ln_final(h)
        return self.lm_head(h)