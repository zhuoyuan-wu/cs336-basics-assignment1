"""cs336_basics/model.py

从零手写 Transformer 语言模型的各个核心架构组件。
"""

import torch
import torch.nn as nn

import math


class RMSNorm(nn.Module):
    """均方根层归一化 (Root Mean Square Layer Normalization)"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 可学习的缩放权重 g_i (gain)，形状为 (d_model,)，初始化全为 1
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms_inv = torch.rsqrt(mean_square + self.eps)
        return x * rms_inv * self.weight

def gelu(x: torch.Tensor) -> torch.Tensor:

    #公式: GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) ->torch.Tensor:

        hidden = gelu(self.w1(x))

        return self.w2(hidden)


def softmax(x: torch.Tensor, dim: int = -1) ->torch.Tensor:

    #Softmax函数

    # 1. 找到指定维度上的最大值 max(x)
    # keepdim=True 保持维度，以便利用广播机制直接做减法
    max_val = torch.max(x, dim=dim, keepdim=True)[0]

    #2.减去最大值，繁殖指数爆炸变成inf
    x_shifted = x - max_val

    #3.计算指数
    exp_x = torch.exp(x_shifted)

    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    attn_pdrop: float | None = None,
) -> torch.Tensor:
    """缩放点积注意力 (Scaled Dot-Product Attention)。

    参数:
        q: 查询张量 (*, seq_len_q, d_k)
        k: 键张量 (*, seq_len_k, d_k)
        v: 值张量 (*, seq_len_k, d_v)
        mask: 可选的布尔掩码张量，True 表示需要遮蔽（概率为 0）
        attn_pdrop: 可选的注意力 dropout 丢弃率
    输出:
        融合上下文后的张量 (*, seq_len_q, d_v)
    """
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    attn_weights = softmax(scores, dim=-1)

    if attn_pdrop is not None and attn_pdrop > 0.0:
        attn_weights = nn.functional.dropout(attn_weights, p=attn_pdrop)
    
    return torch.matmul(attn_weights, v)


class CausalMultiHeadSelfAttention(nn.Module):
    """因果多头自注意力模块"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_pdrop: float | None = None,
    ):
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除！"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attn_pdrop = attn_pdrop

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2
        )
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2
        )
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(
            1, 2
        )

        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )

        attn_out = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            attn_pdrop=self.attn_pdrop if self.training else None,
        )

        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        return self.output_proj(attn_out)

class TransformerBlock(nn.Module):
    """预归一化 (Pre-Norm) Transformer 模块。

    包含因果多头自注意力子层和前馈网络子层，带有残差连接与 Dropout。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
    ):
        super().__init__()
        # 1. 两个子层各自的前置归一化层
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

        # 2. 核心子层：因果多头自注意力与前馈网络
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_pdrop=attn_pdrop,
        )
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff)

        # 3. 残差连接处使用的 Dropout
        self.residual_dropout = nn.Dropout(
            p=residual_pdrop if residual_pdrop is not None else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 x: (Batch, Seq_len, d_model)

        输出: (Batch, Seq_len, d_model)
        """
        # 子层 1: x + Dropout(Attention(RMSNorm(x)))
        norm_x = self.ln1(x)
        attn_out = self.attn(norm_x)
        x = x + self.residual_dropout(attn_out)

        # 子层 2: x + Dropout(FFN(RMSNorm(x)))
        norm_x = self.ln2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.residual_dropout(ffn_out)

        return x


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float | None = None,
        residual_pdrop: float | None = None,
        block_type="baseline",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        # 1. 词嵌入与位置嵌入
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.embedding_dropout = nn.Dropout(
            p=residual_pdrop if residual_pdrop is not None else 0.0
        )

        # 2. 堆叠 Transformer 块
        # 1. 根据消融类型挑选对应的 Block 类
        if block_type == "parallel":
            from cs336_basics.ablation_blocks import ParallelTransformerBlock
            block_cls = ParallelTransformerBlock
        elif block_type == "no_norm":
            from cs336_basics.ablation_blocks import NoNormTransformerBlock
            block_cls = NoNormTransformerBlock
        elif block_type == "post_norm":
            from cs336_basics.ablation_blocks import PostNormTransformerBlock
            block_cls = PostNormTransformerBlock
        else:
            block_cls = TransformerBlock

        # 2. 循环创建对应的层
        self.layers = nn.ModuleList(
            [
                block_cls(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    residual_pdrop=residual_pdrop,
                )
                for _ in range(num_layers)
            ]
        )
        # 3. 最终层归一化
        self.ln_final = nn.Identity() if block_type == "no_norm" else RMSNorm(d_model)

        # 4. 独立的语言模型输出层
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_indices.size()
        assert seq_len <= self.context_length

        # 生成位置索引
        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=in_indices.device
        )

        # 词嵌入 + 位置嵌入 + Dropout
        x = self.token_embeddings(in_indices) + self.position_embeddings(
            positions
        )
        x = self.embedding_dropout(x)

        # 穿过所有层
        for layer in self.layers:
            x = layer(x)

        # 最终归一化与输出映射
        x = self.ln_final(x)
        return self.lm_head(x)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """数值稳定的交叉熵损失函数。

    参数:
        logits: 模型的预测得分，形状为 (..., vocab_size)
        targets: 真实的目标 token ID，形状为 (...)，元素为整数
    返回:
        整个批次上的平均损失值（标量标点张量）
    """

    # 1. 获取词表维度上的最大值 M，用于防止指数溢出
    max_val = torch.max(logits, dim=-1, keepdim=True)[0]

    # 2. 减去最大值计算稳定指数项: exp(o_a - M)
    shifted_logits = logits - max_val
    exp_shifted = torch.exp(shifted_logits)

    
    # 3. 沿词表维度求和并取对数，再加上 M: M + log(sum(exp(o_a - M)))
    log_sum_exp = max_val.squeeze(-1) + torch.log(
        torch.sum(exp_shifted, dim=-1)
    )

    # 4. 取出真实目标词对应的预测 logit: o_y
    # 使用 gather 沿着最后一维根据 targets 索引提取对应位置的值
    # targets.unsqueeze(-1) 将形状变为 (..., 1)
    target_logits = torch.gather(logits, dim=-1, index = targets.unsqueeze(-1))
    target_logits = target_logits.squeeze(-1)

    # 5. 单个样本的交叉熵损失: -o_y + log_sum_exp
    loss_per_token = -target_logits + log_sum_exp

    # 6.返回在所有批次/序列维度上的平均值
    return torch.mean(loss_per_token)


