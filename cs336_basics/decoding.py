import torch
import torch.nn.functional as F

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """
    根据给定的最后一个位置的 logits，应用温度缩放和 Top-p 采样，采样出下一个 token ID。
    
    参数:
        logits: shape 为 (vocab_size,) 的一维张量，来自模型对当前序列最后一个 token 的预测输出
        temperature: 温度超参 (tau > 0)
        top_p: 核采样概率阈值 (0.0 < top_p <= 1.0)
        
    返回:
        采样出的整型 token ID
    """
    # 1. 温度缩放 (Temperature Scaling)
    if temperature <= 0.0:
        # 温度接近 0 即为贪婪采样 (Greedy Search)
        return int(torch.argmax(logits).item())
    
    scaled_logits = logits / temperature
    
    # 2. 计算 Softmax 基础概率分布
    # 为保证数值稳定，减去最大值
    scaled_logits = scaled_logits - torch.max(scaled_logits)
    probs = torch.softmax(scaled_logits, dim=-1)

    # 3. Top-p (Nucleus) 截断与重归一化
    if top_p < 1.0:
        # 降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # 计算前缀累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 确定需要移除的尾部 tokens (保留刚刚使得累积概率 >= top_p 的所有 tokens)
        # 将累积概率右移一位，首位置 0
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
        
        # 将被淘汰的候选词概率置为 0
        sorted_probs[sorted_indices_to_remove] = 0.0
        
        # 重新归一化概率分布
        probs_sum = sorted_probs.sum()
        if probs_sum > 0:
            sorted_probs = sorted_probs / probs_sum
        else:
            # 极端防守保护：若全被刷掉，保留概率最大的那个
            sorted_probs[0] = 1.0
            sorted_probs[1:] = 0.0
            
        # 将排好序的概率映射回原始词表索引对应的概率向量
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)

    # 4. 多项式分布采样
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def generate(
    model: torch.nn.Module,
    prompt_tokens: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    context_length: int | None = None,
    device: str = "cpu",
) -> list[int]:
    """
    给定提示词 prompt，自回归逐 token 生成文本序列。
    
    参数:
        model: 训练好的 TransformerLM 模型
        prompt_tokens: 输入的 prompt token ID 列表
        max_new_tokens: 最大生成的新 token 数量
        eos_token_id: 结束标志符 (如 <|endoftext|>) 的 ID，若遇到则提前截断
        temperature: 温度参数
        top_p: Top-p 采样阈值
        context_length: 模型的最大上下文长度 (防止自回归长度超标)
        device: 计算设备
        
    返回:
        包含 prompt 和所有生成 token 的完整 ID 列表
    """
    model.eval()
    tokens = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 如果序列长度超出模型的最大上下文窗口，需要裁剪只保留最新的 context_length 个 token
            input_tokens = tokens
            if context_length is not None and len(input_tokens) > context_length:
                input_tokens = input_tokens[-context_length:]

            # 构造输入张量，形状为 (batch_size=1, sequence_length)
            x = torch.tensor([input_tokens], dtype=torch.long, device=device)
            
            # 模型前向传播，输出 logits 形状为 (1, seq_len, vocab_size)
            logits = model(x)
            
            # 取出最后一个时间步的预测 logits，形状为 (vocab_size,)
            last_logits = logits[0, -1, :]

            # 采样下一个 token
            next_token = sample_next_token(
                last_logits,
                temperature=temperature,
                top_p=top_p,
            )

            tokens.append(next_token)

            # 若遇到结束符，提前停止生成
            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens