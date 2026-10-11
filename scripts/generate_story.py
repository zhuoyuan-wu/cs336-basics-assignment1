import os
import sys
# 注入项目根目录路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import argparse
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import generate


def parse_args():
    parser = argparse.ArgumentParser(description="TinyStories 文本生成器")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型权重 .pt 文件路径")
    parser.add_argument("--tokenizer_dir", type=str, default="output/tinystories_bpe", help="分词器保存目录")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="提示词")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="最多生成的新 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 核采样阈值")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 模型架构配置 (默认采用 TinyStories 17M 规格，但也支持测试用微型规格)
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="最大上下文窗口")
    parser.add_argument("--d_model", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer Block 层数")
    parser.add_argument("--num_heads", type=int, default=16, help="自注意力头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈网络隐层维度")

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 加载 Tokenizer
    vocab_path = os.path.join(args.tokenizer_dir, "vocab.pkl")
    merges_path = os.path.join(args.tokenizer_dir, "merges.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_path, "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])

    # 2. 构造模型并加载权重
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(args.device)

    print(f"[*] 正在加载 Checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 3. 编码 Prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    try:
        eos_token_id = tokenizer.encode("<|endoftext|>")[0]
    except Exception:
        eos_token_id = None

    print(f"[*] 提示词 (Prompt): {args.prompt}")
    print(f"[*] 生成参数: Temperature={args.temperature}, Top-p={args.top_p}")
    print("=" * 60)

    # 4. 自回归生成
    out_tokens = generate(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
        context_length=args.context_length,
        device=args.device,
    )

    # 5. 解码并打印故事
    generated_text = tokenizer.decode(out_tokens)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()