import os
import sys

# 将项目根目录加入模块搜索路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import pickle
import resource
import argparse
import numpy as np



# 接下来再正常导包
from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="TinyStories BPE 训练与预分词流水线")
    parser.add_argument("--data_dir", type=str, default="data", help="数据存放目录")
    parser.add_argument("--output_dir", type=str, default="output/tinystories_bpe", help="分词器输出保存目录")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--special_token", type=str, default="<|endoftext|>", help="特殊终止符")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    train_txt_path = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-train.txt")
    valid_txt_path = os.path.join(args.data_dir, "TinyStoriesV2-GPT4-valid.txt")

    print(f"==================================================")
    print(f"[*] 步骤 1: 开始在 TinyStories 上训练 BPE 分词器...")
    print(f"[*] 输入语料: {train_txt_path}")
    print(f"[*] 目标词表大小: {args.vocab_size}")
    print(f"==================================================")

    start_time = time.time()
    
    # 1. 训练 BPE
    vocab, merges = train_bpe(
        input_path=train_txt_path,
        vocab_size=args.vocab_size,
        special_tokens=[args.special_token],
    )
    
    elapsed_time = time.time() - start_time
    # 统计峰值内存 (Linux 下单位为 KB)
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(f"[+] BPE 训练完成！耗时: {elapsed_time / 60.0:.2f} 分钟 ({elapsed_time:.1f} 秒)")
    print(f"[+] 峰值内存使用: {peak_ram_mb:.2f} MB ({peak_ram_mb / 1024.0:.2f} GB)")

    # 2. 保存词表和 merges
    vocab_path = os.path.join(args.output_dir, "vocab.pkl")
    merges_path = os.path.join(args.output_dir, "merges.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"[+] 词表已保存至: {vocab_path}")
    print(f"[+] Merges 已保存至: {merges_path}")

    # 3. 统计并寻找最长的 Token (Writeup 必答题)
    longest_token_bytes = b""
    longest_token_id = -1
    for token_id, token_bytes in vocab.items():
        if len(token_bytes) > len(longest_token_bytes):
            longest_token_bytes = token_bytes
            longest_token_id = token_id

    try:
        longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")
    except Exception:
        longest_token_str = repr(longest_token_bytes)

    print(f"--------------------------------------------------")
    print(f"[Writeup 答案提取] 最长的 Token ID: {longest_token_id}")
    print(f"[Writeup 答案提取] 最长 Token 字节长度: {len(longest_token_bytes)} bytes")
    print(f"[Writeup 答案提取] 最长 Token 内容: {repr(longest_token_str)}")
    print(f"--------------------------------------------------")

    # 4. 实例化 Tokenizer，开始对全量语料进行预编码
    print(f"[*] 步骤 2: 正在加载 Tokenizer 对全量语料进行编码...")
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=[args.special_token])

    for split_name, txt_file in [("valid", valid_txt_path), ("train", train_txt_path)]:
        if not os.path.exists(txt_file):
            print(f"[!] 跳过 {split_name}，文件不存在: {txt_file}")
            continue

        bin_file = os.path.join(args.data_dir, f"tinystories_{split_name}.bin")
        print(f"[*] 正在流式编码 {txt_file} -> {bin_file} (uint16)...")
        
        split_start = time.time()
        with open(txt_file, "r", encoding="utf-8") as f:
            # 流式编码，避免几十 GB 内存占用
            token_ids = list(tokenizer.encode_iterable(f))
        
        # 转换为 uint16 格式二进制落盘
        arr = np.array(token_ids, dtype=np.uint16)
        arr.tofile(bin_file)
        
        split_elapsed = time.time() - split_start
        print(f"[+] {split_name} 编码完成！Token 总数: {len(arr):,}，耗时: {split_elapsed:.1f} 秒，已写入 {bin_file}")

    print(f"[√] 实验一全部数据与分词准备工作圆满结束！")


if __name__ == "__main__":
    main()