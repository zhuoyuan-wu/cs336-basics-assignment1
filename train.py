import os
import time
import math
import argparse
import numpy as np
import torch


from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.learning_rate import get_lr_cosine_schedule
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM from scratch (CS336)")

    # 数据集配置
    parser.add_argument("--train_data_path", type=str, required=True, help="训练集二进制文件路径 (.bin 或 .npy)")
    parser.add_argument("--val_data_path", type=str, default=None, help="验证集二进制文件路径")
    parser.add_argument("--data_dtype", type=str, default="uint16", help="数据集存储类型 (如 uint16, int32)")

    # 模型架构配置 (默认采用 TinyStories 17M 规格)
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="最大上下文窗口")
    parser.add_argument("--d_model", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer Block 层数")
    parser.add_argument("--num_heads", type=int, default=16, help="自注意力头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="前馈网络隐层维度 (通常 4 * d_model)")
    parser.add_argument("--attn_pdrop", type=float, default=0.0, help="注意力 Dropout 概率")
    parser.add_argument("--residual_pdrop", type=float, default=0.0, help="残差 Dropout 概率")
    parser.add_argument("--block_type", type=str, default="baseline", choices=["baseline", "parallel", "no_norm", "post_norm"], help="消融实验架构类型")
    # 训练与优化器超参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--max_iters", type=int, default=20000, help="最大训练步数 (iterations)")
    parser.add_argument("--lr", type=float, default=1e-3, help="最高峰值学习率 (alpha_max)")
    parser.add_argument("--min_lr", type=float, default=1e-4, help="余弦退火最小学习率 (alpha_min)")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW 权重衰减系数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="全局梯度裁剪阈值 L2 norm")

    # 评估与落盘 Checkpoint 配置
    parser.add_argument("--eval_interval", type=int, default=500, help="每隔多少步评估一次验证集")
    parser.add_argument("--eval_iters", type=int, default=50, help="每次评估采样多少个 batch 计算均值")
    parser.add_argument("--log_interval", type=int, default=50, help="每隔多少步打印一次日志")
    parser.add_argument("--save_interval", type=int, default=2000, help="每隔多少步保存一次 Checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="权重保存目录")
    parser.add_argument("--resume_path", type=str, default=None, help="从已有 Checkpoint 恢复训练")

    # 系统运行环境
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B 项目名称 (不设则不启用)")
    parser.add_argument("--exp_name", type=str, default="cs336_lm_run", help="实验运行名称")

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, data_arr, batch_size, context_length, eval_iters, device):
    """在数据集上采样 eval_iters 个 batch，无偏估算平均 Loss 与 Perplexity。"""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch(data_arr, batch_size, context_length, device)
        logits = model(inputs)
        # 手写交叉熵计算 (已考虑数值稳定与 batch 均值)
        loss = cross_entropy(logits, targets)
        losses.append(loss.item())
    model.train()
    mean_loss = float(np.mean(losses))
    perplexity = math.exp(mean_loss) if mean_loss < 30.0 else float("inf")
    return mean_loss, perplexity


def main():
    args = parse_args()
    
    # 1. 设定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 2. 初始化 Weights & Biases (如果配置)
    if args.wandb_project and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))

    # 3. 内存映射 (np.memmap) 加载训练与验证集
    print(f"[*] 正在通过 np.memmap 加载训练集: {args.train_data_path}")
    train_data = np.memmap(args.train_data_path, dtype=args.data_dtype, mode="r")
    val_data = None
    if args.val_data_path and os.path.exists(args.val_data_path):
        print(f"[*] 正在通过 np.memmap 加载验证集: {args.val_data_path}")
        val_data = np.memmap(args.val_data_path, dtype=args.data_dtype, mode="r")
    else:
        print("[!] 未指定或未找到独立验证集，将从训练集末尾取数据进行评估。")
        val_data = train_data

    # 4. 实例化模型与优化器
    print("[*] 正在构建 TransformerLM 模型...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        block_type=args.block_type,  
    ).to(args.device)

    # 统计非 Embedding 参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 模型总参数量: {total_params:,} (约 {total_params / 1e6:.2f} M)")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # 5. 断点恢复 (Resume)
    start_iter = 0
    if args.resume_path and os.path.exists(args.resume_path):
        print(f"[*] 正在从 Checkpoint 恢复: {args.resume_path}")
        start_iter = load_checkpoint(args.resume_path, model, optimizer)
        print(f"[*] 成功恢复！从第 {start_iter} 步继续训练。")

    # 6. 开始预训练循环
    print(f"[*] 开始预训练！设备: {args.device}, 总步数: {args.max_iters}, Batch Size: {args.batch_size}")
    model.train()
    step_start_time = time.time()

    for it in range(start_iter, args.max_iters):
        # 6.1 动态调度学习率 (Cosine Annealing with Warmup)
        current_lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        # 6.2 采样批次数据
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)

        # 6.3 前向传播与手写交叉熵 Loss
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # 6.4 反向传播
        loss.backward()

        # 6.5 全局梯度裁剪 (In-place Gradient Clipping)
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), max_l2_norm=args.grad_clip)

        # 6.6 优化器步进更新
        optimizer.step()

        # 6.7 打印训练日志
        if (it + 1) % args.log_interval == 0 or it == 0:
            elapsed = time.time() - step_start_time
            tokens_per_sec = (args.batch_size * args.context_length * args.log_interval) / elapsed if it > 0 else 0
            print(
                f"Step {it+1:6d}/{args.max_iters} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e} | "
                f"吞吐: {tokens_per_sec:.1f} tok/s | "
                f"耗时/区间: {elapsed:.2f}s"
            )
            step_start_time = time.time()

        # 6.8 定期评估验证集
        if (it + 1) % args.eval_interval == 0:
            val_loss, val_ppl = estimate_loss(
                model, val_data, args.batch_size, args.context_length, args.eval_iters, args.device
            )
            print(f"--> [Eval @ Step {it+1}] 验证集 Loss: {val_loss:.4f} | 困惑度 (PPL): {val_ppl:.2f}")
            if args.wandb_project and WANDB_AVAILABLE:
                wandb.log({
                    "iter": it + 1,
                    "train_loss": loss.item(),
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "lr": current_lr,
                })

        # 6.9 保存 Checkpoint
        if (it + 1) % args.save_interval == 0 or (it + 1) == args.max_iters:
            ckpt_file = os.path.join(args.checkpoint_dir, f"model_step_{it+1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_file)
            print(f"[*] Checkpoint 已保存至: {ckpt_file}")

    print("[*] 预训练完成！")


if __name__ == "__main__":
    main()