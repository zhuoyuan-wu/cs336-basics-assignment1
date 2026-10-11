#!/bin/bash
set -e

# 数据路径
TRAIN_BIN="data/tinystories_train.bin"
VAL_BIN="data/tinystories_valid.bin"
CHECKPOINT_DIR="checkpoints/tinystories_base"

# 创建输出目录
mkdir -p $CHECKPOINT_DIR

echo "[*] 启动 TinyStories 17M Transformer 基础预训练..."

# 预训练命令 (单卡 H100 预计 30-40 分钟；A100 约 50-60 分钟)
python train.py \
  --train_data_path $TRAIN_BIN \
  --val_data_path $VAL_BIN \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 2048 \
  --batch_size 64 \
  --max_iters 20000 \
  --lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 1000 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --log_interval 50 \
  --eval_interval 500 \
  --eval_iters 50 \
  --save_interval 5000 \
  --checkpoint_dir $CHECKPOINT_DIR \
  --exp_name "tinystories_17m_baseline"

echo "[√] 预训练任务完成！"