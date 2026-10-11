#!/bin/bash
set -e

TRAIN_BIN="data/tinystories_train.bin"
VAL_BIN="data/tinystories_valid.bin"

echo "=================================================="
echo "1. 运行 Parallel Layers 并行层实验 (PaLM 风格)"
echo "=================================================="
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
  --max_iters 3000 \
  --lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 500 \
  --eval_interval 200 \
  --save_interval 3000 \
  --block_type parallel \
  --checkpoint_dir checkpoints/ablation_parallel \
  --exp_name "ablation_parallel"

echo "=================================================="
echo "2. 运行 No RMSNorm 移除归一化实验 (观察高 lr 发散)"
echo "=================================================="
# 拔掉 RMSNorm 后，1e-3 极易梯度爆炸/发散，见证定海神针失效
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
  --max_iters 1000 \
  --lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 200 \
  --eval_interval 100 \
  --save_interval 1000 \
  --block_type no_norm \
  --checkpoint_dir checkpoints/ablation_no_norm \
  --exp_name "ablation_no_norm" || echo "--> [现象记录] No RMSNorm 在 1e-3 学习率下发散崩溃！"

echo "=================================================="
echo "3. 运行 Post-Norm 传统架构实验 (对比收敛速度)"
echo "=================================================="
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
  --max_iters 3000 \
  --lr 1e-3 \
  --min_lr 1e-4 \
  --warmup_iters 500 \
  --eval_interval 200 \
  --save_interval 3000 \
  --block_type post_norm \
  --checkpoint_dir checkpoints/ablation_post_norm \
  --exp_name "ablation_post_norm"

echo "[√] 三大消融实验全部执行完毕！"