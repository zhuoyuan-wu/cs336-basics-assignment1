#!/bin/bash
set -e

mkdir -p data
cd data

echo "[*] 开始下载 TinyStories 数据集..."
# 使用 hf-mirror 国内外均可满速极速下载
wget -c https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget -c https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

echo "[*] 开始下载 OpenWebText 样本数据集..."
wget -c https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip -f owt_train.txt.gz
wget -c https://hf-mirror.com/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

cd ..
echo "[√] 所有数据下载并解压完毕！"