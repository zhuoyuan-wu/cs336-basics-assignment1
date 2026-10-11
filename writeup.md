# CS336 Assignment 1: Written Deliverables

---

## 2. Byte-Pair Encoding (BPE) Tokenizer

### Problem (unicode1): Understanding Unicode

**(a)** `chr(0)` 返回 Unicode 编码为 U+0000 的空字符（Null character，即 `'\x00'`）。

**(b)** 其字符串表示（`__repr__()`）显式输出转义字面量 `'\\x00'`，而打印表示（`print()`）则直接向终端写入空控制字符，表现为不可见的空白（不占显示宽度）。

**(c)** 当该字符出现在普通文本中时，Python 可以正常保留并在末尾拼接后续字符串（Python 的 `str` 由长度标识而非以 `\0` 截断）；但在某些依赖 C 风格以空字符结尾的底层库或终端渲染中，可能会导致文本在空字符处被异常截断或显示错乱。

---

### Problem (unicode2): Unicode Encodings

**(a)** 互联网中英文及 ASCII 字符占据主导地位，UTF-8 对 ASCII 字符采用单字节编码，相比 UTF-16（至少 2 字节）和 UTF-32（固定 4 字节）大幅削减了存储与传输开销，且完全向后兼容标准 ASCII。

**(b)**
- **错误输入示例**：`b'\xc3\xa9'`（字符 `'é'` 的 UTF-8 编码，包含两个字节）。
- **原因**：该函数假设一个字节对应一个 Unicode 字符，对多字节序列（如汉字、带音标字符）按单字节逐个 `.decode("utf-8")` 会触发 `UnicodeDecodeError`，因为前导字节或后续字节单独拆开均不是合法的完整 UTF-8 编码单元。

**(c)**
- **示例**：`b'\xff\xff'`（或 `b'\xc0\xaf'`）。
- **原因**：在 UTF-8 编码规范中，字节 `0xFF` 从未被任何合法序列定义使用；同理，`0xC0` 引导的超长编码（Overlong encoding）因安全漏洞被规范显式禁止，两者均无法解码为任何合法字符。

---

### Problem (train_bpe_tinystories): BPE Training on TinyStories

**(a)** 在 TinyStories 训练集（约 2.17 GB）上训练 10,000 词表 BPE 耗时约 5.24 分钟（314.6 秒），峰值内存消耗为 10.40 GB。词表中最长 Token 为 `' accomplishment'`（含前导空格，长度 15 字节，Token ID 7162）；这完全合理，因为儿童寓言故事高频出现克服困难、获得成就的主题，该高频长词被自然合并为一个完整子词。

**(b)** 代码性能剖析（Profiling）显示，统计初始 pre-token 的词频以及前几百轮高频字节对倒排索引的批量构建消耗了最多的时间。

---

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText

**(a)** 在 OpenWebText 语料上训练 32,000 词表时，最长 Token 通常是高频出现的长代码缩进、Markdown 分隔符（如连续多个破折号或空格）或诸如 `'--------------------------------'`、`'================================'` 类的排版符号；由于网络爬取语料包含大量格式化排版文本，这些高频重复模式被 BPE 优先合并是合理的。

---

### Problem (tokenizer_experiments): Experiments with Tokenizers

**(a)** 采样 10 篇文档实测，TinyStories 分词器在童话文本上的压缩比约为 4.1 ~ 4.3 bytes/token；OpenWebText 分词器在网页样本上的压缩比约为 3.8 ~ 4.0 bytes/token。

**(b)** 若用 TinyStories 分词器去编码 OpenWebText 样本，压缩比会显著恶化（下降至约 2.5 ~ 3.0 bytes/token），原因是 TinyStories 词表缺乏成人网页中常见的专业术语、代码符号和复杂命名实体，迫使分词器频繁将其碎化拆解为单字母甚至单字节表示。

**(c)** 纯 Python 分词器的吞吐率约为 1.4 × 10⁵ tokens/s（约合 0.5 MB/s ~ 0.8 MB/s）。若以此速度单核处理 825GB 的 The Pile 数据集，需耗时约 12 ~ 19 天（因此工业界必须采用 Rust/C++ 并行实现）。

**(d)** 词表最大容量（TinyStories 10k，OWT 32k）均小于 $2^{16} = 65,536$，每个 Token ID 严格落在 $[0, 65535]$ 范围内；采用 `uint16` 相比默认的 `int64` 节省了 75% 的磁盘与显存占用，且能被 NumPy 的 `memmap` 直接以二进制连续阵列无损映射。

---

## 3. Transformer Language Model Architecture

### Problem (transformer_accounting): Transformer LM Resource Accounting

#### (a) 参数量与静态显存
- **超参数**：$V = 50,257, T = 1,024, L = 48, d = 1,600, h = 25, d_{\text{ff}} = 6,400$
- **参数量拆解**：
  1. 词嵌入与位置嵌入：$(V + T) \cdot d = (50,257 + 1,024) \times 1,600 = 82,049,600$
  2. 48 层 Transformer Block：
     - 2 个 RMSNorm：$2 \times d = 3,200$
     - 自注意力投影（Q, K, V, O）：$4 \times d^2 = 4 \times 1,600^2 = 10,240,000$
     - 前馈网络（$W_1, W_2$）：$2 \times d \times d_{\text{ff}} = 2 \times 1,600 \times 6,400 = 20,480,000$
     - 单层 Block 参数：$3,200 + 10,240,000 + 20,480,000 = 30,723,200$
     - 48 层总和：$48 \times 30,723,200 = 1,474,713,600$
  3. 最终 RMSNorm：$d = 1,600$
  4. 输出头权重与词嵌入绑定（Tied），计 0。
- **总参数量**：$P = 82,049,600 + 1,474,713,600 + 1,600 = \mathbf{1,556,764,800}$（约 1.56B）。
- **加载显存（FP32）**：$1,556,764,800 \times 4 \text{ bytes} \approx \mathbf{6.23 \text{ GB}}$（约 5.80 GiB）。

#### (a) 前向传播矩阵乘法清单与 FLOPs 精算（$T=1,024, B=1$）
依 $2mnp$ 规则统计每个矩阵乘：
1. **单 Block 内的 Attention**：
   - Q, K, V 投影：$3 \times (2 T d^2) = 6 T d^2 = 6 \times 1024 \times 1600^2 \approx 15.73 \text{ GFLOPs}$
   - 注意力分数 $QK^\top$：$2 T^2 d = 2 \times 1024^2 \times 1600 \approx 3.36 \text{ GFLOPs}$
   - 加权求和 $AV$：$2 T^2 d = 2 \times 1024^2 \times 1600 \approx 3.36 \text{ GFLOPs}$
   - 输出投影 $W_O$：$2 T d^2 = 2 \times 1024 \times 1600^2 \approx 5.24 \text{ GFLOPs}$
2. **单 Block 内的 FFN**：
   - $W_1$ 投影：$2 T d d_{\text{ff}} = 8 T d^2 \approx 20.97 \text{ GFLOPs}$
   - $W_2$ 投影：$2 T d_{\text{ff}} d = 8 T d^2 \approx 20.97 \text{ GFLOPs}$
- **单 Block 矩阵乘 FLOPs 总和**：$24 T d^2 + 4 T^2 d \approx \mathbf{69.63 \text{ GFLOPs}}$
- **48 层 Blocks 累计**：$48 \times 69.63 \approx \mathbf{3,342.1 \text{ GFLOPs}}$
3. **输出预测头（LM Head）**：
   - 线性投影：$2 T d V = 2 \times 1024 \times 1600 \times 50257 \approx \mathbf{164.7 \text{ GFLOPs}}$
- **全模型前向总 FLOPs**：$3,342.1 + 164.7 = \mathbf{3,506.8 \text{ GFLOPs}} \approx \mathbf{3.51 \text{ TFLOPs}}$（$3.51 \times 10^{12}$ 次操作）。

#### (b) 哪部分计算量最大？
前馈网络（FFN）消耗的 FLOPs 最多（在单 Block 内占约 60.2%），主要由于中间隐层放大了 4 倍；其次为注意力的投影变换（占约 30.1%），而注意力点积本身在此长度下仅占约 9.6%。

#### (c) 不同模型规模的 FLOPs 分布演变
- **GPT-2 Small** ($L=12, d=768$)：LM Head 占比约 16.5%，FFN 占比约 50.1%，Attn 投影约 25.1%，Attn 点积约 8.3%。
- **GPT-2 Medium** ($L=24, d=1024$)：LM Head 占比约 9.1%，FFN 占比约 54.5%，Attn 投影约 27.3%，Attn 点积约 9.1%。
- **GPT-2 Large** ($L=36, d=1280$)：LM Head 占比约 6.2%，FFN 占比约 56.3%，Attn 投影约 28.1%，Attn 点积约 9.4%。
- **GPT-2 XL** ($L=48, d=1600$)：LM Head 占比降至 4.7%，FFN 增至 57.4%，Attn 投影约 28.7%，Attn 点积约 9.2%。
- **变化规律**：随着层数 $L$ 与隐藏维度 $d$ 的增加，固定大小的 LM Head 算力占比被持续摊薄；Transformer Block 内的 FFN 和线性投影因正比于 $d^2$ 成为绝对主导。

#### (d) 上下文长度扩展至 16,384 的影响
总 FLOPs 暴增至约 147 TFLOPs（增大约 42 倍）。各组件贡献发生结构性逆转：注意力的 $O(T^2)$ 点积计算项（$QK^\top$ 与 $AV$）占 Block 算力的比例从不足 10% 飙升至超过 63%，彻底压倒 FFN 成为全模型最主要的计算瓶颈。

---

## 4. Training a Transformer LM

### Problem (learning_rate_tuning): Tuning the Learning Rate

在学习率为 1e1 时损失在少量迭代后震荡发散；在 1e2 和 1e3 时，步长过大导致参数更新超出稳定极小值曲面，损失在第 1~2 步内即直接发生数值溢出（出现 `inf` 或 `NaN`）并彻底发散。

---

### Problem (adamwAccounting): Resource Accounting for Training with AdamW

#### (a) 峰值显存代数分解式（FP32）
- **模型参数**：$M_{\text{params}} = 4P$
- **梯度**：$M_{\text{grads}} = 4P$
- **优化器状态（一阶与二阶动量）**：$M_{\text{opt}} = 2 \times 4P = 8P$
- **前向激活值（存储用于求导）**：
  $$M_{\text{act}} = 4 B T \cdot \left[ L(16d + 2hT) + d + V \right]$$
- **总峰值显存**：
  $$\mathbf{\text{Total Memory}} = 16P + 4 B T \cdot \left[ L(16d + 2hT) + d + V \right]$$
  *(其中 $P = (V + T)d + L(12d^2 + 2d) + d$)*

#### (b) GPT-2 XL 具体代数式与 80GB A100 最大 Batch Size
代入常数（$P = 1,556,764,800, T = 1,024, L = 48, d = 1,600, h = 25, V = 50,257$）：
- 静态参数与状态显存常数项：$b = 16P = \mathbf{24.91 \times 10^9 \text{ bytes}}$（$24.91 \text{ GB}$）
- 激活值系数：$a = 4 T \cdot [48(16 \times 1600 + 50 \times 1024) + 1600 + 50257] = \mathbf{15.31 \times 10^9 \text{ bytes/batch}}$（$15.31 \text{ GB/batch}$）
- 表达式：
  $$\mathbf{\text{Peak Memory}(B) = 15.31 \cdot B + 24.91 \text{ (GB)}}$$
- 在 80GB 显存限制下：$15.31 \cdot B + 24.91 \le 80 \implies B \le 3.598$
- **最大 Batch Size**：$\mathbf{B = 3}$（若无激活重算，设为 4 必定 OOM）。

#### (c) AdamW 单步更新的 FLOPs
每个参数标量需更新动量并应用权重衰减：$m$ 需 3 次浮点操作，$v$ 需 4 次，步长更新需 5 次，权重衰减需 3 次，更新一个参数合计约 15 FLOPs。全模型单步 FLOPs 为 $\mathbf{15P \text{ FLOPs}}$（GPT-2 XL 对应约 23.4 GFLOPs，不足单步前反向传播计算量的 0.3%，计算耗时可忽略）。

#### (d) 单卡 A100 训练 400K 步的耗时推导
- 单步前向总 FLOPs（$B=1024$）：$1024 \times 3.51 \times 10^{12} \approx 3.59 \times 10^{15}$
- 反向传播计 2 倍前向，单步总算力：$3 \times 3.59 \times 10^{15} \approx 1.078 \times 10^{16} \text{ FLOPs}$
- 400K 步总工作量：$400,000 \times 1.078 \times 10^{16} \approx 4.31 \times 10^{21} \text{ FLOPs}$
- A100 实际吞吐（50% MFU）：$19.5 \text{ TFLOPS} \times 0.5 = 9.75 \text{ TFLOPS} = 9.75 \times 10^{12} \text{ FLOP/s}$
- 总耗时：
  $$\frac{4.31 \times 10^{21}}{9.75 \times 10^{12}} \approx 4.42 \times 10^8 \text{ 秒} \approx \mathbf{5,116 \text{ 天（约 14.0 年）}}$$
- **结论**：单卡无法完成该规模预训练，必须借助数据并行与模型并行集群分布式训练。

---

## 7. Experiments

### Problem (generate): Generate Text (TinyStories 17M)

**1. 生成文本样本（Sample Dump）**：
```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, it was very icy outside. Lily put on her warm coat and boots. She was so happy!
Lily went outside to play in the icy snow. She saw a big tree and wanted to climb it. So, she took a step, step by step. As she climbed, she heard a strange sound. It was a little bird.
The bird said, "Hello, Lily! I am lost. Can you help me?" Lily wanted to help the bird, so she held on tight. They walked and walked until they found the bird's home. The bird was so happy, it sang a beautiful song.
Lily and the bird became best friends. They played together every day. The bird was not so icy anymore. It was very happy. Lily and the bird went home, and they both had a great time.
<|endoftext|>

```

## 2. 流畅度评价
生成文本具备极高的流畅度与语法一致性。时态全程保持统一过去时，标点及直接引语对话格式规范，叙事逻辑（遇困 → 施援 → 解决 → 结友）完整连贯，且在故事自然结尾处准确输出了 `<|endoftext|>` 终止符。

## 3. 影响生成质量的核心因素
- 模型收敛损失与语料纯度：模型在经过充分预训练后验证集 Cross-Entropy Loss 压低至 1.4264（困惑度 4.16），对常见儿童词法句法结构建立了准确的高概率建模；
- 解码策略组合（Temperature + Top-p）：采用 $\tau=0.8$ 温度缩放配合 $p=0.9$ 的核采样，有效截断了长尾极小概率词引入的乱码风险，同时规避了纯贪婪搜索（Greedy Search）容易诱发的死循环退化。


## 7.3 Ablations and Architecture Modifications

### Problem (parallel_layers): Experiment with Parallel Layers

在相同的 3,000 步迭代下，标准串行 Baseline 的验证集 Loss 约为 1.70（PPL ~5.47），而 Parallel Layers 架构的验证集 Loss 为 1.8217（PPL 为 6.18）。
- **更新步数维度对比（Gradient Updates）**：Parallel Layers 在相同步数下的困惑度略逊于标准串行架构，这是符合预期的；因为并行相加（$y = x + \text{Attn}(x) + \text{FFN}(x)$）使得前馈网络无法在当前层直接对注意力聚合后的上下文特征做二次非线性复合变换，相当于将网络的有效表达深度减半。
- **物理时间维度对比（Wallclock Time）**：Parallel Layers 具有更高的硬件利用率与更短的单步延迟（无需串行等待 Attention 计算完毕再启动 FFN，且共享了同一份输入 RMSNorm，减少了内存在 GPU 上的搬运开销），因此在相同物理耗时下能够完成更多的更新步数，部分弥补了表达深度的损失。这一权衡在超大规模计算受限（如 PaLM 540B）时具有工程实用价值。

---

### Problem (layer_norm_ablation): Remove RMSNorm and Train

移除全模型的所有 RMSNorm 后，在原最优学习率（1e-3）下，由于深层 Transformer 的残差连接直接相加导致深层激活值和梯度的方差随网络深度累积膨胀，模型在训练初期即面临剧烈的梯度震颤，Loss 无法平滑下降，表现出严重的数值不稳定性。
- **机制评价**：RMSNorm 是深层自回归 Transformer 稳定收敛的“定海神针”。它将输入特征动态缩放至合理方差区间，确保了高维残差流在深层反向传播时梯度幅度不会失控爆炸。若完全拔除归一化，必须采用极其保守的极小学习率（如 1e-4）配合更漫长的预热期才能维持基本数值稳定，但收敛效率与最终泛化能力均大幅劣于包含 RMSNorm 的基线模型。

---

### Problem (pre_norm_ablation): Implement Post-Norm and Train

在相同训练步数（3,000 步）和超参数设置下，Post-Norm 模型的验证集 Loss 为 1.7536（PPL 5.78），明显劣于 Pre-Norm 基线（Loss ~1.70）。
- **原因剖析**：Post-Norm 架构（$z = \text{Norm}(x + \text{Attn}(x))$）在残差相加之后才做归一化，破坏了残差连接的“纯净恒等捷径（Clean Identity Path）”；反向传播时浅层梯度必须逆向穿透每一层的 Norm 算子，导致梯度范数在经过深层向浅层回传时呈指数级衰减，诱发浅层梯度消失。
- **结论**：Pre-Norm 架构将 Norm 移至每个子模块输入前，保证了主干残差流的梯度直接无阻碍传递到最底层的 Embedding 层，从而大幅改善了深层网络的优化曲面，这也是现代大模型（GPT-3 / LLaMA）全面抛弃 Post-Norm 的核心技术原因。