"""cs336_basics/bpe.py

高效字节级 BPE 分词器训练实现。
"""

from collections import defaultdict
import regex

GPT2_SPLIT_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """从训练文本中学习字节级 BPE 词表和合并规则。

    文本首先按照 GPT-2 的正则表达式切分成若干独立片段；每个片段
    再拆成单字节 token。之后反复选择出现频率最高的相邻 token pair，
    将它们合并成一个新 token，直到达到目标词表大小。返回的 merges
    按学习顺序排列，编码时也必须按照这个顺序应用。

    这里的频率是“片段出现次数”加权后的频率，而不是不同片段的数量。
    例如某个片段在语料中出现 10 次，其中的 pair 就应当贡献 10 次。
    """
    if special_tokens is None:
        special_tokens = []

    # 词表 ID 先分配给 special token，再分配给 0~255 的全部单字节。
    # 这样无论训练文本中是否出现某个字节，所有原始字节都一定可以编码。
    vocab: dict[int, bytes] = {}
    current_id = 0

    for token_str in special_tokens:
        vocab[current_id] = token_str.encode("utf-8")
        current_id += 1

    for b in range(256):
        vocab[current_id] = bytes([b])
        current_id += 1

    # 每执行一次合并，就会向词表增加一个 token；因此目标词表大小
    # 减去初始词表大小，就是最多可以学习的合并规则数量。
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # 预分词必须先于字节拆分完成。这样 pair 只会在同一个 GPT-2
    # 片段内部统计，不会错误地跨越单词、空格或标点的边界。
    compiled_pattern = regex.compile(GPT2_SPLIT_REGEX)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 使用“唯一片段 -> 出现次数”的形式压缩语料：相同片段只保存一份
    # token 序列，但在 pair 统计时仍然用它的频次进行加权。
    raw_word_counts = defaultdict(int)
    for match in compiled_pattern.finditer(text):
        raw_word_counts[match.group().encode("utf-8")] += 1

    # words[word_idx] 保存一个唯一片段当前的 token 序列。
    # 序列会在每轮合并后被替换，因此使用 list 而不是不可变 tuple。
    words: list[list[bytes]] = [
        [bytes([b]) for b in w] for w in raw_word_counts.keys()
    ]
    # word_freqs[word_idx] 与 words[word_idx] 一一对应，且在训练过程中不变。
    word_freqs: list[int] = list(raw_word_counts.values())

    # pair_counts 保存全局加权频率：pair_counts[(a, b)] 表示相邻的 a、b
    # 在完整语料中出现了多少次。
    # pair_to_words 是倒排索引，只记录 pair 出现在哪些唯一片段中。
    # 两张表分别服务于“选择最高频 pair”和“快速找到需要更新的片段”。
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for word_idx, (word_tokens, freq) in enumerate(zip(words, word_freqs)):
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word_idx)

    merges: list[tuple[bytes, bytes]] = []

    # 每轮只改变包含 best_pair 的片段。其他片段的 token 序列没有改变，
    # 所以它们的 pair 计数和倒排索引也不需要重新计算。
    for _ in range(num_merges):
        if not pair_counts:
            break

        # 先按频率最大化；频率相同的时候，按 pair 的字节序取较大的一个。
        # 这个确定性的 tie-break 规则保证相同输入每次都会得到相同词表。
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        # 正常情况下计数为正。计数为 0 表示已经没有有效的连续 pair，
        # 继续循环不会产生有意义的合并。
        if pair_counts[best_pair] == 0:
            break

        # 新 token 的内容就是两个旧 token 的字节拼接；merges 保存旧 token
        # 的顺序，供后续 tokenizer 按学习顺序重放这些合并。
        merges.append(best_pair)
        vocab[current_id] = best_pair[0] + best_pair[1]
        current_id += 1

        # 核心优化：只遍历包含 best_pair 的片段。删除 best_pair 后，
        # pair_to_words 中剩余条目仍然表示尚未更新的旧状态。
        affected_word_indices = list(pair_to_words[best_pair])
        # best_pair 本轮已经被选中并处理，不应在下一轮继续作为旧 pair 存在。
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word_idx in affected_word_indices:
            word = words[word_idx]
            freq = word_freqs[word_idx]

            # 第一步：撤销该片段旧 token 序列对全局账本的贡献。
            # 同一个唯一片段可能出现很多次，所以每个 pair 都要减去 freq，
            # 而不是只减 1；倒排索引也要同步移除这个片段下标。
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                if p in pair_counts:
                    pair_counts[p] -= freq
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                if p in pair_to_words:
                    pair_to_words[p].discard(word_idx)

            # 第二步：从左到右扫描旧序列，把每个不重叠的 best_pair 替换成
            # 一个拼接后的 token。i += 2 可以确保一次合并不会重复消费 token。
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # 替换缓存中的当前状态，后续轮次将在这个新序列上继续合并。
            words[word_idx] = new_word

            # 第三步：把新序列中的 pair 重新加入全局账本和倒排索引。
            # 只有合并边界附近的 pair 会发生变化，但这里重新登记整个新序列，
            # 可以保持实现简单，并保证两张索引表始终与 words 完全一致。
            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word_idx)

    return vocab, merges