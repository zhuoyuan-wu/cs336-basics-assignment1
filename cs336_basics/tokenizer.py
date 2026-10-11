"""cs336_basics/tokenizer.py

BPE 分词器类的实现。
支持文本与 token ID 之间的双向编解码、从文件反序列化加载、以及对大文件的流式分词。
"""

from collections.abc import Iterable, Iterator
import json
import os
import pickle
import regex

# GPT-2 官方使用的预分词正则表达式
GPT2_SPLIT_REGEX = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """初始化 Tokenizer。

        参数:
            vocab: 词表映射，格式为 {token_id: token_bytes}
            merges: 合并规则列表，元素为 (token1_bytes, token2_bytes)，顺序决定优先级
            special_tokens: 特殊标记字符串列表（如 ["<|endoftext|>"]）
        """
        # 浅拷贝一份词表，避免外部修改影响内部状态
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = list(merges)

        # 1. 建立合并规则的优先级字典 (bpe_ranks)
        # 越早被创建的合并规则（index 越小），优先级越高（rank 越小）
        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # 2. 处理特殊标记 (special_tokens)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # 作业硬性规定：如果某些 special_token 还没有在词表里，需要将它们追加到词表末尾！
        existing_byte_values = set(self.vocab.values())
        next_id = max(self.vocab.keys(), default=-1) + 1

        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes not in existing_byte_values:
                self.vocab[next_id] = token_bytes
                existing_byte_values.add(token_bytes)
                next_id += 1

        # 3. 建立反向词表：从字节串映射到整数 ID {token_bytes: token_id}，编码查表时 O(1) 速度
        self.byte_to_id: dict[bytes, int] = {
            v: k for k, v in self.vocab.items()
        }

        # 4. 构建针对特殊字符的正则匹配模式
        # 当文本中包含特殊字符时，需要将其作为最高优先级完整抠出，不能被普通正则切散
        if self.special_tokens:
            # 按长度从长到短排序，防止前缀重叠时短的先匹配（例如避免 <|end|> 抢先匹配了 <|endoftext|>）
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            # 对特殊符号（如 |、<、>）进行正则转义
            escaped_specials = [regex.escape(s) for s in sorted_specials]
            self.special_pattern = regex.compile(
                "(" + "|".join(escaped_specials) + ")"
            )
        else:
            self.special_pattern = None

        # 编译 GPT-2 的文本预分词正则
        self.text_pattern = regex.compile(GPT2_SPLIT_REGEX)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        """类方法：从磁盘上的序列化文件加载并构建 Tokenizer。

        兼容 pickle 格式与 json/文本 格式。
        """
        vocab_path = str(vocab_filepath)
        merges_path = str(merges_filepath)

        # 1. 加载 vocab 词表
        # 首先尝试用 pickle 加载（常用于二进制序列化）
        try:
            with open(vocab_path, "rb") as f:
                raw_vocab = pickle.load(f)
                # 确保 key 是 int，value 是 bytes
                vocab = {int(k): bytes(v) for k, v in raw_vocab.items()}
        except Exception:
            # 如果不是 pickle，按 JSON 读取
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw_vocab = json.load(f)
                vocab = {}
                for k, v in raw_vocab.items():
                    # 处理 JSON 中存的字符串或字符编码
                    val_bytes = (
                        v.encode("latin-1")
                        if isinstance(v, str)
                        else bytes(v)
                    )
                    vocab[int(k)] = val_bytes

        # 2. 加载 merges 合并规则
        try:
            with open(merges_path, "rb") as f:
                raw_merges = pickle.load(f)
                merges = [(bytes(p[0]), bytes(p[1])) for p in raw_merges]
        except Exception:
            merges = []
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(" ")
                    if len(parts) == 2:
                        merges.append(
                            (
                                parts[0].encode("latin-1"),
                                parts[1].encode("latin-1"),
                            )
                        )

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _bpe_merge(self, word_tokens: list[bytes]) -> list[bytes]:
        """核心内部方法：对单个词块的字节碎片列表，贪心执行 BPE 合并。

        算法逻辑：
        反复扫描当前碎片中的所有相邻 pair，
        找出在 self.bpe_ranks 中排名最靠前（即优先级最高）的那一对进行粘合，
        直到没有任何一对能继续合并为止。
        """
        if len(word_tokens) <= 1:
            return word_tokens

        tokens = word_tokens

        while True:
            min_rank = float("inf")
            best_pair = None

            # 遍历当前单词中所有相邻的字节对
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                # 查询该 pair 在合并规则中的优先级
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            # 如果没有找到任何可以合并的 pair，说明已经合并到极限，退出
            if best_pair is None or min_rank == float("inf"):
                break

            # 就地执行合并：把所有的 (best_pair[0], best_pair[1]) 替换为 (best_pair[0] + best_pair[1])
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2  # 跳过被合并的两个碎片
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            if len(tokens) <= 1:
                break

        return tokens

    def encode(self, text: str) -> list[int]:
        """将输入的文本字符串编码为整数 token ID 列表。"""
        if not text:
            return []

        # 1. 如果存在特殊字符，先将特殊字符作为分界线把原文本切开
        # 比如 "hello<|endoftext|>world" 会被切成 ["hello", "<|endoftext|>", "world"]
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]

        token_ids: list[int] = []

        for part in parts:
            if not part:
                continue

            # 2. 如果这部分是特殊字符，直接作为整体查表转为 ID
            if part in self.special_tokens:
                token_ids.append(self.byte_to_id[part.encode("utf-8")])
                continue

            # 3. 如果是普通文本，使用 GPT-2 正则进行粗粒度预分词
            for match in self.text_pattern.finditer(part):
                word_bytes = match.group().encode("utf-8")

                # 先打散为单字节列表，如 b"cat" -> [b'c', b'a', b't']
                word_tokens = [bytes([b]) for b in word_bytes]

                # 应用 BPE 贪心合并规则
                merged_tokens = self._bpe_merge(word_tokens)

                # 将每个合并后的 token 字节串转换为对应的整数 ID
                for tok in merged_tokens:
                    token_ids.append(self.byte_to_id[tok])

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """流式编码：针对大文件或数据流，逐行/逐块进行分词，

        惰性产出（yield）token ID，避免内存暴涨。
        """
        for text_chunk in iterable:
            # yield from 将子列表里的每一个 ID 逐个产出给调用方
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        """将整数 token ID 列表还原解码为原始文本字符串。"""
        if not ids:
            return ""

        # 1. 查找每个 ID 对应的原始字节串，并全部拼接在一起
        raw_bytes = b"".join(self.vocab[i] for i in ids)

        # 2. 将拼接好的字节串以 UTF-8 解码回文本
        # 注意作业规范：传入 errors='replace'，遇到残缺非法的字节自动替换为  (U+FFFD)
        return raw_bytes.decode("utf-8", errors="replace")