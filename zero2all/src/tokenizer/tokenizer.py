import os
import json
import regex as re
from typing import List, Dict, Union, Optional, Tuple, Any


class ByteLevelBPETokenizer:
    """
    字节级BPE分词器实现
    基于Qwen系列模型的分词器设计
    """
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        add_prefix_space: bool = False,
        special_tokens: Dict[str, int] = None,
    ):
        self.add_prefix_space = add_prefix_space
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 默认的特殊token
        self.special_tokens = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
        }
        
        # 合并用户提供的特殊token
        if special_tokens:
            self.special_tokens.update(special_tokens)
        
        # 加载词汇表和合并规则
        if vocab_file and os.path.isfile(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}
        else:
            # 初始化空词汇表
            self.encoder = {}
            self.decoder = {}
        
        if merges_file and os.path.isfile(merges_file):
            with open(merges_file, 'r', encoding='utf-8') as f:
                bpe_merges = f.read().split('\n')
                # 跳过文件头部的注释
                bpe_merges = [m for m in bpe_merges if m and not m.startswith('#')]
            self.bpe_ranks = {tuple(merge.split()): i for i, merge in enumerate(bpe_merges)}
        else:
            # 初始化空的合并规则
            self.bpe_ranks = {}
        
        # 编译常用的正则表达式
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        # 初始化缓存
        self.cache = {}
    
    def _bytes_to_unicode(self) -> Dict[int, str]:
        """
        字节到Unicode的映射，用于字节级分词
        """
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs.copy()
        n = 0
        
        # 处理控制字符和不在上述范围内的字符
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        
        # 返回字节到Unicode字符的映射
        return {b: chr(c) for b, c in zip(bs, cs)}
    
    def _encode_byte_tokens(self, text: str) -> List[str]:
        """
        将文本编码为字节级别的token
        """
        if self.add_prefix_space and not text.startswith(" "):
            text = " " + text
        
        # 使用正则表达式拆分文本
        result = []
        for token in re.findall(self.pattern, text):
            # 对每个字符的每个字节进行编码
            token_bytes = token.encode('utf-8')
            token_chars = [self.byte_encoder[b] for b in token_bytes]
            result.extend(token_chars)
        
        return result
    
    def _bpe(self, token: str) -> str:
        """
        应用BPE（字节对编码）算法
        """
        # 检查缓存
        if token in self.cache:
            return self.cache[token]
        
        # 将token拆分为单个字符
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # 找到最高优先级的合并对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # 如果找不到合并规则，退出
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            
            # 应用合并规则
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            if len(word) == 1:
                break
                
            pairs = self._get_pairs(word)
        
        # 将结果添加到缓存
        result = ' '.join(word)
        self.cache[token] = result
        return result
    
    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """
        从word中获取相邻对
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token ID
        """
        bpe_tokens = []
        
        # 处理特殊token
        for special_token, token_id in self.special_tokens.items():
            if special_token in text:
                text = text.replace(special_token, f" {special_token} ")
        
        # 编码文本
        tokens = self._encode_byte_tokens(text)
        for token in tokens:
            bpe_result = self._bpe(token)
            for bpe_token in bpe_result.split(' '):
                # 查找token ID
                if bpe_token in self.encoder:
                    bpe_tokens.append(self.encoder[bpe_token])
                else:
                    # 处理未知token
                    bpe_tokens.append(self.encoder.get("<unk>", len(self.encoder) - 1))
        
        return bpe_tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将token ID解码为文本
        """
        # 处理特殊token
        tokens = []
        for token_id in token_ids:
            # 查找特殊token
            if token_id in self.special_tokens.values():
                for special_token, special_id in self.special_tokens.items():
                    if token_id == special_id:
                        tokens.append(special_token)
                        break
            # 查找普通token
            elif token_id in self.decoder:
                tokens.append(self.decoder[token_id])
        
        # 合并token并解码字节
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
        return text
    
    def __call__(
        self, 
        text: Union[str, List[str]], 
        padding: bool = False, 
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        便捷的编码接口
        """
        # 处理单个文本或文本列表
        if isinstance(text, str):
            text = [text]
        
        # 编码每个文本
        input_ids = [self.encode(t) for t in text]
        
        # 应用截断
        if truncation and max_length:
            input_ids = [ids[:max_length] for ids in input_ids]
        
        # 应用填充
        attention_mask = None
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            if max_length:
                max_len = min(max_len, max_length)
            
            attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids]
            input_ids = [ids + [self.special_tokens["<pad>"]] * (max_len - len(ids)) for ids in input_ids]
        
        # 根据需要转换为张量
        if return_tensors == "pt":
            import torch
            input_ids = torch.tensor(input_ids)
            if attention_mask:
                attention_mask = torch.tensor(attention_mask)
        
        # 构建返回结果
        result = {"input_ids": input_ids}
        if attention_mask:
            result["attention_mask"] = attention_mask
        
        return result
    
    def save_pretrained(self, save_dir: str) -> None:
        """
        保存分词器到指定目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存词汇表
        vocab_file = os.path.join(save_dir, "vocab.json")
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        
        # 保存合并规则
        merges_file = os.path.join(save_dir, "merges.txt")
        with open(merges_file, 'w', encoding='utf-8') as f:
            # 添加注释头
            f.write("# Byte-level BPE merges\n")
            for merge, _ in sorted(self.bpe_ranks.items(), key=lambda x: x[1]):
                f.write(f"{merge[0]} {merge[1]}\n")
        
        # 保存配置
        config_file = os.path.join(save_dir, "tokenizer_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            config = {
                "add_prefix_space": self.add_prefix_space,
                "special_tokens": self.special_tokens,
            }
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"分词器保存到 {save_dir}")
    
    @classmethod
    def from_pretrained(cls, pretrained_dir: str) -> 'ByteLevelBPETokenizer':
        """
        从预训练目录加载分词器
        """
        # 加载配置
        config_file = os.path.join(pretrained_dir, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}
        
        # 创建分词器实例
        vocab_file = os.path.join(pretrained_dir, "vocab.json")
        merges_file = os.path.join(pretrained_dir, "merges.txt")
        
        return cls(
            vocab_file=vocab_file,
            merges_file=merges_file,
            add_prefix_space=config.get("add_prefix_space", False),
            special_tokens=config.get("special_tokens", None),
        )
    
    def train(
        self, 
        files: Union[str, List[str]], 
        vocab_size: int = 151646, 
        min_frequency: int = 2, 
        special_tokens: List[str] = None,
    ) -> None:
        """
        在文本语料库上训练分词器
        
        注意: 这是一个简化版本的训练过程，完整实现需要更多代码
        """
        # 处理单个文件或文件列表
        if isinstance(files, str):
            files = [files]
        
        # 收集词频
        word_freqs = {}
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 编码为字节级别的token
                    tokens = self._encode_byte_tokens(line)
                    for token in tokens:
                        if token not in word_freqs:
                            word_freqs[token] = 0
                        word_freqs[token] += 1
        
        # 过滤低频词
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_frequency}
        
        # 初始化BPE
        pairs = self._compute_pair_frequencies(word_freqs)
        self.bpe_ranks = {}
        
        # 迭代合并
        vocab_size_threshold = vocab_size - len(self.special_tokens)
        for i, (pair, _) in enumerate(pairs):
            if len(self.encoder) >= vocab_size_threshold:
                break
                
            # 更新合并规则
            self.bpe_ranks[pair] = i
            
            # 更新词汇表
            self._update_vocabulary_after_merge(pair, word_freqs)
            
            # 重新计算对频率
            pairs = self._compute_pair_frequencies(word_freqs)
        
        # 添加特殊token
        if special_tokens:
            for token in special_tokens:
                if token not in self.encoder:
                    self.encoder[token] = len(self.encoder)
        
        # 更新解码器
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 清除缓存
        self.cache = {}
    
    def _compute_pair_frequencies(self, word_freqs: Dict[str, int]) -> List[Tuple[Tuple[str, str], int]]:
        """
        计算字节对频率，用于BPE训练
        """
        pairs = {}
        for word, freq in word_freqs.items():
            chars = tuple(word)
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                if pair not in pairs:
                    pairs[pair] = 0
                pairs[pair] += freq
        
        # 按频率排序
        return sorted(pairs.items(), key=lambda x: -x[1])
    
    def _update_vocabulary_after_merge(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> None:
        """
        合并一对字节后更新词汇表
        """
        first, second = pair
        new_token = first + second
        
        # 添加新token到词汇表
        if new_token not in self.encoder:
            self.encoder[new_token] = len(self.encoder)
        
        # 更新单词列表
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            # 使用BPE合并规则处理单词
            i = 0
            new_word = []
            chars = list(word)
            
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == first and chars[i + 1] == second:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(chars[i])
                    i += 1
            
            # 更新词频表
            new_word = ''.join(new_word)
            new_word_freqs[new_word] = freq
        
        word_freqs.clear()
        word_freqs.update(new_word_freqs) 