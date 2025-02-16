import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import random
import numpy as np

@dataclass
class MockConfig:
    """Mock的模型配置"""
    hidden_size: int = 768
    vocab_size: int = 1000
    max_position_embeddings: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 2

class MockQwenForCausalLM(nn.Module):
    """Mock的Qwen模型"""
    
    def __init__(self, config: Optional[MockConfig] = None):
        super().__init__()
        self.config = config or MockConfig()
        
        # 简单的embedding层和线性层
        self.embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_size
        )
        self.lm_head = nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Mock的前向传播"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 生成随机hidden states
        hidden_states = []
        num_layers = 3  # 模拟3层transformer
        
        # 生成每一层的hidden states
        for _ in range(num_layers):
            layer_hidden = torch.randn(
                batch_size,
                seq_len,
                self.config.hidden_size,
                device=device
            )
            hidden_states.append(layer_hidden)
        
        # 最后一层的hidden states用于生成logits
        last_hidden = hidden_states[-1]
        logits = self.lm_head(last_hidden)
        
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": tuple(hidden_states),  # 转换为tuple以匹配transformers的输出
                "last_hidden_state": last_hidden,
            }
        return logits
        
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Mock的生成函数"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 生成固定长度的序列
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # 确保生成的序列长度不超过max_length
        current_length = input_ids.shape[1]
        remaining_length = max_length - current_length
        if remaining_length <= 0:
            # 如果已经达到最大长度，直接返回输入
            return input_ids.repeat(num_return_sequences, 1)
            
        # 生成新的token
        output_sequences = []
        for _ in range(num_return_sequences):
            # 生成新的token序列
            new_tokens = torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, remaining_length),
                device=device
            )
            # 将输入和新生成的token拼接
            sequence = torch.cat([input_ids, new_tokens], dim=1)
            output_sequences.append(sequence)
        
        # 将所有序列堆叠在一起
        # shape: [batch_size * num_return_sequences, seq_len]
        return torch.cat(output_sequences, dim=0)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Mock的加载函数"""
        return cls()
        
    def save_pretrained(self, save_directory: str):
        """Mock的保存函数"""
        pass

class MockQwenTokenizer:
    """Mock的tokenizer"""
    
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token = "[PAD]"
        self.eos_token = "</s>"
        self.pad_token_id = 0
        self.eos_token_id = 2
        
    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mock的tokenization"""
        if isinstance(text, str):
            text = [text]
            
        # 为每个文本生成随机长度的token序列
        input_ids_list = []
        attention_mask_list = []
        
        for t in text:
            # 生成随机长度的序列
            length = random.randint(10, max_length or 512)
            seq = torch.randint(0, self.vocab_size, (length,))
            mask = torch.ones(length)
            
            input_ids_list.append(seq)
            attention_mask_list.append(mask)
        
        # Padding
        if padding:
            max_len = max(len(ids) for ids in input_ids_list)
            for i in range(len(input_ids_list)):
                pad_len = max_len - len(input_ids_list[i])
                if pad_len > 0:
                    input_ids_list[i] = torch.cat([
                        input_ids_list[i],
                        torch.full((pad_len,), self.pad_token_id)
                    ])
                    attention_mask_list[i] = torch.cat([
                        attention_mask_list[i],
                        torch.zeros(pad_len)
                    ])
        
        # Stack tensors
        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        
        # 创建与transformers tokenizer输出格式一致的对象
        class TokenizerOutput(dict):
            """Mock的tokenizer输出"""
            def __init__(self, input_ids, attention_mask):
                super().__init__()
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                
            def __getattr__(self, key):
                if key in self:
                    return self[key]
                if key == "to":
                    return self._to
                return super().__getattribute__(key)
        
            def _to(self, device):
                """模拟to方法"""
                self.input_ids = self.input_ids.to(device)
                self.attention_mask = self.attention_mask.to(device)
                return self
        
        output = TokenizerOutput(input_ids, attention_mask)
        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        
        return output
    
    def batch_decode(
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Mock的解码函数"""
        return [f"Mock response {i+1}" for i in range(len(sequences))]
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """Mock的编码函数"""
        # 生成随机token序列
        length = random.randint(5, max_length or 20)
        return torch.randint(0, self.vocab_size, (length,)).tolist()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Mock的加载函数"""
        return cls() 