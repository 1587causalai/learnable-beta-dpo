from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class DPODataset(Dataset):
    """DPO训练数据集
    
    处理包含prompt和偏好对（chosen/rejected responses）的数据
    """
    
    def __init__(
        self,
        prompts: List[str],
        chosen_responses: List[str],
        rejected_responses: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        assert len(prompts) == len(chosen_responses) == len(rejected_responses), \
            "所有输入列表长度必须相同"
            
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.prompts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]
        
        # 构建完整的文本序列
        chosen_text = f"{prompt}{chosen}"
        rejected_text = f"{prompt}{rejected}"
        
        # tokenization
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokens.input_ids.squeeze(0),
            "chosen_attention_mask": chosen_tokens.attention_mask.squeeze(0),
            "rejected_input_ids": rejected_tokens.input_ids.squeeze(0),
            "rejected_attention_mask": rejected_tokens.attention_mask.squeeze(0),
        }
        
    @classmethod
    def from_hf_dataset(
        cls,
        dataset,
        tokenizer: PreTrainedTokenizer,
        prompt_col: str = "prompt",
        chosen_col: str = "chosen",
        rejected_col: str = "rejected",
        max_length: int = 512,
    ) -> "DPODataset":
        """从Hugging Face数据集创建DPODataset
        
        Args:
            dataset: Hugging Face数据集
            tokenizer: tokenizer实例
            prompt_col: prompt列名
            chosen_col: chosen response列名
            rejected_col: rejected response列名
            max_length: 最大序列长度
            
        Returns:
            DPODataset实例
        """
        prompts = dataset[prompt_col]
        chosen_responses = dataset[chosen_col]
        rejected_responses = dataset[rejected_col]
        
        return cls(
            prompts=prompts,
            chosen_responses=chosen_responses,
            rejected_responses=rejected_responses,
            tokenizer=tokenizer,
            max_length=max_length,
        )

    @classmethod
    def collate_fn(cls, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """自定义的collate函数，处理不同长度的序列
        
        Args:
            batch: 包含多个样本的列表
            
        Returns:
            包含batch数据的字典
        """
        # 获取batch中的最大长度
        max_chosen_length = max(x["chosen_input_ids"].size(0) for x in batch)
        max_rejected_length = max(x["rejected_input_ids"].size(0) for x in batch)
        
        # 初始化输出张量
        batch_size = len(batch)
        chosen_input_ids = torch.full(
            (batch_size, max_chosen_length),
            batch[0]["chosen_input_ids"].new_full((1,), 0)[0]  # pad_token_id
        )
        chosen_attention_mask = torch.zeros((batch_size, max_chosen_length))
        rejected_input_ids = torch.full(
            (batch_size, max_rejected_length),
            batch[0]["rejected_input_ids"].new_full((1,), 0)[0]  # pad_token_id
        )
        rejected_attention_mask = torch.zeros((batch_size, max_rejected_length))
        
        # 填充数据
        for i, item in enumerate(batch):
            # Chosen
            c_len = item["chosen_input_ids"].size(0)
            chosen_input_ids[i, :c_len] = item["chosen_input_ids"]
            chosen_attention_mask[i, :c_len] = item["chosen_attention_mask"]
            
            # Rejected
            r_len = item["rejected_input_ids"].size(0)
            rejected_input_ids[i, :r_len] = item["rejected_input_ids"]
            rejected_attention_mask[i, :r_len] = item["rejected_attention_mask"]
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }
