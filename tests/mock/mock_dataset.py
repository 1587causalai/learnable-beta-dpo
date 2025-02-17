from typing import Dict, List, Union
import random
import torch

class MockDataset:
    """Mock的偏好数据集"""
    
    def __init__(self, size: int = 100):
        self.size = size
        self.data = self._generate_data()
        
    def _generate_data(self) -> Dict[str, List[str]]:
        """生成mock数据"""
        prompts = [
            f"Mock prompt {i}" for i in range(self.size)
        ]
        chosen = [
            f"Mock chosen response for prompt {i}" for i in range(self.size)
        ]
        rejected = [
            f"Mock rejected response for prompt {i}" for i in range(self.size)
        ]
        
        return {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
        }
    
    def __getitem__(self, key: Union[str, int]) -> Union[List[str], Dict[str, str]]:
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, int):
            return {
                "prompt": self.data["prompt"][key],
                "chosen": self.data["chosen"][key],
                "rejected": self.data["rejected"][key]
            }
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __len__(self) -> int:
        return self.size
        
    @staticmethod
    def collate_fn(batch):
        """将batch数据整理成模型需要的格式"""
        batch_size = len(batch)
        seq_len = 10  # 为了简单起见，使用固定长度
        
        # 生成随机的input_ids和attention_mask
        chosen_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        chosen_attention_mask = torch.ones_like(chosen_input_ids)
        rejected_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        rejected_attention_mask = torch.ones_like(rejected_input_ids)
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
        }

def load_dataset(name: str, split: str = "train") -> MockDataset:
    """Mock的数据集加载函数"""
    size = 100 if split == "train" else 20
    return MockDataset(size=size) 