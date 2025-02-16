from typing import Dict, List
import random

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
    
    def __getitem__(self, key: str) -> List[str]:
        return self.data[key]
    
    def __len__(self) -> int:
        return self.size

def load_dataset(name: str, split: str = "train") -> MockDataset:
    """Mock的数据集加载函数"""
    size = 100 if split == "train" else 20
    return MockDataset(size=size) 