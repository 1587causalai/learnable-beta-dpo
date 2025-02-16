import unittest
import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.data.dataset import DPODataset
from tests.mock.mock_qwen import MockQwenTokenizer
from tests.mock.mock_dataset import load_dataset, MockDataset

class TestDPODataset(unittest.TestCase):
    """测试DPO数据集处理"""
    
    def setUp(self):
        """设置测试环境"""
        self.tokenizer = MockQwenTokenizer()
        self.mock_dataset = load_dataset("mock_dataset")
        self.max_length = 512
        
    def test_dataset_creation(self):
        """测试数据集创建"""
        dataset = DPODataset(
            prompts=self.mock_dataset["prompt"],
            chosen_responses=self.mock_dataset["chosen"],
            rejected_responses=self.mock_dataset["rejected"],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        # 检查数据集大小
        self.assertEqual(len(dataset), len(self.mock_dataset))
        
        # 检查单个样本
        sample = dataset[0]
        self.assertIn("chosen_input_ids", sample)
        self.assertIn("chosen_attention_mask", sample)
        self.assertIn("rejected_input_ids", sample)
        self.assertIn("rejected_attention_mask", sample)
        
        # 检查张量形状
        self.assertEqual(sample["chosen_input_ids"].dim(), 1)
        self.assertEqual(sample["chosen_attention_mask"].dim(), 1)
        self.assertLessEqual(sample["chosen_input_ids"].size(0), self.max_length)
        
    def test_from_hf_dataset(self):
        """测试从HuggingFace数据集创建"""
        dataset = DPODataset.from_hf_dataset(
            dataset=self.mock_dataset,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        # 检查数据集大小
        self.assertEqual(len(dataset), len(self.mock_dataset))
        
        # 检查单个样本
        sample = dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("chosen_input_ids", sample)
        self.assertIn("rejected_input_ids", sample)
        
    def test_collate_fn(self):
        """测试批处理函数"""
        dataset = DPODataset(
            prompts=self.mock_dataset["prompt"][:4],  # 只使用4个样本
            chosen_responses=self.mock_dataset["chosen"][:4],
            rejected_responses=self.mock_dataset["rejected"][:4],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        
        # 创建一个batch
        batch = [dataset[i] for i in range(4)]
        collated = dataset.collate_fn(batch)
        
        # 检查collated batch的结构
        self.assertIn("chosen_input_ids", collated)
        self.assertIn("chosen_attention_mask", collated)
        self.assertIn("rejected_input_ids", collated)
        self.assertIn("rejected_attention_mask", collated)
        
        # 检查batch维度
        self.assertEqual(collated["chosen_input_ids"].size(0), 4)
        self.assertEqual(collated["rejected_input_ids"].size(0), 4)
        
        # 检查padding
        # 由于每个样本的长度可能不同，我们只需要确保attention_mask的和是一个有效值
        chosen_lengths = collated["chosen_attention_mask"].sum(dim=1)
        self.assertTrue(torch.all(chosen_lengths > 0))
        
    def test_input_validation(self):
        """测试输入验证"""
        # 测试长度不匹配的输入
        with self.assertRaises(AssertionError):
            DPODataset(
                prompts=self.mock_dataset["prompt"][:5],
                chosen_responses=self.mock_dataset["chosen"][:4],  # 长度不匹配
                rejected_responses=self.mock_dataset["rejected"][:5],
                tokenizer=self.tokenizer,
            )
            
    def test_max_length_truncation(self):
        """测试最大长度截断"""
        small_max_length = 32
        dataset = DPODataset(
            prompts=self.mock_dataset["prompt"],
            chosen_responses=self.mock_dataset["chosen"],
            rejected_responses=self.mock_dataset["rejected"],
            tokenizer=self.tokenizer,
            max_length=small_max_length,
        )
        
        # 检查所有样本是否被截断到指定长度
        sample = dataset[0]
        self.assertLessEqual(sample["chosen_input_ids"].size(0), small_max_length)
        self.assertLessEqual(sample["rejected_input_ids"].size(0), small_max_length)
        
    def test_empty_inputs(self):
        """测试空输入处理"""
        # 创建空输入
        empty_dataset = DPODataset(
            prompts=[""],
            chosen_responses=[""],
            rejected_responses=[""],
            tokenizer=self.tokenizer,
        )
        
        # 检查空输入的处理
        sample = empty_dataset[0]
        self.assertGreater(sample["chosen_input_ids"].size(0), 0)  # 应该至少包含特殊token
        self.assertGreater(sample["rejected_input_ids"].size(0), 0)

if __name__ == "__main__":
    unittest.main() 