import unittest
import torch
import sys
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.utils.metrics import MetricsCalculator
from tests.mock.mock_qwen import MockQwenTokenizer

class TestMetricsCalculator(unittest.TestCase):
    """测试评估指标计算"""
    
    def setUp(self):
        """设置测试环境"""
        self.tokenizer = MockQwenTokenizer()
        self.metrics_calculator = MetricsCalculator(self.tokenizer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_dpo_metrics(self):
        """测试DPO相关指标计算"""
        # 创建测试数据
        batch_size = 100
        beta_chosen = torch.randn(batch_size, device=self.device) + 2  # 确保大部分是正数
        beta_rejected = torch.randn(batch_size, device=self.device) + 1
        ppl_chosen = torch.exp(torch.randn(batch_size, device=self.device))
        ppl_rejected = torch.exp(torch.randn(batch_size, device=self.device))
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_dpo_metrics(
            beta_chosen=beta_chosen,
            beta_rejected=beta_rejected,
            ppl_chosen=ppl_chosen,
            ppl_rejected=ppl_rejected,
        )
        
        # 检查指标
        self.assertIsInstance(metrics.beta_mean, float)
        self.assertIsInstance(metrics.beta_std, float)
        self.assertIsInstance(metrics.ppl_mean, float)
        self.assertIsInstance(metrics.ppl_std, float)
        self.assertIsInstance(metrics.chosen_rejected_diff, float)
        self.assertIsInstance(metrics.chosen_rejected_overlap, float)
        
        # 检查数值范围
        self.assertGreater(metrics.beta_mean, 0)
        self.assertGreaterEqual(metrics.beta_std, 0)
        self.assertGreater(metrics.ppl_mean, 0)
        self.assertGreaterEqual(metrics.ppl_std, 0)
        self.assertGreaterEqual(metrics.chosen_rejected_overlap, 0)
        self.assertLessEqual(metrics.chosen_rejected_overlap, 1)
        
    def test_generation_metrics(self):
        """测试生成质量相关指标计算"""
        # 创建测试数据
        generated_texts = [
            "This is a generated response.",
            "Another generated response with some repetition repetition.",
            "A third response that is completely different.",
        ]
        reference_texts = [
            "This is a reference response.",
            "A different reference response.",
            "The third reference response.",
        ]
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_generation_metrics(
            generated_texts=generated_texts,
            reference_texts=reference_texts,
        )
        
        # 检查指标
        self.assertIsInstance(metrics.bleu_score, float)
        self.assertIsInstance(metrics.rouge_scores, dict)
        self.assertIsInstance(metrics.response_length_mean, float)
        self.assertIsInstance(metrics.response_length_std, float)
        self.assertIsInstance(metrics.vocab_diversity, float)
        self.assertIsInstance(metrics.repetition_rate, float)
        
        # 检查数值范围
        self.assertGreaterEqual(metrics.bleu_score, 0)
        self.assertLessEqual(metrics.bleu_score, 1)
        self.assertGreaterEqual(metrics.vocab_diversity, 0)
        self.assertLessEqual(metrics.vocab_diversity, 1)
        self.assertGreaterEqual(metrics.repetition_rate, 0)
        self.assertLessEqual(metrics.repetition_rate, 1)
        
        # 检查ROUGE分数
        for rouge_type in ["rouge1", "rouge2", "rougeL"]:
            self.assertIn(rouge_type, metrics.rouge_scores)
            self.assertGreaterEqual(metrics.rouge_scores[rouge_type], 0)
            self.assertLessEqual(metrics.rouge_scores[rouge_type], 1)
            
    def test_distribution_overlap(self):
        """测试分布重叠度计算"""
        # 创建两个明显不同的分布
        dist1 = np.random.normal(0, 1, 1000)
        dist2 = np.random.normal(3, 1, 1000)
        
        overlap = self.metrics_calculator._calculate_distribution_overlap(dist1, dist2)
        
        # 检查重叠度
        self.assertGreaterEqual(overlap, 0)
        self.assertLessEqual(overlap, 1)
        self.assertLess(overlap, 0.5)  # 由于分布差异明显，重叠度应该较小
        
    def test_repetition_rate(self):
        """测试重复率计算"""
        # 创建测试文本
        texts = [
            "no repetition here",
            "some words words are repeated",
            "a lot lot lot of repetition repetition repetition",
        ]
        
        repetition_rate = self.metrics_calculator._calculate_repetition_rate(texts)
        
        # 检查重复率
        self.assertGreaterEqual(repetition_rate, 0)
        self.assertLessEqual(repetition_rate, 1)
        
    def test_format_metrics(self):
        """测试指标格式化"""
        # 创建测试数据
        dpo_metrics = self.metrics_calculator.calculate_dpo_metrics(
            beta_chosen=torch.randn(10) + 2,
            beta_rejected=torch.randn(10) + 1,
            ppl_chosen=torch.exp(torch.randn(10)),
            ppl_rejected=torch.exp(torch.randn(10)),
        )
        
        generation_metrics = self.metrics_calculator.calculate_generation_metrics(
            generated_texts=["Test text 1", "Test text 2"],
            reference_texts=["Ref text 1", "Ref text 2"],
        )
        
        # 格式化指标
        formatted_metrics = self.metrics_calculator.format_metrics(
            dpo_metrics=dpo_metrics,
            generation_metrics=generation_metrics,
        )
        
        # 检查格式化结果
        self.assertIsInstance(formatted_metrics, dict)
        self.assertIn("beta_mean", formatted_metrics)
        self.assertIn("bleu_score", formatted_metrics)
        self.assertIn("rouge_rouge1", formatted_metrics)
        
        # 检查所有值都是浮点数
        for value in formatted_metrics.values():
            self.assertIsInstance(value, float)

if __name__ == "__main__":
    unittest.main() 