import os
import sys
import torch
import unittest
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from tests.mock.mock_qwen import MockQwenForCausalLM, MockQwenTokenizer
from tests.mock.mock_dataset import load_dataset
from src.models.beta_head import BetaHead
from src.models.dpo_model import DynamicBetaDPOModel
from src.data.dataset import DPODataset
from src.trainers.dpo_trainer import DPOTrainer

class TestFullPipeline(unittest.TestCase):
    """测试完整的训练和推理流程"""
    
    def setUp(self):
        """设置测试环境"""
        self.model = MockQwenForCausalLM()
        self.tokenizer = MockQwenTokenizer()
        self.beta_head = BetaHead(
            input_dim=self.model.config.hidden_size,
            head_type="linear",
        )
        
        # 创建DPO模型
        self.dpo_model = DynamicBetaDPOModel(
            base_model=self.model,
            beta_head=self.beta_head,
            tokenizer=self.tokenizer,
        )
        
        # 加载数据集
        dataset = load_dataset("mock_dataset")
        self.train_dataset = DPODataset(
            prompts=dataset["prompt"],
            chosen_responses=dataset["chosen"],
            rejected_responses=dataset["rejected"],
            tokenizer=self.tokenizer,
        )
        
        # 创建输出目录
        self.output_dir = "test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_training(self):
        """测试训练流程"""
        # 创建训练器
        trainer = DPOTrainer(
            model=self.dpo_model,
            train_dataset=self.train_dataset,
            batch_size=4,
            num_epochs=2,
            output_dir=self.output_dir,
            use_wandb=False,
        )
        
        # 运行训练
        trainer.train()
        
        # 检查模型文件是否保存
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "best_model")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "best_model", "beta_head.pt")))
    
    def test_inference(self):
        """测试推理流程"""
        self.dpo_model.eval()
        
        # 测试单个prompt
        prompt = "Test prompt"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # 计算beta值
        beta, ppl = self.dpo_model.get_dynamic_beta(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        # 生成回复
        outputs = self.dpo_model.base_model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=1,
        )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 检查输出
        self.assertIsInstance(beta, torch.Tensor)
        self.assertIsInstance(ppl, torch.Tensor)
        self.assertEqual(len(responses), 1)
    
    def test_batch_inference(self):
        """测试批量推理"""
        self.dpo_model.eval()
        
        # 测试多个prompt
        prompts = ["Test prompt 1", "Test prompt 2"]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        # 计算beta值
        beta, ppl = self.dpo_model.get_dynamic_beta(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        # 生成回复
        outputs = self.dpo_model.base_model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=2,
        )
        
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 检查输出
        self.assertEqual(beta.shape[0], len(prompts))
        self.assertEqual(ppl.shape[0], len(prompts))
        self.assertEqual(len(responses), len(prompts) * 2)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

if __name__ == "__main__":
    unittest.main() 