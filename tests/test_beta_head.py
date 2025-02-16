import unittest
import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from src.models.beta_head import BetaHead

class TestBetaHead(unittest.TestCase):
    """测试BetaHead网络"""
    
    def setUp(self):
        """设置测试环境"""
        self.input_dim = 768
        self.hidden_dim = 128
        self.epsilon = 0.1
        self.batch_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_linear_beta_head(self):
        """测试线性BetaHead"""
        beta_head = BetaHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            epsilon=self.epsilon,
            head_type="linear"
        ).to(self.device)
        
        # 创建测试输入
        context_embedding = torch.randn(
            self.batch_size,
            self.input_dim,
            device=self.device
        )
        ppl = torch.ones(self.batch_size, device=self.device) * 2.0
        
        # 计算beta值
        beta = beta_head(context_embedding, ppl)
        
        # 检查输出
        self.assertEqual(beta.shape, (self.batch_size,))
        self.assertTrue(torch.all(beta > 0))  # beta应该是正数
        
    def test_mlp_beta_head(self):
        """测试MLP BetaHead"""
        beta_head = BetaHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            epsilon=self.epsilon,
            head_type="mlp"
        ).to(self.device)
        
        # 创建测试输入
        context_embedding = torch.randn(
            self.batch_size,
            self.input_dim,
            device=self.device
        )
        ppl = torch.ones(self.batch_size, device=self.device) * 2.0
        
        # 计算beta值
        beta = beta_head(context_embedding, ppl)
        
        # 检查输出
        self.assertEqual(beta.shape, (self.batch_size,))
        self.assertTrue(torch.all(beta > 0))
        
    def test_epsilon_bounds(self):
        """测试f(x)的取值范围是否在[1-ε, 1+ε]内"""
        beta_head = BetaHead(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            epsilon=self.epsilon,
            head_type="linear"
        ).to(self.device)
        
        # 创建大量测试样本
        num_samples = 1000
        context_embedding = torch.randn(
            num_samples,
            self.input_dim,
            device=self.device
        )
        ppl = torch.ones(num_samples, device=self.device)
        
        # 计算beta值
        beta = beta_head(context_embedding, ppl)
        
        # 由于beta = w * ppl * f(x)，且ppl = 1，所以beta/w就是f(x)
        fx = beta / torch.abs(beta_head.w)
        
        # 检查f(x)的范围
        self.assertTrue(torch.all(fx >= 1 - self.epsilon))
        self.assertTrue(torch.all(fx <= 1 + self.epsilon))
        
    def test_invalid_head_type(self):
        """测试无效的head_type"""
        with self.assertRaises(ValueError):
            BetaHead(
                input_dim=self.input_dim,
                head_type="invalid_type"
            )
            
    def test_input_validation(self):
        """测试输入验证"""
        beta_head = BetaHead(
            input_dim=self.input_dim,
            head_type="linear"
        ).to(self.device)
        
        # 测试错误的输入维度
        wrong_embedding = torch.randn(self.batch_size, device=self.device)  # 缺少一个维度
        ppl = torch.ones(self.batch_size, device=self.device)
        
        with self.assertRaises(AssertionError):
            beta_head(wrong_embedding, ppl)
            
        # 测试不匹配的batch size
        wrong_ppl = torch.ones(self.batch_size + 1, device=self.device)  # batch size不匹配
        correct_embedding = torch.randn(
            self.batch_size,
            self.input_dim,
            device=self.device
        )
        
        with self.assertRaises(AssertionError):
            beta_head(correct_embedding, wrong_ppl)

if __name__ == "__main__":
    unittest.main() 