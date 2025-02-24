import os
import argparse
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
import sys
sys.path.insert(0, project_root)

from tests.mock.mock_qwen import MockQwenForCausalLM, MockQwenTokenizer
from tests.mock.mock_dataset import load_dataset
from src.models.beta_head import BetaHead
from src.models.dpo_model import DynamicBetaDPOModel
from src.data.dataset import DPODataset
from src.trainers.dpo_trainer import DPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="训练Learnable Beta DPO模型")
    
    # 模型参数
    parser.add_argument("--beta_head_type", type=str, default="linear",
                      choices=["linear", "mlp"], help="BetaHead网络类型")
    parser.add_argument("--hidden_dim", type=int, default=128,
                      help="BetaHead隐藏层维度（当使用MLP时）")
    parser.add_argument("--epsilon", type=float, default=0.1,
                      help="BetaHead的epsilon参数")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=8,
                      help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="学习率")
    parser.add_argument("--num_epochs", type=int, default=3,
                      help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="warmup步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="梯度裁剪阈值")
    
    # 其他参数
    parser.add_argument("--output_dir", type=str, default="outputs",
                      help="输出目录")
    parser.add_argument("--use_wandb", action="store_true",
                      help="是否使用wandb记录训练过程")
    parser.add_argument("--seed", type=int, default=42,
                      help="随机种子")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 加载mock模型和tokenizer
    model = MockQwenForCausalLM()
    tokenizer = MockQwenTokenizer()
    
    # 创建BetaHead
    beta_head = BetaHead(
        input_dim=model.config.hidden_size,
        hidden_dim=args.hidden_dim,
        epsilon=args.epsilon,
        head_type=args.beta_head_type,
    )
    
    # 创建DPO模型
    dpo_model = DynamicBetaDPOModel(
        base_model=model,
        beta_head=beta_head,
        tokenizer=tokenizer,
    )
    
    # 加载mock数据集
    train_dataset = load_dataset("mock_dataset", split="train")
    eval_dataset = load_dataset("mock_dataset", split="test")
    
    # 创建训练器
    trainer = DPOTrainer(
        model=dpo_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 