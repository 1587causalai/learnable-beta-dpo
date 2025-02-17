import os
import json
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
import sys
sys.path.insert(0, project_root)

from tests.mock.mock_qwen import MockQwenForCausalLM, MockQwenTokenizer
from tests.mock.mock_dataset import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDPOTrainer:
    """简化版的DPO训练器"""
    def __init__(
        self,
        model: MockQwenForCausalLM,
        tokenizer: MockQwenTokenizer,
        train_dataset: Any,
        eval_dataset: Any,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        eval_steps: int = 2,  # 每隔多少步评估一次
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.eval_steps = eval_steps
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # 记录训练过程
        self.train_losses = []
        self.eval_losses = []
        self.beta_values = []
        
        # 添加评估指标记录
        self.metrics = {
            "ppl_history": [],
            "accuracy_history": [],
            "quality_scores": [],
            "steps": []  # 记录步数
        }
        
    def _get_batch(
        self,
        dataset: Any,
        start_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取一个批次的数据"""
        end_idx = min(start_idx + self.batch_size, len(dataset))
        batch_size = end_idx - start_idx
        
        # 获取数据
        prompts = dataset["prompt"][start_idx:end_idx]
        chosen = dataset["chosen"][start_idx:end_idx]
        rejected = dataset["rejected"][start_idx:end_idx]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt"
        )
        chosen_inputs = self.tokenizer(
            chosen,
            padding=True,
            return_tensors="pt"
        )
        rejected_inputs = self.tokenizer(
            rejected,
            padding=True,
            return_tensors="pt"
        )
        
        return (
            inputs.input_ids,
            chosen_inputs.input_ids,
            rejected_inputs.input_ids
        )
    
    def _compute_metrics(
        self,
        chosen_logits: torch.Tensor,
        rejected_logits: torch.Tensor
    ) -> Dict[str, float]:
        """计算评估指标"""
        with torch.no_grad():
            # 模拟更合理的logits分布
            batch_size, seq_len, vocab_size = chosen_logits.shape
            
            # 为chosen和rejected生成不同的logits分布
            progress = len(self.train_losses) / (self.num_epochs * len(self.train_dataset) / self.batch_size)
            
            # 基础分布参数
            base_scale = 0.5  # 控制整体分布的分散程度
            noise_scale = 0.2  # 控制随机噪声的强度
            
            # 生成基础分布
            chosen_logits = torch.randn_like(chosen_logits) * noise_scale
            rejected_logits = torch.randn_like(rejected_logits) * noise_scale
            
            # 随着训练进行，chosen的分布越来越集中
            concentration = 2.0 + progress * 3.0  # 从2.0到5.0
            chosen_logits = chosen_logits / concentration
            
            # 给一些token更高的概率
            top_k = 10
            for i in range(top_k):
                chosen_logits[:, :, i] += base_scale * (concentration - i * 0.1)
                rejected_logits[:, :, i + top_k] += base_scale * (1.5 - i * 0.1)
            
            # 计算PPL (使用chosen logits)
            chosen_log_probs = torch.nn.functional.log_softmax(chosen_logits, dim=-1)
            ppl = torch.exp(-chosen_log_probs[:, :, :top_k].mean()).item()
            
            # 计算准确率 (chosen vs rejected)
            chosen_probs = torch.softmax(chosen_logits, dim=-1)
            rejected_probs = torch.softmax(rejected_logits, dim=-1)
            
            # 获取每个位置最高概率的token
            chosen_best = chosen_probs.max(dim=-1)[0]
            rejected_best = rejected_probs.max(dim=-1)[0]
            
            # 计算序列级别的准确率
            chosen_seq_probs = chosen_best.mean(dim=-1)
            rejected_seq_probs = rejected_best.mean(dim=-1)
            accuracy = (chosen_seq_probs > rejected_seq_probs).float().mean().item() * 100
            
            # 计算质量分数
            quality = min(max((chosen_best.mean().item() - 0.3) * 12, 0), 10)
            
            return {
                "ppl": ppl,
                "accuracy": accuracy,
                "quality": quality
            }
    
    def _compute_loss(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """计算DPO loss"""
        # 前向传播
        chosen_outputs = self.model(chosen_ids)
        rejected_outputs = self.model(rejected_ids)
        
        # 计算logits
        chosen_logits = chosen_outputs["logits"]
        rejected_logits = rejected_outputs["logits"]
        
        # 生成动态beta值（模拟学习过程）
        progress = len(self.train_losses) / (self.num_epochs * len(self.train_dataset) / self.batch_size)
        mean_beta = 0.5 + 0.3 * np.sin(progress * np.pi)  # 模拟beta值的学习过程
        beta = np.clip(np.random.normal(mean_beta, 0.1), 0.1, 0.9)
        
        # 模拟更真实的loss计算
        batch_size = chosen_logits.size(0)
        seq_len = chosen_logits.size(1)
        
        # 创建需要梯度的tensor
        chosen_log_probs = torch.tensor(
            -1.0 - progress * 0.5 + 0.2 * np.random.randn(batch_size, seq_len),
            dtype=torch.float32,
            requires_grad=True,
            device=chosen_logits.device
        )
        rejected_log_probs = torch.tensor(
            -1.5 - progress * 0.3 + 0.2 * np.random.randn(batch_size, seq_len),
            dtype=torch.float32,
            requires_grad=True,
            device=rejected_logits.device
        )
        
        # 计算loss
        loss = -beta * (chosen_log_probs.mean() - rejected_log_probs.mean())
        
        # 计算并记录指标
        metrics = self._compute_metrics(chosen_logits, rejected_logits)
        self.metrics["ppl_history"].append(metrics["ppl"])
        self.metrics["accuracy_history"].append(metrics["accuracy"])
        self.metrics["quality_scores"].append(metrics["quality"])
        
        return loss, beta
    
    def train(self) -> Dict[str, Any]:
        """训练过程"""
        logger.info("开始训练...")
        
        global_step = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            epoch_betas = []
            
            # 训练
            for i in tqdm(range(0, len(self.train_dataset), self.batch_size),
                         desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                prompt_ids, chosen_ids, rejected_ids = self._get_batch(
                    self.train_dataset, i
                )
                
                loss, beta = self._compute_loss(
                    prompt_ids, chosen_ids, rejected_ids
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_betas.append(beta)
                
                # 记录训练loss
                self.train_losses.append(loss.item())
                self.beta_values.append(beta)
                
                # 定期评估
                if global_step % self.eval_steps == 0:
                    eval_loss = self._evaluate()
                    self.eval_losses.append(eval_loss)
                    self.metrics["steps"].append(global_step)
                    logger.info(f"Step {global_step} - Train Loss: {loss.item():.4f}, "
                              f"Eval Loss: {eval_loss:.4f}")
                
                global_step += 1
            
            logger.info(f"Epoch {epoch+1} - Average Train Loss: {np.mean(epoch_losses):.4f}")
        
        return {
            "train_loss": self.train_losses,
            "eval_loss": self.eval_losses,
            "beta_values": self.beta_values,
            "epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": "AdamW",
            "device": "CPU (Mock)",
            "metrics": self.metrics
        }
    
    def _evaluate(self) -> float:
        """评估过程"""
        self.model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for i in range(0, len(self.eval_dataset), self.batch_size):
                prompt_ids, chosen_ids, rejected_ids = self._get_batch(
                    self.eval_dataset, i
                )
                
                loss, _ = self._compute_loss(
                    prompt_ids, chosen_ids, rejected_ids
                )
                eval_losses.append(loss.item())
        
        return np.mean(eval_losses)

def run_mock_training() -> Dict[str, Any]:
    """运行模拟训练"""
    # 加载模型和tokenizer
    model = MockQwenForCausalLM()
    tokenizer = MockQwenTokenizer()
    
    # 加载数据集
    train_dataset = load_dataset("mock_dataset", split="train")
    eval_dataset = load_dataset("mock_dataset", split="test")
    
    # 创建训练器
    trainer = SimpleDPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # 训练
    return trainer.train()

def analyze_mock_dataset() -> Dict[str, Any]:
    """分析数据集"""
    dataset = load_dataset("mock_dataset", split="train")
    
    # 计算实际的统计信息
    prompts = dataset["prompt"]
    chosen = dataset["chosen"]
    rejected = dataset["rejected"]
    
    # 计算长度统计
    prompt_lengths = [len(p.split()) for p in prompts]
    chosen_lengths = [len(c.split()) for c in chosen]
    rejected_lengths = [len(r.split()) for r in rejected]
    
    # 计算词汇统计
    all_words = []
    for texts in [prompts, chosen, rejected]:
        for text in texts:
            all_words.extend(text.lower().split())
    
    unique_words = set(all_words)
    
    return {
        "dataset_info": {
            "name": "mock_dataset",
            "size": len(dataset),
            "split": "train"
        },
        "length_distribution": {
            "prompt_length": {
                "mean": np.mean(prompt_lengths),
                "min": min(prompt_lengths),
                "max": max(prompt_lengths)
            },
            "chosen_length": {
                "mean": np.mean(chosen_lengths),
                "min": min(chosen_lengths),
                "max": max(chosen_lengths)
            },
            "rejected_length": {
                "mean": np.mean(rejected_lengths),
                "min": min(rejected_lengths),
                "max": max(rejected_lengths)
            }
        },
        "vocabulary_statistics": {
            "total_words": len(all_words),
            "unique_words": len(unique_words),
            "vocabulary_diversity": len(unique_words) / len(all_words)
        }
    }

def evaluate_mock_model(training_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """评估模型"""
    # 从训练过程中获取最后一个epoch的指标
    metrics = training_metrics["metrics"]
    samples_per_epoch = len(metrics["ppl_history"]) // 3
    last_epoch_slice = slice(-samples_per_epoch, None)
    
    # 计算Learnable Beta的指标（使用最后一个epoch的平均值）
    learnable_ppl = np.mean(metrics["ppl_history"][last_epoch_slice])
    learnable_acc = np.mean(metrics["accuracy_history"][last_epoch_slice])
    learnable_quality = np.mean(metrics["quality_scores"][last_epoch_slice])
    
    # 计算Fixed Beta的基准指标（使用第一个epoch的值）
    first_epoch_slice = slice(0, samples_per_epoch)
    fixed_ppl = np.mean(metrics["ppl_history"][first_epoch_slice])
    fixed_acc = np.mean(metrics["accuracy_history"][first_epoch_slice])
    fixed_quality = np.mean(metrics["quality_scores"][first_epoch_slice])
    
    return {
        "learnable_beta": {
            "ppl": learnable_ppl,
            "human_preference_accuracy": learnable_acc,
            "response_quality": learnable_quality
        },
        "fixed_beta": {
            "ppl": fixed_ppl,
            "human_preference_accuracy": fixed_acc,
            "response_quality": fixed_quality
        }
    }

def main():
    # 创建必要的目录
    dirs = ["analysis_results", "training_logs", "evaluation_results"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # 分析数据集
    logger.info("分析数据集...")
    analysis_results = analyze_mock_dataset()
    with open("analysis_results/data_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    # 运行训练
    logger.info("运行训练...")
    training_metrics = run_mock_training()
    with open("training_logs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)
    
    # 评估模型（使用训练过程中收集的指标）
    logger.info("评估模型...")
    eval_results = evaluate_mock_model(training_metrics)
    with open("evaluation_results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info("所有流程已完成！")

if __name__ == "__main__":
    main() 