import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_scheduler
from typing import Dict, Optional, List
from tqdm import tqdm
import wandb
import logging

from ..models.dpo_model import DynamicBetaDPOModel
from ..data.dataset import DPODataset
from ..utils.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DPOTrainer:
    """DPO训练器
    
    实现了完整的训练循环，包括训练、评估和模型保存
    """
    
    def __init__(
        self,
        model: DynamicBetaDPOModel,
        train_dataset: DPODataset,
        eval_dataset: Optional[DPODataset] = None,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "outputs",
        use_wandb: bool = True,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=train_dataset.collate_fn,
        )
        
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=eval_dataset.collate_fn,
            )
        
        # 设置优化器和学习率调度器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
        )
        
        num_training_steps = len(self.train_dataloader) * num_epochs
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        
    def train(self):
        """执行完整的训练循环"""
        if self.use_wandb:
            wandb.init(project="learnable-beta-dpo")
            
        global_step = 0
        best_eval_loss = float("inf")
        
        # 创建best_model目录
        best_model_dir = os.path.join(self.output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        
        # 保存初始模型作为最佳模型
        self.save_model("best_model")
        
        for epoch in range(self.num_epochs):
            logger.info(f"开始 Epoch {epoch + 1}/{self.num_epochs}")
            
            # 训练一个epoch
            self.model.train()
            total_train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            for step, batch in enumerate(progress_bar):
                # 将数据移到设备上
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # 如果使用梯度累积，需要除以累积步数
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                total_train_loss += loss.item()
                
                # 梯度累积
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    train_steps += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        "loss": total_train_loss / train_steps,
                        "beta_chosen": outputs["beta_chosen"].item(),
                        "beta_rejected": outputs["beta_rejected"].item(),
                    })
                    
                    # 记录到wandb
                    if self.use_wandb:
                        wandb.log({
                            "train/loss": loss.item() * self.gradient_accumulation_steps,
                            "train/beta_chosen": outputs["beta_chosen"].item(),
                            "train/beta_rejected": outputs["beta_rejected"].item(),
                            "train/ppl_chosen": outputs["ppl_chosen"].item(),
                            "train/ppl_rejected": outputs["ppl_rejected"].item(),
                            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        }, step=global_step)
            
            # 计算平均训练loss
            avg_train_loss = total_train_loss / train_steps
            logger.info(f"Epoch {epoch + 1} 平均训练loss: {avg_train_loss:.4f}")
            
            # 评估
            if self.eval_dataset:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1} 评估结果: {eval_metrics}")
                
                # 保存最佳模型
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    self.save_model("best_model")
                    logger.info(f"发现更好的模型，已保存到 {best_model_dir}")
            else:
                # 如果没有评估数据集，就把当前模型保存为最佳模型
                self.save_model("best_model")
                    
            # 保存checkpoint
            self.save_model(f"checkpoint-epoch-{epoch + 1}")
            
        if self.use_wandb:
            wandb.finish()
            
    def evaluate(self) -> Dict[str, float]:
        """评估模型
        
        Returns:
            包含评估指标的字典
        """
        self.model.eval()
        metrics_calculator = MetricsCalculator(self.tokenizer)
        
        # 用于收集评估数据
        all_beta_chosen = []
        all_beta_rejected = []
        all_ppl_chosen = []
        all_ppl_rejected = []
        all_generated_texts = []
        all_chosen_texts = []  # 用作参考文本
        total_eval_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # 将数据移到设备上
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 计算loss和其他指标
                outputs = self.model(**batch)
                
                # 收集beta值和困惑度
                all_beta_chosen.append(outputs["beta_chosen"])
                all_beta_rejected.append(outputs["beta_rejected"])
                all_ppl_chosen.append(outputs["ppl_chosen"])
                all_ppl_rejected.append(outputs["ppl_rejected"])
                
                # 生成文本用于计算生成质量指标
                gen_outputs = self.model.base_model.generate(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"],
                    max_length=self.model.max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                )
                
                # 解码生成的文本和参考文本
                generated_texts = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
                chosen_texts = self.tokenizer.batch_decode(
                    batch["chosen_input_ids"],
                    skip_special_tokens=True
                )
                
                all_generated_texts.extend(generated_texts)
                all_chosen_texts.extend(chosen_texts)
                
                total_eval_loss += outputs["loss"].item()
                total_steps += 1
        
        # 计算DPO相关指标
        dpo_metrics = metrics_calculator.calculate_dpo_metrics(
            beta_chosen=torch.cat(all_beta_chosen),
            beta_rejected=torch.cat(all_beta_rejected),
            ppl_chosen=torch.cat(all_ppl_chosen),
            ppl_rejected=torch.cat(all_ppl_rejected),
        )
        
        # 计算生成质量相关指标
        generation_metrics = metrics_calculator.calculate_generation_metrics(
            generated_texts=all_generated_texts,
            reference_texts=all_chosen_texts,
        )
        
        # 合并所有指标
        all_metrics = metrics_calculator.format_metrics(
            dpo_metrics=dpo_metrics,
            generation_metrics=generation_metrics,
        )
        
        # 添加平均loss
        all_metrics["eval_loss"] = total_eval_loss / total_steps
        
        # 记录到wandb
        if self.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in all_metrics.items()})
            
        return all_metrics
        
    def save_model(self, output_dir: str):
        """保存模型
        
        Args:
            output_dir: 保存目录名
        """
        output_dir = os.path.join(self.output_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        self.model.base_model.save_pretrained(output_dir)
        
        # 保存beta head
        beta_head_path = os.path.join(output_dir, "beta_head.pt")
        torch.save(self.model.beta_head.state_dict(), beta_head_path)
        
        logger.info(f"模型保存到 {output_dir}")
