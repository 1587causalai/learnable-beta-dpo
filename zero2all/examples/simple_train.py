#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.llm import QwenLLM
from src.tokenizer.tokenizer import ByteLevelBPETokenizer


class TextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # 获取词表大小用于验证
        self.vocab_size = len(tokenizer.encoder) + len(tokenizer.special_tokens)
        print(f"数据集使用词表大小: {self.vocab_size}")
        
        # 加载数据
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 分割成多个段落
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # 编码每个段落
        for paragraph in paragraphs:
            tokenized = tokenizer(paragraph, truncation=True, max_length=max_length+1)
            if len(tokenized['input_ids'][0]) > 2:  # 确保有足够的token
                # 验证所有token ID在词表范围内
                valid_tokens = []
                for token_id in tokenized['input_ids'][0]:
                    if 0 <= token_id < self.vocab_size:
                        valid_tokens.append(token_id)
                    else:
                        # 对于超出范围的token，使用<unk>或第一个特殊token
                        valid_tokens.append(0)  # 假设0是PAD或UNK
                
                if len(valid_tokens) > 2:  # 再次检查有效长度
                    self.examples.append(valid_tokens)
        
        print(f"加载了 {len(self.examples)} 个有效文本段落")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # 创建输入和标签（预测下一个token）
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        # 如果长度不够，进行填充
        if len(input_ids) < self.max_length:
            pad_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.special_tokens["<pad>"]] * pad_length
            labels = labels + [-100] * pad_length  # -100表示忽略这些位置的损失
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # 将数据移到设备上
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"]
        
        # 计算损失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1), 
            ignore_index=-100
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载或创建分词器
    if os.path.exists(args.tokenizer_path):
        print(f"加载分词器从 {args.tokenizer_path}")
        tokenizer = ByteLevelBPETokenizer.from_pretrained(args.tokenizer_path)
    else:
        print("创建并训练新的分词器")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            [args.data_file],
            vocab_size=args.vocab_size,
            min_frequency=2
        )
        tokenizer.save_pretrained(args.tokenizer_path)
    
    # 创建数据集和数据加载器
    dataset = TextDataset(args.data_file, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # 创建或加载模型
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载模型从 {args.model_path}")
        model = QwenLLM.from_pretrained(args.model_path)
    else:
        print("创建新模型")
        # 获取实际的词表大小
        vocab_size = len(tokenizer.encoder) + len(tokenizer.special_tokens)
        print(f"分词器词表大小: {vocab_size}")
        
        config = {
            "vocab_size": vocab_size,  # 使用实际的词表大小
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_layers,
            "num_q_heads": args.num_q_heads,
            "num_kv_heads": args.num_kv_heads,
            "head_dim": args.head_dim,
            "intermediate_size": args.intermediate_size,
            "max_position_embeddings": args.max_length,
        }
        model = QwenLLM(config=config)
    
    model.to(device)
    model.print_model_stats()
    
    # 创建优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(dataloader), 
        eta_min=args.learning_rate / 10
    )
    
    # 训练循环
    for epoch in range(args.epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{args.epochs}, 平均损失: {avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        print(f"保存检查点到 {checkpoint_path}")
    
    # 保存最终模型
    model.save_pretrained(args.output_dir)
    print(f"保存最终模型到 {args.output_dir}")
    
    # 生成一些示例文本
    if args.prompt:
        print("\n生成示例文本:")
        model.eval()
        model.sample_and_print(args.prompt, tokenizer, max_length=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenLLM 简单训练脚本")
    
    # 数据和模型路径
    parser.add_argument("--data_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer", help="分词器目录")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径（可选）")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    
    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=151646, help="词表大小")
    parser.add_argument("--hidden_size", type=int, default=896, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=24, help="Transformer层数")
    parser.add_argument("--num_q_heads", type=int, default=14, help="查询头数量")
    parser.add_argument("--num_kv_heads", type=int, default=2, help="键值头数量")
    parser.add_argument("--head_dim", type=int, default=64, help="注意力头维度")
    parser.add_argument("--intermediate_size", type=int, default=4864, help="中间层大小")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    
    # 生成参数
    parser.add_argument("--prompt", type=str, default=None, help="生成示例文本的提示（可选）")
    
    args = parser.parse_args()
    main(args) 