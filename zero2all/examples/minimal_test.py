#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化测试脚本，用于测试模型是否正常工作
"""
import os
import sys
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer.tokenizer import ByteLevelBPETokenizer
from src.model.llm import QwenLLM

# 设置模型和分词器路径
model_path = "output/tiny_test_model"
tokenizer_path = "tokenizer"

# 使用固定的输入ID进行测试
def test_fixed_input():
    print("加载分词器...")
    tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_path)
    
    print(f"分词器词表大小: {len(tokenizer.encoder) + len(tokenizer.special_tokens)}")
    print(f"特殊token: {tokenizer.special_tokens}")
    
    print("\n加载模型...")
    model = QwenLLM.from_pretrained(model_path)
    model.print_model_stats()
    
    # 使用一个确定在范围内的简单输入
    input_ids = torch.tensor([[0, 1, 2, 0, 1]]) # 使用特殊token ID
    
    print("\n使用固定输入ID进行前向传播...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    
    logits = outputs["logits"]
    print(f"输出形状: {logits.shape}")
    print("前向传播成功!")
    
    # 尝试手动调用生成函数
    print("\n尝试生成...")
    max_length = 10
    
    with torch.no_grad():
        # 简化的生成逻辑
        curr_ids = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            outputs = model(input_ids=curr_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
    
    print(f"生成的token IDs: {curr_ids[0].tolist()}")
    
    # 解码生成的token
    try:
        decoded = tokenizer.decode(curr_ids[0].tolist())
        print(f"解码结果: {decoded}")
    except Exception as e:
        print(f"解码出错: {e}")

if __name__ == "__main__":
    test_fixed_input() 