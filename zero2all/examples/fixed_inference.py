#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完全固定的推理脚本，只使用特殊token进行测试
"""
import os
import sys
import torch
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer.tokenizer import ByteLevelBPETokenizer
from src.model.llm import QwenLLM

def main():
    """固定的测试推理"""
    # 设置路径
    model_path = "output/tiny_test_model"
    tokenizer_path = "tokenizer"
    
    print("加载分词器...")
    tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_path)
    
    print(f"分词器词表大小: {len(tokenizer.encoder) + len(tokenizer.special_tokens)}")
    print(f"特殊token: {tokenizer.special_tokens}")
    
    print("\n加载模型...")
    model = QwenLLM.from_pretrained(model_path)
    model.print_model_stats()
    
    # 使用特殊token作为起始输入
    bos_id = tokenizer.special_tokens["<bos>"]
    input_ids = torch.tensor([[bos_id]], device='cpu')
    
    print(f"\n使用起始token {bos_id} 开始生成...")
    
    # 手动实现生成逻辑
    max_length = 20
    temperature = 0.7
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        curr_ids = input_ids
        for i in range(max_length - 1):
            # 前向传播
            outputs = model(input_ids=curr_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 贪婪解码
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            print(f"步骤 {i+1}: 选择token {next_token.item()}")
            
            # 添加新token到序列
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # 如果生成了EOS，提前结束
            if next_token.item() == tokenizer.special_tokens.get("<eos>", -1):
                print(f"生成了EOS token，停止生成")
                break
    
    generation_time = time.time() - start_time
    
    # 手动解码token
    token_names = []
    for token_id in curr_ids[0].tolist():
        token_name = "<unknown>"
        for special_token, special_id in tokenizer.special_tokens.items():
            if token_id == special_id:
                token_name = special_token
                break
        token_names.append(f"{token_name}({token_id})")
    
    print(f"\n生成完成！用时: {generation_time:.2f}秒")
    print(f"生成的token序列: {curr_ids[0].tolist()}")
    print(f"解码后的token名称: {' '.join(token_names)}")

if __name__ == "__main__":
    main() 