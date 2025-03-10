#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
注意力机制对比示例脚本
展示标准多头注意力(MHSA)与分组查询注意力(GQA)的区别
"""
import os
import sys
import torch
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.attention import compare_mhsa_gqa
from src.model.llm import QwenLLM
from src.tokenizer.tokenizer import ByteLevelBPETokenizer

def create_mini_models():
    """创建微型模型用于对比不同注意力机制"""
    # 共享配置
    common_config = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "head_dim": 32,
        "intermediate_size": 512,
        "max_position_embeddings": 1024,
    }
    
    # 创建使用MHSA的模型
    mhsa_config = common_config.copy()
    mhsa_config.update({
        "num_q_heads": 4,
        "num_kv_heads": 4,  # 在MHSA中，num_kv_heads = num_q_heads
        "attention_type": "mhsa",
    })
    
    # 创建使用GQA的模型
    gqa_config = common_config.copy()
    gqa_config.update({
        "num_q_heads": 4,
        "num_kv_heads": 1,  # 在GQA中，num_kv_heads < num_q_heads
        "attention_type": "gqa",
    })
    
    # 实例化模型
    mhsa_model = QwenLLM(config=mhsa_config)
    gqa_model = QwenLLM(config=gqa_config)
    
    return mhsa_model, gqa_model

def profile_memory_usage(model, seq_lengths):
    """分析模型在不同序列长度下的内存使用"""
    results = []
    
    for seq_len in seq_lengths:
        # 生成随机输入
        input_ids = torch.randint(0, 100, (1, seq_len))
        
        # 记录内存使用前后变化
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
        
        # 前向传播
        with torch.no_grad():
            # 确保使用KV缓存
            model(input_ids=input_ids, use_cache=True)
        
        # 计算内存变化
        peak_mem = torch.cuda.max_memory_allocated()
        memory_usage = peak_mem - start_mem
        
        results.append(memory_usage / (1024 * 1024))  # 转换为MB
    
    return results

def profile_speed(model, seq_length, num_iterations=10):
    """分析模型在特定序列长度下的推理速度"""
    # 生成随机输入
    input_ids = torch.randint(0, 100, (1, seq_length))
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            model(input_ids=input_ids)
    
    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_ids=input_ids)
    
    total_time = time.time() - start_time
    return total_time / num_iterations  # 平均每次的时间

def main():
    """主函数"""
    # 首先打印注意力机制的理论比较
    compare_mhsa_gqa()
    
    print("\n创建用于测试的微型模型...")
    mhsa_model, gqa_model = create_mini_models()
    
    # 输出模型参数信息
    mhsa_params = sum(p.numel() for p in mhsa_model.parameters())
    gqa_params = sum(p.numel() for p in gqa_model.parameters())
    
    print(f"\n模型参数数量比较:")
    print(f"MHSA模型: {mhsa_params:,} 参数")
    print(f"GQA模型:  {gqa_params:,} 参数")
    print(f"参数减少: {mhsa_params - gqa_params:,} ({100*(mhsa_params-gqa_params)/mhsa_params:.2f}%)")
    
    # 如果有GPU，进行内存和速度测试
    if torch.cuda.is_available():
        print("\n移动模型到CUDA进行内存和速度测试...")
        mhsa_model.to("cuda")
        gqa_model.to("cuda")
        
        # 测试不同序列长度下的内存使用
        seq_lengths = [32, 64, 128, 256, 512, 1024]
        
        print("\n测试内存使用量...")
        mhsa_memory = profile_memory_usage(mhsa_model, seq_lengths)
        gqa_memory = profile_memory_usage(gqa_model, seq_lengths)
        
        print("\n序列长度vs内存使用量(MB):")
        print(f"序列长度 | MHSA内存 | GQA内存 | 内存节省(%)")
        print("-" * 50)
        for i, seq_len in enumerate(seq_lengths):
            saving = 100 * (mhsa_memory[i] - gqa_memory[i]) / mhsa_memory[i] if mhsa_memory[i] > 0 else 0
            print(f"{seq_len:9} | {mhsa_memory[i]:8.2f} | {gqa_memory[i]:7.2f} | {saving:8.2f}%")
        
        # 测试推理速度
        print("\n测试推理速度...")
        test_seq_len = 512
        mhsa_time = profile_speed(mhsa_model, test_seq_len)
        gqa_time = profile_speed(gqa_model, test_seq_len)
        
        print(f"\n序列长度 {test_seq_len} 的推理时间:")
        print(f"MHSA: {mhsa_time*1000:.2f} ms/batch")
        print(f"GQA:  {gqa_time*1000:.2f} ms/batch")
        speed_change = 100 * (mhsa_time - gqa_time) / mhsa_time
        print(f"速度提升: {speed_change:.2f}%")
    else:
        print("\n未检测到CUDA，跳过内存和速度测试。")
    
    print("\n总结:")
    print("1. GQA通过键值头共享，大幅减少模型参数量和内存占用")
    print("2. GQA使KV缓存规模与KV头数成比例，而不是查询头数")
    print("3. 在保持性能相近的情况下，GQA可以显著提高推理效率")
    print("4. 这就是为什么Qwen等现代高效LLM都采用GQA架构的原因")

if __name__ == "__main__":
    main() 