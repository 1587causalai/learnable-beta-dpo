#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.llm import QwenLLM
from src.tokenizer.tokenizer import ByteLevelBPETokenizer


def main(args):
    # 加载分词器
    if not os.path.exists(args.tokenizer_path):
        raise ValueError(f"分词器路径 {args.tokenizer_path} 不存在")
    
    print(f"加载分词器从 {args.tokenizer_path}")
    tokenizer = ByteLevelBPETokenizer.from_pretrained(args.tokenizer_path)
    
    # 加载模型
    if not os.path.exists(args.model_path):
        raise ValueError(f"模型路径 {args.model_path} 不存在")
    
    print(f"加载模型从 {args.model_path}")
    model = QwenLLM.from_pretrained(args.model_path)
    
    # 移动到指定设备
    device = torch.device(args.device)
    model.to(device)
    
    # 打印模型统计信息
    model.print_model_stats()
    
    # 设置为评估模式
    model.eval()
    
    # 如果提供了输入文件
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # 使用提供的提示或默认提示
        prompts = [args.prompt or "这是一个测试，请模型继续生成文本："]
    
    # 对每个提示生成文本
    for i, prompt in enumerate(prompts):
        print(f"\n=== 生成 #{i+1} ===")
        generated_text = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=not args.no_sample,
        )
        
        # 如果提供了输出文件，保存结果
        if args.output_file:
            with open(args.output_file, 'a', encoding='utf-8') as f:
                f.write(f"=== 提示 #{i+1} ===\n")
                f.write(f"{prompt}\n\n")
                f.write(f"=== 生成 #{i+1} ===\n")
                f.write(f"{generated_text}\n\n")
                f.write("-" * 50 + "\n\n")


def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_length=100, 
    temperature=0.7, 
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
):
    """生成文本的辅助函数"""
    # 获取分词器词表大小和特殊token
    vocab_size = len(tokenizer.encoder) + len(tokenizer.special_tokens)
    print(f"分词器词表大小: {vocab_size}")
    print(f"特殊token: {tokenizer.special_tokens}")
    
    # 编码提示 - 注意：如果分词器词表太小，这可能不会产生有用的编码
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.transformer.embedding.weight.device)
        print(f"输入token IDs: {input_ids[0].tolist()}")
    except Exception as e:
        print(f"分词器编码错误: {e}，使用固定特殊token作为输入...")
        # 使用特殊token作为备用输入
        input_ids = torch.tensor([[tokenizer.special_tokens["<bos>"]]], device=model.transformer.embedding.weight.device)
    
    # 记录起始时间
    import time
    start_time = time.time()
    
    # 使用简化的生成方法(避免使用模型的generate方法)
    with torch.no_grad():
        # 手动实现生成逻辑
        curr_ids = input_ids
        for _ in range(max_length - input_ids.shape[1]):
            # 前向传播
            outputs = model(input_ids=curr_ids)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 实现top-p采样
            if do_sample and top_p < 1.0:
                import torch.nn.functional as F
                # 计算概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # 移除概率累积和超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 创建掩码
                indices_to_remove = sorted_indices[sorted_indices_to_remove].tolist()
                next_token_logits[0, indices_to_remove] = float('-inf')
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪搜索
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # 添加新token
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # 如果生成了结束符，可以提前终止
            if next_token.item() == tokenizer.special_tokens.get("<eos>", -1):
                break
    
    # 计算生成时间
    generation_time = time.time() - start_time
    
    # 解码生成的文本
    try:
        generated_text = tokenizer.decode(curr_ids[0].tolist())
    except Exception as e:
        print(f"解码错误: {e}，尝试手动解码...")
        # 手动解码token ID
        tokens = []
        for token_id in curr_ids[0].tolist():
            for special_token, special_id in tokenizer.special_tokens.items():
                if token_id == special_id:
                    tokens.append(special_token)
                    break
            else:
                tokens.append(f"<ID_{token_id}>")
        generated_text = " ".join(tokens)
    
    # 计算生成的token数量和速度
    num_new_tokens = curr_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = num_new_tokens / generation_time if generation_time > 0 else 0
    
    # 打印生成信息
    print(f"输入:\n{prompt}")
    print(f"\n输出:\n{generated_text}")
    print(f"\n生成统计:")
    print(f"  温度: {temperature}, Top-p: {top_p}")
    print(f"  新生成的token数量: {num_new_tokens}")
    print(f"  生成时间: {generation_time:.2f}秒")
    print(f"  生成速度: {tokens_per_second:.2f} tokens/秒")
    print(f"  生成的token IDs: {curr_ids[0].tolist()}")
    
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenLLM 简单推理脚本")
    
    # 模型和分词器路径
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="分词器路径")
    
    # 输入/输出
    parser.add_argument("--prompt", type=str, default=None, help="生成文本的提示")
    parser.add_argument("--input_file", type=str, default=None, help="包含多个提示的输入文件")
    parser.add_argument("--output_file", type=str, default=None, help="保存生成文本的输出文件")
    
    # 生成参数
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚")
    parser.add_argument("--no_sample", action="store_true", help="使用贪婪解码而不是采样")
    
    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    
    args = parser.parse_args()
    
    # 确保提供了prompt或input_file
    if args.prompt is None and args.input_file is None:
        parser.error("必须提供--prompt或--input_file")
    
    main(args) 