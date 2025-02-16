import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple
import json
import logging

from src.models.beta_head import BetaHead
from src.models.dpo_model import DynamicBetaDPOModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的Learnable Beta DPO模型进行推理")
    
    parser.add_argument("--model_dir", type=str, required=True,
                      help="训练好的模型目录，应包含model和beta_head.pt")
    parser.add_argument("--max_length", type=int, default=512,
                      help="最大序列长度")
    parser.add_argument("--num_return_sequences", type=int, default=1,
                      help="每个prompt生成的回复数量")
    parser.add_argument("--temperature", type=float, default=0.7,
                      help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                      help="nucleus sampling的概率阈值")
    parser.add_argument("--input_file", type=str,
                      help="输入文件路径，每行一个prompt")
    parser.add_argument("--output_file", type=str,
                      help="输出文件路径")
    
    return parser.parse_args()

def load_model(model_dir: str, max_length: int) -> Tuple[DynamicBetaDPOModel, AutoTokenizer]:
    """加载模型和tokenizer
    
    Args:
        model_dir: 模型目录
        max_length: 最大序列长度
        
    Returns:
        model: 加载的模型
        tokenizer: 加载的tokenizer
    """
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # 加载beta head
    beta_head_path = os.path.join(model_dir, "beta_head.pt")
    if not os.path.exists(beta_head_path):
        raise ValueError(f"Beta head weights not found at {beta_head_path}")
    
    beta_head = BetaHead(
        input_dim=model.config.hidden_size,
        head_type="linear",  # 默认使用linear
    )
    beta_head.load_state_dict(torch.load(beta_head_path))
    
    # 创建DPO模型
    dpo_model = DynamicBetaDPOModel(
        base_model=model,
        beta_head=beta_head,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    return dpo_model, tokenizer

def generate_response(
    model: DynamicBetaDPOModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 512,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[str], float]:
    """生成回复
    
    Args:
        model: DPO模型
        tokenizer: tokenizer
        prompt: 输入prompt
        max_length: 最大序列长度
        num_return_sequences: 生成的回复数量
        temperature: 采样温度
        top_p: nucleus sampling的概率阈值
        
    Returns:
        responses: 生成的回复列表
        beta: 计算的beta值
    """
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    
    # 计算beta值
    beta, ppl = model.get_dynamic_beta(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
    )
    
    # 生成回复
    outputs = model.base_model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
    )
    
    # 解码输出
    responses = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )
    
    return responses, beta.item()

def interactive_mode(model: DynamicBetaDPOModel, tokenizer: AutoTokenizer, args):
    """交互式推理模式"""
    logger.info("进入交互模式。输入 'q' 退出。")
    
    while True:
        prompt = input("\n请输入prompt: ")
        if prompt.lower() == 'q':
            break
            
        responses, beta = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        print(f"\n计算的beta值: {beta:.4f}")
        for i, response in enumerate(responses, 1):
            print(f"\n回复 {i}:")
            print(response)

def batch_mode(model: DynamicBetaDPOModel, tokenizer: AutoTokenizer, args):
    """批处理推理模式"""
    # 读取输入文件
    with open(args.input_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    results = []
    for prompt in prompts:
        responses, beta = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=args.max_length,
            num_return_sequences=args.num_return_sequences,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        results.append({
            "prompt": prompt,
            "responses": responses,
            "beta": beta,
        })
    
    # 保存结果
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到 {args.output_file}")

def main():
    args = parse_args()
    
    # 加载模型和tokenizer
    model, tokenizer = load_model(args.model_dir, args.max_length)
    model.eval()
    
    # 根据是否提供输入文件选择模式
    if args.input_file and args.output_file:
        batch_mode(model, tokenizer, args)
    else:
        interactive_mode(model, tokenizer, args)

if __name__ == "__main__":
    main() 