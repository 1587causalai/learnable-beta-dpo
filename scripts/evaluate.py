import os
import argparse
import torch
from pathlib import Path
import json
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
import sys
sys.path.insert(0, project_root)

from src.models.beta_head import BetaHead
from src.models.dpo_model import DynamicBetaDPOModel
from src.utils.metrics import MetricsCalculator
from tests.mock.mock_qwen import MockQwenForCausalLM, MockQwenTokenizer
from tests.mock.mock_dataset import load_dataset as load_mock_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="评估DPO模型")
    
    parser.add_argument("--model_dir", type=str, required=True,
                      help="模型目录，包含模型权重和beta_head.pt")
    parser.add_argument("--test_file", type=str,
                      help="测试数据文件路径，每行一个JSON格式的样本")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                      help="评估结果输出目录")
    parser.add_argument("--batch_size", type=int, default=4,
                      help="批处理大小")
    parser.add_argument("--use_mock", action="store_true",
                      help="是否使用mock模型和数据进行测试")
    
    return parser.parse_args()

def load_test_data(file_path: Optional[str] = None, use_mock: bool = False) -> List[Dict]:
    """加载测试数据"""
    if use_mock:
        logger.info("使用mock数据集")
        dataset = load_mock_dataset("mock_dataset", split="test")
        data = []
        for i in range(len(dataset["prompt"])):
            data.append({
                "prompt": dataset["prompt"][i],
                "chosen": dataset["chosen"][i],
                "rejected": dataset["rejected"][i],
            })
        return data
    
    if not file_path:
        raise ValueError("在非mock模式下必须提供测试文件路径")
        
    logger.info(f"从文件加载测试数据: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate_model(
    model: DynamicBetaDPOModel,
    tokenizer: MockQwenTokenizer,
    test_data: List[Dict],
    batch_size: int,
) -> Dict:
    """评估模型性能"""
    model.eval()
    metrics_calculator = MetricsCalculator(tokenizer)
    
    all_beta_chosen = []
    all_beta_rejected = []
    all_ppl_chosen = []
    all_ppl_rejected = []
    all_generated_texts = []
    all_reference_texts = []
    
    # 批处理评估
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i+batch_size]
        
        # 准备输入
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        
        # Tokenize
        chosen_inputs = tokenizer(
            [p + c for p, c in zip(prompts, chosen)],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        rejected_inputs = tokenizer(
            [p + r for p, r in zip(prompts, rejected)],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        # 计算beta值
        with torch.no_grad():
            beta_chosen, ppl_chosen = model.get_dynamic_beta(
                chosen_inputs.input_ids,
                chosen_inputs.attention_mask,
            )
            beta_rejected, ppl_rejected = model.get_dynamic_beta(
                rejected_inputs.input_ids,
                rejected_inputs.attention_mask,
            )
            
            # 生成文本
            prompt_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(model.device)
            
            generated = model.base_model.generate(
                **prompt_inputs,
                max_length=512,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # 收集结果
        all_beta_chosen.append(beta_chosen)
        all_beta_rejected.append(beta_rejected)
        all_ppl_chosen.append(ppl_chosen)
        all_ppl_rejected.append(ppl_rejected)
        all_generated_texts.extend(generated_texts)
        all_reference_texts.extend(chosen)  # 使用chosen作为参考
    
    # 合并所有结果
    beta_chosen = torch.cat(all_beta_chosen)
    beta_rejected = torch.cat(all_beta_rejected)
    ppl_chosen = torch.cat(all_ppl_chosen)
    ppl_rejected = torch.cat(all_ppl_rejected)
    
    # 计算指标
    dpo_metrics = metrics_calculator.calculate_dpo_metrics(
        beta_chosen=beta_chosen,
        beta_rejected=beta_rejected,
        ppl_chosen=ppl_chosen,
        ppl_rejected=ppl_rejected,
    )
    
    generation_metrics = metrics_calculator.calculate_generation_metrics(
        generated_texts=all_generated_texts,
        reference_texts=all_reference_texts,
    )
    
    # 格式化结果
    return metrics_calculator.format_metrics(
        dpo_metrics=dpo_metrics,
        generation_metrics=generation_metrics,
    )

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    if args.use_mock:
        logger.info("使用mock模型")
        base_model = MockQwenForCausalLM()
        tokenizer = MockQwenTokenizer()
    else:
        logger.info(f"加载模型: {args.model_dir}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    
    # 加载beta head
    beta_head = BetaHead(
        input_dim=base_model.config.hidden_size,
        head_type="linear",
    )
    beta_head_path = os.path.join(args.model_dir, "beta_head.pt")
    beta_head.load_state_dict(torch.load(beta_head_path))
    
    # 创建DPO模型
    model = DynamicBetaDPOModel(
        base_model=base_model,
        beta_head=beta_head,
        tokenizer=tokenizer,
    )
    
    # 加载测试数据
    test_data = load_test_data(args.test_file, args.use_mock)
    
    # 评估模型
    logger.info("开始评估...")
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        batch_size=args.batch_size,
    )
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存到: {output_file}")
    
    # 打印主要指标
    logger.info("\n主要评估指标:")
    logger.info(f"BLEU Score: {metrics['bleu_score']:.4f}")
    logger.info(f"ROUGE-1: {metrics['rouge_rouge1']:.4f}")
    logger.info(f"Beta Mean: {metrics['beta_mean']:.4f}")
    logger.info(f"PPL Mean: {metrics['ppl_mean']:.4f}")

if __name__ == "__main__":
    main() 