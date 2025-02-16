import os
import argparse
import torch
from pathlib import Path
import json
from typing import Dict, List, Optional
import logging
from datasets import load_dataset

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent)
import sys
sys.path.insert(0, project_root)

from src.data.dataset import DPODataset
from tests.mock.mock_qwen import MockQwenTokenizer
from tests.mock.mock_dataset import load_dataset as load_mock_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="分析DPO数据集")
    
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf",
                      help="数据集名称")
    parser.add_argument("--split", type=str, default="train",
                      help="数据集分片")
    parser.add_argument("--output_dir", type=str, default="analysis_results",
                      help="分析结果输出目录")
    parser.add_argument("--use_mock", action="store_true",
                      help="是否使用mock数据进行测试")
    
    return parser.parse_args()

def analyze_length_distribution(
    prompts: List[str],
    chosen_responses: List[str],
    rejected_responses: List[str]
) -> Dict:
    """分析文本长度分布"""
    prompt_lengths = [len(p.split()) for p in prompts]
    chosen_lengths = [len(r.split()) for r in chosen_responses]
    rejected_lengths = [len(r.split()) for r in rejected_responses]
    
    return {
        "prompt_length": {
            "mean": sum(prompt_lengths) / len(prompt_lengths),
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
        },
        "chosen_length": {
            "mean": sum(chosen_lengths) / len(chosen_lengths),
            "min": min(chosen_lengths),
            "max": max(chosen_lengths),
        },
        "rejected_length": {
            "mean": sum(rejected_lengths) / len(rejected_lengths),
            "min": min(rejected_lengths),
            "max": max(rejected_lengths),
        }
    }

def analyze_vocabulary(texts: List[str]) -> Dict:
    """分析词汇统计"""
    # 统计所有单词
    words = []
    for text in texts:
        words.extend(text.lower().split())
    
    unique_words = set(words)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_words[:100])  # 只保留前100个高频词
    
    return {
        "total_words": len(words),
        "unique_words": len(unique_words),
        "vocabulary_diversity": len(unique_words) / len(words),
        "top_100_words": top_words
    }

def analyze_response_differences(
    chosen_responses: List[str],
    rejected_responses: List[str]
) -> Dict:
    """分析chosen和rejected响应的差异"""
    # 计算长度差异
    length_diffs = [
        len(c.split()) - len(r.split())
        for c, r in zip(chosen_responses, rejected_responses)
    ]
    
    # 计算词汇重叠
    overlaps = []
    for c, r in zip(chosen_responses, rejected_responses):
        c_words = set(c.lower().split())
        r_words = set(r.lower().split())
        overlap = len(c_words & r_words) / len(c_words | r_words)
        overlaps.append(overlap)
    
    return {
        "length_difference": {
            "mean": sum(length_diffs) / len(length_diffs),
            "std": (sum((x - sum(length_diffs)/len(length_diffs))**2 for x in length_diffs) / len(length_diffs))**0.5,
        },
        "vocabulary_overlap": {
            "mean": sum(overlaps) / len(overlaps),
            "min": min(overlaps),
            "max": max(overlaps),
        }
    }

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    if args.use_mock:
        logger.info("使用mock数据集进行分析")
        dataset = load_mock_dataset("mock_dataset", split=args.split)
        tokenizer = MockQwenTokenizer()
    else:
        logger.info(f"加载数据集: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split=args.split)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1_5B", trust_remote_code=True)
    
    # 获取数据
    prompts = dataset["prompt"]
    chosen_responses = dataset["chosen"]
    rejected_responses = dataset["rejected"]
    
    # 进行分析
    logger.info("分析文本长度分布...")
    length_stats = analyze_length_distribution(
        prompts, chosen_responses, rejected_responses
    )
    
    logger.info("分析词汇统计...")
    vocab_stats = analyze_vocabulary(
        prompts + chosen_responses + rejected_responses
    )
    
    logger.info("分析响应差异...")
    diff_stats = analyze_response_differences(
        chosen_responses, rejected_responses
    )
    
    # 合并所有分析结果
    analysis_results = {
        "dataset_info": {
            "name": args.dataset_name,
            "split": args.split,
            "size": len(prompts),
        },
        "length_distribution": length_stats,
        "vocabulary_statistics": vocab_stats,
        "response_differences": diff_stats,
    }
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "data_analysis.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"分析结果已保存到: {output_file}")

if __name__ == "__main__":
    main() 