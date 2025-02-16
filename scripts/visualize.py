import os
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="可视化分析结果")
    
    parser.add_argument("--data_analysis", type=str,
                      help="数据分析结果文件路径")
    parser.add_argument("--eval_results", type=str,
                      help="评估结果文件路径")
    parser.add_argument("--output_dir", type=str, default="visualization_results",
                      help="可视化结果输出目录")
    
    return parser.parse_args()

def plot_length_distribution(data: Dict, output_dir: str):
    """绘制长度分布图"""
    plt.figure(figsize=(10, 6))
    
    categories = ["prompt", "chosen", "rejected"]
    means = [data[f"{c}_length"]["mean"] for c in categories]
    mins = [data[f"{c}_length"]["min"] for c in categories]
    maxs = [data[f"{c}_length"]["max"] for c in categories]
    
    x = range(len(categories))
    plt.bar(x, means, yerr=np.array([means - np.array(mins), np.array(maxs) - means]),
           capsize=5, alpha=0.7)
    
    plt.xticks(x, [c.capitalize() for c in categories])
    plt.ylabel("Token Length")
    plt.title("Text Length Distribution")
    
    plt.savefig(os.path.join(output_dir, "length_distribution.png"))
    plt.close()

def plot_vocabulary_stats(data: Dict, output_dir: str):
    """绘制词汇统计图"""
    # 词频分布
    plt.figure(figsize=(12, 6))
    
    words = list(data["top_100_words"].keys())[:20]  # 只显示前20个词
    freqs = list(data["top_100_words"].values())[:20]
    
    plt.bar(range(len(words)), freqs, alpha=0.7)
    plt.xticks(range(len(words)), words, rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.title("Top 20 Words Frequency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "word_frequency.png"))
    plt.close()
    
    # 词汇多样性
    plt.figure(figsize=(8, 6))
    plt.pie([data["unique_words"], data["total_words"] - data["unique_words"]],
            labels=["Unique Words", "Repeated Words"],
            autopct="%1.1f%%",
            colors=sns.color_palette("pastel"))
    plt.title("Vocabulary Diversity")
    
    plt.savefig(os.path.join(output_dir, "vocabulary_diversity.png"))
    plt.close()

def plot_response_differences(data: Dict, output_dir: str):
    """绘制响应差异图"""
    plt.figure(figsize=(10, 6))
    
    # 长度差异分布
    mean = data["length_difference"]["mean"]
    std = data["length_difference"]["std"]
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    y = np.exp(-(x - mean)**2 / (2*std**2)) / (std * np.sqrt(2*np.pi))
    
    plt.plot(x, y)
    plt.axvline(mean, color="r", linestyle="--", label=f"Mean: {mean:.2f}")
    plt.fill_between(x, y, alpha=0.3)
    
    plt.xlabel("Length Difference (Chosen - Rejected)")
    plt.ylabel("Density")
    plt.title("Response Length Difference Distribution")
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "length_difference.png"))
    plt.close()

def plot_evaluation_metrics(data: Dict, output_dir: str):
    """绘制评估指标图"""
    # Beta值分布
    plt.figure(figsize=(10, 6))
    
    metrics = ["beta_mean", "beta_std", "beta_min", "beta_max"]
    values = [data[m] for m in metrics]
    
    plt.bar(range(len(metrics)), values, alpha=0.7)
    plt.xticks(range(len(metrics)), [m.replace("beta_", "").capitalize() for m in metrics])
    plt.ylabel("Value")
    plt.title("Beta Value Statistics")
    
    plt.savefig(os.path.join(output_dir, "beta_stats.png"))
    plt.close()
    
    # 生成质量指标
    plt.figure(figsize=(10, 6))
    
    metrics = ["bleu_score", "rouge_rouge1", "rouge_rouge2", "rouge_rougeL"]
    values = [data[m] for m in metrics]
    
    plt.bar(range(len(metrics)), values, alpha=0.7)
    plt.xticks(range(len(metrics)), ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L"], rotation=45)
    plt.ylabel("Score")
    plt.title("Generation Quality Metrics")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generation_metrics.png"))
    plt.close()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 可视化数据分析结果
    if args.data_analysis:
        logger.info("可视化数据分析结果...")
        with open(args.data_analysis, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        plot_length_distribution(data["length_distribution"], args.output_dir)
        plot_vocabulary_stats(data["vocabulary_statistics"], args.output_dir)
        plot_response_differences(data["response_differences"], args.output_dir)
    
    # 可视化评估结果
    if args.eval_results:
        logger.info("可视化评估结果...")
        with open(args.eval_results, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        plot_evaluation_metrics(data, args.output_dir)
    
    logger.info(f"可视化结果已保存到: {args.output_dir}")

if __name__ == "__main__":
    main() 