import torch
import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

@dataclass
class DPOMetrics:
    """DPO训练相关的评估指标"""
    beta_mean: float
    beta_std: float
    beta_min: float
    beta_max: float
    ppl_mean: float
    ppl_std: float
    chosen_rejected_diff: float
    chosen_rejected_overlap: float

@dataclass
class GenerationMetrics:
    """生成质量相关的评估指标"""
    bleu_score: float
    rouge_scores: Dict[str, float]
    response_length_mean: float
    response_length_std: float
    vocab_diversity: float
    repetition_rate: float

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    def calculate_dpo_metrics(
        self,
        beta_chosen: torch.Tensor,
        beta_rejected: torch.Tensor,
        ppl_chosen: torch.Tensor,
        ppl_rejected: torch.Tensor,
    ) -> DPOMetrics:
        """计算DPO相关的评估指标
        
        Args:
            beta_chosen: chosen样本的beta值 [batch_size]
            beta_rejected: rejected样本的beta值 [batch_size]
            ppl_chosen: chosen样本的困惑度 [batch_size]
            ppl_rejected: rejected样本的困惑度 [batch_size]
            
        Returns:
            DPO相关的评估指标
        """
        # 将tensor转换为numpy数组
        beta_chosen = beta_chosen.detach().cpu().numpy()
        beta_rejected = beta_rejected.detach().cpu().numpy()
        ppl_chosen = ppl_chosen.detach().cpu().numpy()
        ppl_rejected = ppl_rejected.detach().cpu().numpy()
        
        # 计算beta值的统计信息
        all_betas = np.concatenate([beta_chosen, beta_rejected])
        beta_mean = np.mean(all_betas)
        beta_std = np.std(all_betas)
        beta_min = np.min(all_betas)
        beta_max = np.max(all_betas)
        
        # 计算困惑度的统计信息
        ppl_mean = np.mean(np.concatenate([ppl_chosen, ppl_rejected]))
        ppl_std = np.std(np.concatenate([ppl_chosen, ppl_rejected]))
        
        # 计算chosen和rejected之间的差异
        chosen_rejected_diff = np.mean(beta_chosen) - np.mean(beta_rejected)
        
        # 计算chosen和rejected的分布重叠程度
        chosen_rejected_overlap = self._calculate_distribution_overlap(
            beta_chosen, beta_rejected
        )
        
        return DPOMetrics(
            beta_mean=float(beta_mean),
            beta_std=float(beta_std),
            beta_min=float(beta_min),
            beta_max=float(beta_max),
            ppl_mean=float(ppl_mean),
            ppl_std=float(ppl_std),
            chosen_rejected_diff=float(chosen_rejected_diff),
            chosen_rejected_overlap=float(chosen_rejected_overlap),
        )
    
    def calculate_generation_metrics(
        self,
        generated_texts: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> GenerationMetrics:
        """计算生成质量相关指标计算
        
        Args:
            generated_texts: 生成的文本列表
            reference_texts: 参考文本列表（如果有）
            
        Returns:
            生成质量相关的评估指标
        """
        # 计算BLEU和ROUGE分数（如果有参考文本）
        if reference_texts:
            bleu_scores = []
            rouge_scores = defaultdict(list)
            
            for gen, ref in zip(generated_texts, reference_texts):
                # 简单的分词（按空格分割）
                gen_tokens = gen.lower().split()
                ref_tokens = ref.lower().split()
                
                # 确保有足够的token计算BLEU
                if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoothing)
                    bleu_scores.append(bleu)
                
                # 计算ROUGE
                rouge_score = self.rouge_scorer.score(ref, gen)
                for key, value in rouge_score.items():
                    rouge_scores[key].append(value.fmeasure)
            
            bleu_score = np.mean(bleu_scores) if bleu_scores else 0.0
            rouge_dict = {k.replace("-", "_"): np.mean(v) for k, v in rouge_scores.items()}
        else:
            bleu_score = 0.0
            rouge_dict = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        # 计算响应长度统计
        response_lengths = [len(text.split()) for text in generated_texts]
        response_length_mean = np.mean(response_lengths)
        response_length_std = np.std(response_lengths)
        
        # 计算词汇多样性
        all_tokens = []
        for text in generated_texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        vocab_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # 计算重复率
        repetition_rate = self._calculate_repetition_rate(generated_texts)
        
        return GenerationMetrics(
            bleu_score=float(bleu_score),
            rouge_scores=rouge_dict,
            response_length_mean=float(response_length_mean),
            response_length_std=float(response_length_std),
            vocab_diversity=float(vocab_diversity),
            repetition_rate=float(repetition_rate),
        )
    
    def _calculate_distribution_overlap(
        self,
        dist1: np.ndarray,
        dist2: np.ndarray,
        num_bins: int = 50,
    ) -> float:
        """计算两个分布的重叠程度
        
        使用直方图方法计算两个分布的重叠面积
        """
        min_val = min(dist1.min(), dist2.min())
        max_val = max(dist1.max(), dist2.max())
        
        hist1, bins = np.histogram(dist1, bins=num_bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        
        # 计算重叠面积
        overlap = np.minimum(hist1, hist2).sum() * (bins[1] - bins[0])
        return overlap
    
    def _calculate_repetition_rate(self, texts: List[str], ngram: int = 3) -> float:
        """计算文本的重复率
        
        使用n-gram方法计算文本中的重复片段比例
        """
        total_ngrams = 0
        repeated_ngrams = 0
        
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) < ngram:
                continue
                
            # 提取n-grams
            text_ngrams = [
                tuple(tokens[i:i+ngram])
                for i in range(len(tokens)-ngram+1)
            ]
            
            # 计算重复的n-grams
            ngram_counts = defaultdict(int)
            for ng in text_ngrams:
                ngram_counts[ng] += 1
                
            total_ngrams += len(text_ngrams)
            repeated_ngrams += sum(count - 1 for count in ngram_counts.values() if count > 1)
        
        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def format_metrics(
        self,
        dpo_metrics: Optional[DPOMetrics] = None,
        generation_metrics: Optional[GenerationMetrics] = None,
    ) -> Dict[str, float]:
        """将评估指标格式化为字典形式
        
        Args:
            dpo_metrics: DPO相关的评估指标
            generation_metrics: 生成质量相关的评估指标
            
        Returns:
            包含所有评估指标的字典
        """
        metrics = {}
        
        if dpo_metrics:
            metrics.update({
                "beta_mean": dpo_metrics.beta_mean,
                "beta_std": dpo_metrics.beta_std,
                "beta_min": dpo_metrics.beta_min,
                "beta_max": dpo_metrics.beta_max,
                "ppl_mean": dpo_metrics.ppl_mean,
                "ppl_std": dpo_metrics.ppl_std,
                "chosen_rejected_diff": dpo_metrics.chosen_rejected_diff,
                "chosen_rejected_overlap": dpo_metrics.chosen_rejected_overlap,
            })
        
        if generation_metrics:
            metrics.update({
                "bleu_score": generation_metrics.bleu_score,
                **{f"rouge_{k}": v for k, v in generation_metrics.rouge_scores.items()},
                "response_length_mean": generation_metrics.response_length_mean,
                "response_length_std": generation_metrics.response_length_std,
                "vocab_diversity": generation_metrics.vocab_diversity,
                "repetition_rate": generation_metrics.repetition_rate,
            })
        
        return metrics
