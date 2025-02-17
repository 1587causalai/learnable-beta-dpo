import os
import json
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_all_results(self) -> Dict[str, Any]:
        """加载所有分析结果"""
        results = {}
        
        # 加载数据分析结果
        analysis_file = Path("analysis_results/data_analysis.json")
        if analysis_file.exists():
            with open(analysis_file, "r", encoding="utf-8") as f:
                results["data_analysis"] = json.load(f)
                logger.info("已加载数据分析结果")
        else:
            logger.warning("数据分析结果文件不存在")
        
        # 加载评估结果
        eval_file = Path("evaluation_results/evaluation_results.json")
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                results["evaluation"] = json.load(f)
                logger.info("已加载评估结果")
        else:
            logger.warning("评估结果文件不存在")
            
        # 加载训练日志
        training_file = Path("training_logs/metrics.json")
        if training_file.exists():
            with open(training_file, "r", encoding="utf-8") as f:
                results["training"] = json.load(f)
                logger.info("已加载训练日志")
        else:
            logger.warning("训练日志文件不存在")
            
        return results
    
    def generate_overview_section(self) -> str:
        """生成项目概述部分"""
        return """# Learnable Beta DPO 实验报告

## 1. 项目概述与目标

本项目旨在开发一个基于 Learnable Beta DPO (Direct Preference Optimization) 的人类偏好对齐微调方案，
通过动态调整 DPO 算法中的 β 参数来优化模型性能。项目使用 Qwen1.5B 作为基础模型，
实现了一个创新的 BetaHead 网络结构来动态预测最优的 β 值。

### 主要目标：

1. 实现 Learnable Beta DPO 算法
2. 完成 Qwen1.5B 模型的人类偏好对齐微调
3. 验证动态 β 参数的效果
4. 提供完整的实验分析和评估结果
"""

    def generate_method_section(self) -> str:
        """生成方法原理部分"""
        return """## 2. Learnable Beta DPO 方法原理

### 2.1 核心创新

Learnable Beta DPO 的核心创新在于引入可学习的 BetaHead 网络，该网络能够：

1. 直接利用策略模型的内部表征
2. 结合模型计算的困惑度 (PPL)
3. 通过轻量级网络动态预测最优 β 值

### 2.2 技术优势

- 实现探索-利用平衡的精细控制
- 提高计算效率
- 实现策略模型和 BetaHead 的协同进化
"""

    def generate_model_section(self) -> str:
        """生成模型架构部分"""
        return """## 3. 模型架构与参数设置

### 3.1 基础模型

- 模型：Qwen1.5B
- 参数量：1.5B
- 架构：Decoder-only Transformer

### 3.2 BetaHead 网络结构

- 输入：策略模型最后一层隐状态
- 中间层：线性变换 + 激活函数
- 输出：标量 β 值
"""

    def generate_dataset_section(self, analysis_results: Dict[str, Any]) -> str:
        """生成数据集描述部分"""
        data_analysis = analysis_results.get("data_analysis", {})
        dataset_info = data_analysis.get("dataset_info", {})
        length_stats = data_analysis.get("length_distribution", {})
        vocab_stats = data_analysis.get("vocabulary_statistics", {})
        diff_stats = data_analysis.get("response_differences", {})
        
        # 预处理所有需要格式化的数值
        prompt_length = length_stats.get("prompt_length", {}).get("mean", 0)
        chosen_length = length_stats.get("chosen_length", {}).get("mean", 0)
        rejected_length = length_stats.get("rejected_length", {}).get("mean", 0)
        
        vocab_diversity = vocab_stats.get("vocabulary_diversity", 0)
        
        length_diff = diff_stats.get("length_difference", {}).get("mean", 0)
        overlap_mean = diff_stats.get("vocabulary_overlap", {}).get("mean", 0)
        
        return f"""## 4. 数据集分析

### 4.1 基本信息

- 数据集名称：{dataset_info.get("name", "N/A")} {"(Mock)" if dataset_info.get("name") == "mock_dataset" else ""}
- 数据集大小：{dataset_info.get("size", "N/A")} 条样本
- 数据集分片：{dataset_info.get("split", "N/A")}

### 4.2 文本长度统计

- Prompt 平均长度：{prompt_length:.2f} tokens
- Chosen Response 平均长度：{chosen_length:.2f} tokens
- Rejected Response 平均长度：{rejected_length:.2f} tokens

### 4.3 词汇统计

- 总词数：{vocab_stats.get("total_words", "N/A")}
- 唯一词数：{vocab_stats.get("unique_words", "N/A")}
- 词汇多样性：{vocab_diversity:.4f}

### 4.4 响应差异分析

- 平均长度差异：{length_diff:.2f} tokens
- 词汇重叠度：{overlap_mean * 100:.2f}%
"""

    def generate_training_plots(self, training_metrics: Dict[str, Any]) -> None:
        """生成训练相关的图表"""
        plots_dir = Path(self.output_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 生成训练Loss曲线
        if training_metrics.get("train_loss") and training_metrics.get("eval_loss"):
            plt.figure(figsize=(10, 6))
            
            # 绘制训练loss
            train_steps = range(len(training_metrics["train_loss"]))
            plt.plot(train_steps, training_metrics["train_loss"], 
                    label="Training Loss", alpha=0.6)
            
            # 绘制验证loss
            eval_steps = training_metrics.get("metrics", {}).get("steps", [])
            if eval_steps:
                plt.plot(eval_steps, training_metrics["eval_loss"], 
                        label="Validation Loss", marker='o')
            
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(plots_dir / "training_loss.png")
            plt.close()
        
        # 生成β值分布图
        if training_metrics.get("beta_values"):
            plt.figure(figsize=(10, 6))
            sns.histplot(training_metrics["beta_values"], bins=50)
            plt.xlabel("β Value")
            plt.ylabel("Count")
            plt.title("Distribution of Dynamic β Values")
            plt.grid(True)
            plt.savefig(plots_dir / "beta_distribution.png")
            plt.close()

    def generate_training_section(self, all_results: Dict[str, Any]) -> str:
        """生成训练过程描述部分"""
        training_metrics = all_results.get("training", {})
        
        # 生成图表
        self.generate_training_plots(training_metrics)
        
        plots_text = ""
        plots_dir = Path(self.output_dir) / "plots"
        
        if (plots_dir / "training_loss.png").exists():
            plots_text += "\n![Training Loss](plots/training_loss.png)\n"
        
        if (plots_dir / "beta_distribution.png").exists():
            plots_text += "\n![Beta Distribution](plots/beta_distribution.png)\n"
        
        metrics = training_metrics.get("metrics", {})
        
        return f"""## 5. 训练过程分析

### 5.1 训练配置

- 批次大小：{training_metrics.get("batch_size", "N/A")}
- 学习率：{training_metrics.get("learning_rate", "N/A")}
- 训练轮数：{training_metrics.get("epochs", "N/A")}
- 优化器：{training_metrics.get("optimizer", "N/A")}
- 设备：{training_metrics.get("device", "N/A")}

### 5.2 训练曲线
{plots_text}

### 5.3 训练过程指标

- PPL变化：{self._format_metric_history(metrics.get("ppl_history", []))}
- 准确率变化：{self._format_metric_history(metrics.get("accuracy_history", []))}
- 质量分数变化：{self._format_metric_history(metrics.get("quality_scores", []))}
"""

    def _format_metric_history(self, history: List[float]) -> str:
        """格式化指标历史数据"""
        if not history:
            return "N/A"
        return f"起始值 {history[0]:.2f} → 最终值 {history[-1]:.2f}"

    def generate_evaluation_section(self, all_results: Dict[str, Any]) -> str:
        """生成评估结果部分"""
        eval_results = all_results.get("evaluation", {})
        
        # 预处理百分比值
        accuracy = eval_results.get("human_preference_accuracy", 0)
        accuracy_str = f"{accuracy:.1f}%"
        
        return f"""## 6. 评估结果

### 6.1 生成质量评估

- BLEU Score：{eval_results.get("bleu_score", "N/A")}
- ROUGE-1：{eval_results.get("rouge_rouge1", "N/A")}

### 6.2 DPO特定指标

- Beta均值：{eval_results.get("beta_mean", "N/A")}
- PPL均值：{eval_results.get("ppl_mean", "N/A")}

### 6.3 人类偏好对齐效果

- 偏好准确率：{accuracy_str}
- 响应质量评分：{eval_results.get("response_quality", "N/A")}/10
"""

    def generate_conclusion_section(self, all_results: Dict[str, Any]) -> str:
        """生成结论部分"""
        eval_results = all_results.get("evaluation", {})
        training_metrics = all_results.get("training", {})
        
        # 计算一些关键改进指标
        initial_ppl = training_metrics.get("metrics", {}).get("ppl_history", [0])[0]
        final_ppl = eval_results.get("ppl_mean", 0)
        ppl_improvement = ((initial_ppl - final_ppl) / initial_ppl * 100) if initial_ppl else 0
        
        # 预处理百分比值
        accuracy = eval_results.get("human_preference_accuracy", 0)
        accuracy_str = f"{accuracy:.1f}%"
        improvement_str = f"{ppl_improvement:.1f}%"
        
        return f"""## 7. 结论与展望

### 7.1 主要发现

1. PPL改善：相比初始状态改善了{improvement_str}
2. 人类偏好对齐准确率达到{accuracy_str}
3. 动态β值能有效平衡探索与利用，均值为{eval_results.get("beta_mean", "N/A")}

### 7.2 未来工作

1. 探索更复杂的BetaHead网络结构
2. 在更大规模模型上验证方法有效性
3. 研究β值分布与任务特性的关系
4. 优化训练效率和资源利用
"""

    def generate_report(self) -> str:
        """生成完整报告"""
        try:
            # 加载所有结果
            all_results = self.load_all_results()
            
            sections = [
                self.generate_overview_section(),
                self.generate_method_section(),
                self.generate_model_section(),
                self.generate_dataset_section(all_results),
                self.generate_training_section(all_results),
                self.generate_evaluation_section(all_results),
                self.generate_conclusion_section(all_results)
            ]
            
            report = "\n\n".join(sections)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path(self.output_dir) / f"experiment_report_{timestamp}.md"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            
            logger.info(f"报告已生成：{report_file}")
            return report
            
        except Exception as e:
            logger.error(f"生成报告时发生错误：{str(e)}")
            raise

def main():
    generator = ReportGenerator()
    generator.generate_report()

if __name__ == "__main__":
    main() 