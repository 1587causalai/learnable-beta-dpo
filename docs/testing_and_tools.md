# 测试和工具脚本使用指南

本文档详细说明了项目中的测试系统和工具脚本的使用方法。

## 1. 测试系统

### 1.1 运行测试

运行所有测试：
```bash
python -m unittest tests/test_*.py -v
```

运行特定测试：
```bash
# 运行特定测试文件
python -m unittest tests/test_beta_head.py -v

# 运行特定测试类
python -m unittest tests.test_beta_head.TestBetaHead -v

# 运行特定测试方法
python -m unittest tests.test_beta_head.TestBetaHead.test_linear_beta_head -v
```

### 1.2 测试组件

1. BetaHead 测试 (`test_beta_head.py`)
   - `test_linear_beta_head`: 测试线性 BetaHead
   - `test_mlp_beta_head`: 测试 MLP BetaHead
   - `test_epsilon_bounds`: 测试 f(x) 的取值范围
   - `test_input_validation`: 测试输入验证
   - `test_invalid_head_type`: 测试无效的 head 类型

2. 数据集测试 (`test_dataset.py`)
   - `test_dataset_creation`: 测试数据集创建
   - `test_from_hf_dataset`: 测试从 HuggingFace 数据集创建
   - `test_collate_fn`: 测试批处理函数
   - `test_input_validation`: 测试输入验证
   - `test_max_length_truncation`: 测试最大长度截断
   - `test_empty_inputs`: 测试空输入处理

3. 指标计算测试 (`test_metrics.py`)
   - `test_dpo_metrics`: 测试 DPO 相关指标计算
   - `test_generation_metrics`: 测试生成质量相关指标计算
   - `test_distribution_overlap`: 测试分布重叠度计算
   - `test_repetition_rate`: 测试重复率计算
   - `test_format_metrics`: 测试指标格式化

4. 完整流程测试 (`test_full_pipeline.py`)
   - `test_training`: 测试训练流程
   - `test_inference`: 测试推理流程
   - `test_batch_inference`: 测试批量推理

### 1.3 Mock 环境

项目提供了完整的 mock 环境用于测试：

1. Mock 模型 (`mock_qwen.py`)
   ```python
   from tests.mock.mock_qwen import MockQwenForCausalLM, MockQwenTokenizer
   
   # 创建 mock 模型
   model = MockQwenForCausalLM()
   tokenizer = MockQwenTokenizer()
   ```

2. Mock 数据集 (`mock_dataset.py`)
   ```python
   from tests.mock.mock_dataset import load_dataset
   
   # 加载 mock 数据集
   dataset = load_dataset("mock_dataset", split="train")
   ```

## 2. 工具脚本

### 2.1 数据分析脚本

```bash
# 使用真实数据集
python scripts/analyze_data.py \
    --dataset_name Anthropic/hh-rlhf \
    --split train \
    --output_dir analysis_results

# 使用 mock 数据测试
python scripts/analyze_data.py --use_mock
```

分析内容包括：
- 文本长度分布
- 词汇统计
- 响应差异分析

### 2.2 模型评估脚本

```bash
# 使用真实模型
python scripts/evaluate.py \
    --model_dir path/to/model \
    --test_file test_data.jsonl \
    --output_dir evaluation_results

# 使用 mock 环境测试
python scripts/evaluate.py \
    --model_dir mock_model \
    --use_mock
```

评估指标包括：
- DPO 相关指标（beta 值统计、PPL 等）
- 生成质量指标（BLEU、ROUGE 等）
- 词汇多样性和重复率

### 2.3 可视化脚本

```bash
python scripts/visualize.py \
    --data_analysis analysis_results/data_analysis.json \
    --eval_results evaluation_results/evaluation_results.json \
    --output_dir visualization_results
```

可视化内容包括：
- 长度分布图
- 词频分布图
- 词汇多样性饼图
- 响应差异分布图
- Beta 值统计图
- 生成质量指标图

## 3. 最佳实践

1. 开发新功能时的测试流程：
   ```bash
   # 1. 先运行所有测试确保基线正常
   python -m unittest tests/test_*.py -v
   
   # 2. 使用 mock 环境测试新功能
   python scripts/your_script.py --use_mock
   
   # 3. 添加新的测试用例
   # 4. 再次运行所有测试
   python -m unittest tests/test_*.py -v
   ```

2. 使用工具脚本分析改动效果：
   ```bash
   # 1. 分析数据
   python scripts/analyze_data.py --use_mock
   
   # 2. 评估模型
   python scripts/evaluate.py --model_dir mock_model --use_mock
   
   # 3. 可视化结果
   python scripts/visualize.py \
       --data_analysis analysis_results/data_analysis.json \
       --eval_results evaluation_results/evaluation_results.json
   ```

3. 调试建议：
   - 优先使用 mock 环境进行快速测试
   - 确保每个改动都有对应的测试用例
   - 使用工具脚本验证改动效果
   - 保持测试和文档的同步更新 