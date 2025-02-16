# Learnable Beta DPO

这是一个基于可学习 beta 值的 DPO (Direct Preference Optimization) 实现。项目的核心思想是设计一个可学习的函数：

$$\beta(x) = w \cdot PPL(x) \cdot f(x)$$

其中：
- $w$ 是一个可学习的参数
- $PPL(x)$ 是上下文 $x$ 的困惑度
- $f(x)$ 是上下文 $x$ 的函数，其取值范围为 $[1-\epsilon, 1+\epsilon]$
- 具体实现中，$f(x) = 1 + \epsilon \cdot \tanh(NN(x))$

## 项目进展

### 1. 核心功能实现 ✅

- [x] BetaHead 网络
  - [x] 线性和 MLP 两种结构
  - [x] 动态 beta 值计算
  - [x] PPL 和上下文相关调整
  - [x] 输入验证和错误处理

- [x] DPO 模型
  - [x] 基础模型集成
  - [x] 动态 beta 计算
  - [x] 损失函数实现
  - [x] 生成功能支持

### 2. 数据处理 ✅

- [x] DPODataset 实现
  - [x] 标准偏好数据格式支持
  - [x] HuggingFace 数据集集成
  - [x] 高效批处理
  - [x] 数据验证和清理

### 3. 训练系统 ✅

- [x] DPOTrainer 实现
  - [x] 完整训练循环
  - [x] 梯度累积
  - [x] 学习率调度
  - [x] 模型保存/加载
  - [x] Wandb 监控

### 4. 评估系统 ✅

- [x] DPO 相关指标
  - [x] Beta 值统计
  - [x] PPL 统计
  - [x] 分布重叠度
  - [x] 差异分析

- [x] 生成质量指标
  - [x] BLEU 分数
  - [x] ROUGE 分数
  - [x] 响应长度统计
  - [x] 词汇多样性
  - [x] 重复率

### 5. 测试系统 ✅

- [x] Mock 环境
  - [x] Mock Qwen 模型
  - [x] Mock 数据集
  - [x] Mock Tokenizer

- [x] 单元测试
  - [x] BetaHead 测试
  - [x] 评估指标测试
  - [x] 数据集测试

- [x] 集成测试
  - [x] 端到端流程测试
  - [x] 训练流程测试
  - [x] 推理流程测试

### 6. 工具脚本 ✅

- [x] 数据分析脚本
  - [x] 长度分布分析
  - [x] 词汇统计
  - [x] 响应差异分析

- [x] 评估脚本
  - [x] DPO 指标评估
  - [x] 生成质量评估
  - [x] 批量评估支持

- [x] 可视化脚本
  - [x] 数据分布可视化
  - [x] 评估指标可视化
  - [x] 结果导出

## 项目结构

```
learnable-beta-dpo-quick-start/
├── src/                      # 源代码
│   ├── models/              # 模型实现
│   │   ├── beta_head.py     # BetaHead网络
│   │   └── dpo_model.py     # DPO模型
│   ├── data/                # 数据处理
│   │   └── dataset.py       # 数据集实现
│   ├── trainers/            # 训练相关
│   │   └── dpo_trainer.py   # 训练器
│   └── utils/               # 工具函数
│       └── metrics.py       # 评估指标
├── scripts/                  # 脚本文件
│   ├── train.py            # 训练脚本
│   ├── inference.py        # 推理脚本
│   ├── analyze_data.py     # 数据分析
│   ├── evaluate.py         # 模型评估
│   └── visualize.py        # 结果可视化
├── tests/                   # 测试文件
│   ├── mock/               # Mock环境
│   │   ├── mock_qwen.py    # Mock模型
│   │   └── mock_dataset.py # Mock数据
│   ├── test_beta_head.py   # BetaHead测试
│   ├── test_metrics.py     # 指标测试
│   ├── test_dataset.py     # 数据集测试
│   └── test_full_pipeline.py # 完整流程测试
├── docs/                    # 文档
│   └── testing_and_tools.md # 测试和工具指南
└── requirements.txt         # 项目依赖
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python scripts/train.py \
    --model_name_or_path Qwen/Qwen-1_5B \
    --beta_head_type linear \
    --dataset_name Anthropic/hh-rlhf \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --use_wandb
```

### 3. 评估模型

```bash
# 使用真实模型评估
python scripts/evaluate.py \
    --model_dir path/to/model \
    --test_file test_data.jsonl

# 使用 mock 环境测试
python scripts/evaluate.py \
    --model_dir mock_model \
    --use_mock
```

### 4. 分析数据

```bash
# 分析真实数据集
python scripts/analyze_data.py \
    --dataset_name Anthropic/hh-rlhf

# 使用 mock 数据测试
python scripts/analyze_data.py --use_mock
```

### 5. 可视化结果

```bash
python scripts/visualize.py \
    --data_analysis analysis_results/data_analysis.json \
    --eval_results evaluation_results/evaluation_results.json
```

### 6. 运行测试

```bash
# 运行所有测试
python -m unittest tests/test_*.py -v

# 运行特定测试
python -m unittest tests/test_beta_head.py -v
```

更多详细信息请参考 [测试和工具脚本使用指南](docs/testing_and_tools.md)。

## 贡献

欢迎提交 Issue 和 Pull Request！在提交代码前，请确保：

1. 所有测试都已通过
2. 新功能有对应的测试用例
3. 文档已更新
4. 代码风格符合项目规范

## 许可证

MIT License
