# Zero2All LLM 项目

从零开始建立一个类似 Qwen1.5-0.5B 的大语言模型。

## 项目目标

本项目旨在帮助理解大型语言模型的技术细节，通过从零构建一个小型但功能完整的类 Qwen 模型。具体目标包括：

1. 实现一个基于Transformer架构的0.5B参数的语言模型
2. 采用现代设计（分组查询注意力、字节级BPE分词等）
3. 提供完整的训练和推理代码
4. 提供清晰的代码注释和文档说明

## 模型架构

我们实现的模型架构与Qwen1.5-0.5B类似：

- **隐藏层大小**: 896
- **层数**: 24
- **查询头数**: 14
- **KV头数**: 2
- **注意力头大小**: 64
- **中间层大小**: 4864
- **词表大小**: 151,646
- **上下文窗口**: 支持32K

详细架构设计请参考 `model_architecture.md` 文件。

## 目录结构

```
zero2all/
├── data/               # 训练和评估数据
├── src/
│   ├── tokenizer/      # 分词器实现
│   ├── model/          # 模型实现
│   │   ├── attention.py    # 注意力机制实现
│   │   ├── transformer.py  # Transformer实现
│   │   ├── mlp.py          # 多层感知机实现
│   │   ├── embedding.py    # 嵌入层实现
│   │   └── llm.py          # 语言模型实现
│   ├── trainer/        # 训练相关代码
│   │   ├── optimizer.py    # 优化器
│   │   ├── scheduler.py    # 学习率调度
│   │   └── trainer.py      # 训练循环
│   └── utils/          # 工具函数
├── examples/           # 示例代码
│   ├── simple_train.py     # 简单训练示例
│   └── simple_inference.py # 简单推理示例
├── checkpoints/        # 模型检查点
└── scripts/            # 脚本工具
```

## 安装依赖

```bash
# 克隆仓库
git clone https://github.com/yourusername/zero2all.git
cd zero2all

# 安装依赖
pip install -e .
```

## 使用示例

### 训练模型

```bash
# 简单训练示例
python examples/simple_train.py \
  --data_file path/to/your/data.txt \
  --output_dir output/my_model \
  --num_layers 24 \
  --hidden_size 896 \
  --batch_size 8 \
  --epochs 3
```

### 推理示例

```bash
# 简单推理示例
python examples/simple_inference.py \
  --model_path output/my_model \
  --tokenizer_path output/my_model \
  --prompt "这是一个测试，请模型继续生成文本："
```

## 模型训练流程

1. **数据准备**: 准备高质量的训练语料
2. **分词器训练**: 训练字节级BPE分词器
3. **模型预训练**: 大规模预训练（自回归语言建模）
4. **能力评估**: 使用标准基准测试评估模型性能

## 参考资料

- [Qwen技术报告](https://qwenlm.github.io/blog/)
- [Transformer论文](https://arxiv.org/abs/1706.03762)
- [旋转位置编码](https://arxiv.org/abs/2104.09864)
- [分组查询注意力机制](https://arxiv.org/abs/2305.13245)

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

