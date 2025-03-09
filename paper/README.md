# Learnable Beta DPO 论文 

本目录包含"可学习Beta值的DPO算法研究"论文的LaTeX源代码。

## 文件结构

```
paper/
├── main.tex           # 主LaTeX文件
├── sections/          # 各章节内容
│   ├── introduction.tex  # 引言
│   ├── theory.tex        # 理论基础
│   ├── method.tex        # 方法
│   ├── experiments.tex   # 实验设置
│   ├── results.tex       # 实验结果
│   └── conclusion.tex    # 结论
├── figures/           # 图表文件目录
├── references.bib     # 参考文献
└── README.md          # 本文件
```

## 编译方法

使用以下命令编译论文：

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## 图表文件

需要添加以下图表文件到 `figures/` 目录：

1. `beta_head_architecture.pdf` - BetaHead网络架构图
2. `domain_performance.pdf` - 不同方法在各领域的性能对比图
3. `beta_distribution.pdf` - 不同任务类型下β值的分布图

## 注意事项

- 论文使用IEEE会议格式
- 中英文混排，主要使用中文
- 所有公式使用LaTeX标准数学环境
- 参考文献使用BibTeX管理 