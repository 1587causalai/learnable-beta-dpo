# Learnable Beta DPO 论文 LaTeX 源文件

本目录包含 Learnable Beta DPO 论文的完整 LaTeX 源代码。

## 文件结构

- `main.tex` - 主LaTeX文件，包含文档结构和格式设置
- `sections/` - 存放各个章节的内容
  - `introduction.tex` - 引言
  - `theory.tex` - 理论基础
  - `method.tex` - 方法部分
  - `experiments.tex` - 实验设置
  - `results.tex` - 实验结果与分析
  - `conclusion.tex` - 结论与未来工作
- `references.bib` - BibTeX格式的参考文献
- `figures/` - 图表目录（需要自行添加图表文件）

## 编译说明

使用以下命令编译论文：

```bash
# 生成辅助文件
pdflatex main
# 生成参考文献
bibtex main
# 更新引用和交叉引用
pdflatex main
pdflatex main
```

## 需要添加的图表

需要在 `figures/` 目录中添加以下图表：

1. `beta_head_architecture.pdf` - BetaHead网络架构图
2. `domain_performance.pdf` - 不同方法在各领域的偏好一致率对比图
3. `beta_distribution.pdf` - 不同任务类型下β值的分布图

## 注意事项

- 本模板基于IEEE会议格式
- 数学公式使用LaTeX标准语法
- 参考文献使用BibTeX管理
- 图表应使用矢量格式（如PDF）以保证打印质量 