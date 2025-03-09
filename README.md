# Learnable Beta DPO 论文

本分支包含"可学习Beta值的DPO算法研究"论文的LaTeX源代码。

## 文件结构

```
latex/
├── main.tex          # 主LaTeX文件
├── sections/         # 章节文件
│   ├── introduction.tex  # 引言
│   ├── theory.tex        # 理论基础
│   ├── method.tex        # 方法
│   ├── experiments.tex   # 实验设置
│   ├── results.tex       # 实验结果
│   └── conclusion.tex    # 结论
├── figures/          # 图表文件
└── references.bib    # 参考文献
```

## 编译方法

使用以下命令编译论文：

```bash
cd latex
pdflatex main
bibtex main
pdflatex main
pdflatex main
``` 