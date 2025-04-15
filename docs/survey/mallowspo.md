# MallowsPO: Fine-Tune Your LLM with Preference Dispersions 泛读笔记

## 论文概述
- **标题**：MallowsPO: Fine-Tune Your LLM with Preference Dispersions  
- **作者**：Haoxian Chen, Hanyang Zhao, Henry Lam, David D. Yao, Wenpin Tang  
- **主要内容**：论文提出了一种名为 MallowsPO 的新方法，用于优化大型语言模型（LLM）的直接偏好优化（Direct Preference Optimization, DPO）。通过引入 Mallows 偏好排名理论，MallowsPO 解决了 DPO 在表征人类偏好多样性方面的不足。

## 核心问题
- **DPO 的局限性**：DPO（直接偏好优化）无法充分捕捉人类偏好的多样性，导致模型在某些场景下的表现不够理想。  
- **解决方案**：MallowsPO 引入了 **偏好分散**（Preference Dispersions）的概念，结合 Mallows 模型，提升模型对人类偏好的理解和优化能力。

## Mallows 偏好排名理论
- **基本概念**：Mallows 模型通过以下两个关键参数描述偏好排名的分布：  
  - 中心排名 $\mu_0$：表示偏好的基准排序。  
  - 分散参数 $\phi \in (0, 1]$：衡量偏好分布的分散程度。  
    - $\phi \to 0$：偏好高度集中，所有人倾向于相似的排序。  
    - $\phi \to 1$：偏好接近均匀分布，个体差异较大。  
- **距离度量**：  
  - Mallows-θ：基于 Spearman's rho 距离。  
  - Mallows-φ：基于 Kendall's tau 距离。

## MallowsPO 方法
- **目标**：通过 Mallows 模型优化语言模型 $\pi$，使其生成的回答更符合人类偏好的多样性分布。  
- **优化目标**：
$$\min_\pi -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \left( g_{d, \phi(x)} \left( \beta \log \frac{\pi(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right) \right]$$
  - $g_{d, \phi(x)}$：链接函数，依赖于选择的距离度量 $d$ 和分散参数 $\phi(x)$。  
  - $y_w$ 和 $y_l$：分别表示偏好中"胜者"（更优）和"败者"（较差）的回答。  
- **分散参数 $\phi(x)$**：通过基于 Shannon 熵的统计方法估计：
$$-\phi^* \log \left( \frac{1}{2 \log n} \sum_{i=1}^{N-1} \left[ H(Y_{i+1} \mid Y_i = y_i^w) + H(Y_{i+1} \mid Y_i = y_i^l) \right] \right)$$
该方法根据数据动态调整分散程度。

## 主要贡献
1. **形式化偏好分散**：首次将偏好分散的概念引入 DPO，提出了 MallowsPO 方法。  
2. **分散指数近似**：设计了一种近似方法来估计分散参数 $\phi(x)$，并通过合成数据验证其有效性。  
3. **多功能优化目标**：开发了基于不同距离度量的 MallowsPO-θ 和 MallowsPO-φ，并提出了统一的 $\Psi$PO 模型。  
4. **广泛实验验证**：在多个场景（如合成数据和语言模型微调）中，MallowsPO 的性能优于传统 DPO。

## 实验结果
- **合成 bandits 实验**：在合成数据测试中，MallowsPO 展示了其捕捉偏好分散的能力。  
- **语言模型微调**：  
  - 在 Anthropic HH 数据集上微调 Pythia 2.8B，MallowsPO 的表现优于 DPO。  
  - 在 UltraFeedback 数据集上微调 Llama3-8B-Instruct，MallowsPO 同样取得了更好的结果。  
- **结论**：MallowsPO 显著提升了模型对偏好多样性的理解能力，整体性能优于 DPO。

## 结论
MallowsPO 通过引入 Mallows 模型和偏好分散的概念，克服了 DPO 在处理人类偏好多样性方面的局限性。这种方法在数学上严谨且实践效果显著，为语言模型的偏好优化提供了新的研究方向。

## 与 Learnable Beta DPO 的关系
MallowsPO 的上下文缩放机制（通过 $\phi(x)$ 进行动态调整）与 Learnable Beta DPO 中的动态 $\beta$ 参数计算有相似之处。两者都试图根据输入上下文动态调整优化强度，但出发点不同：
- MallowsPO 关注偏好的分散性和多样性
- Learnable Beta DPO 关注模型对输入的确定性程度

这两种方法可能在某些场景下存在互补效应，值得进一步研究结合的可能性。 