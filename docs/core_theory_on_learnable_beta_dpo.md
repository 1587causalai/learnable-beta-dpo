# Learnable Beta DPO 的数学理论基础

## 1. 引言

### 1.1 背景：Direct Preference Optimization (DPO) 算法简介

Direct Preference Optimization (DPO) 是一种直接优化语言模型以符合人类偏好的算法。它避免了传统的强化学习方法中复杂的奖励建模和策略迭代过程，通过直接比较模型对 chosen 和 rejected 样本的输出，优化策略模型。DPO 以其简洁性和高效性，在对齐大型语言模型 (LLMs) 方面取得了显著的成功。

### 1.2 传统 DPO 的局限性：固定 Beta 值

在标准的 DPO 算法中，一个关键的超参数是 $\beta$，它控制着模型在参考策略和奖励信息之间的权衡。从信息融合的角度来看：

- 较大的 $\beta$ 值使模型更倾向于遵循参考策略 $\pi_{\text{ref}}$，保持原有行为
- 较小的 $\beta$ 值使模型更多地利用奖励信息，进行策略调整

然而，传统的 DPO 通常使用固定的 $\beta$ 值，这带来了两个主要局限：

1. **上下文不敏感**：不同场景下可能需要不同的探索-利用权衡
   - 在模型熟悉的领域，应该更多地保持参考策略的行为
   - 在模型不熟悉的领域，应该更多地从奖励信息中学习

2. **优化效率受限**：固定的权衡策略可能导致
   - 在某些场景下过度保守，错过学习机会
   - 在某些场景下过度激进，损失已有能力

这种"一刀切"的方式无法针对不同的上下文动态调整学习策略，限制了模型在复杂、多变场景下的优化效果。

### 1.3 Learnable Beta DPO 的核心思想：动态 Beta 值

Learnable Beta DPO 旨在克服传统 DPO 的局限性，核心思想是引入一个**动态的、可学习的 $\beta$ 值**，使其能够根据输入上下文 $x$ 自适应地调整。通过学习一个函数 $\beta(x)$，模型可以根据上下文的特性，更精细地控制 preference loss 的强度，从而更有效地学习人类偏好。

## 2. 回顾：标准 DPO 算法

### 2.1 偏好学习的目标：Bradley-Terry 模型

DPO 的理论基础是 Bradley-Terry 模型，该模型用于建模成对比较的偏好关系。假设我们有两个模型输出 $y_w$ (winner) 和 $y_l$ (loser)，对于给定的上下文 $x$，我们希望模型学习到一个策略 $\pi_\theta(y|x)$，使得 $y_w$ 比 $y_l$ 更受偏好的概率符合 Bradley-Terry 模型：

$$P(\text{winner} = y_w | x, y_w, y_l) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}$$

其中 $r(x, y)$ 是一个奖励函数，代表了模型输出 $y$ 在上下文 $x$ 下的奖励值。在 DPO 中，我们不直接学习奖励函数 $r(x, y)$，而是直接优化策略模型 $\pi_\theta(y|x)$。

### 2.2 DPO Loss Function 的推导

#### 2.2.1 从 Bradley-Terry 模型到 Loss Function

DPO 的目标是最大化 winner 样本的概率，同时最小化 loser 样本的概率。我们可以通过最大似然估计来推导 DPO 的 Loss Function。给定数据集 $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$，我们希望最大化以下似然函数：

$$L(\theta) = \prod_{i=1}^N P(\text{winner} = y_w^{(i)} | x^{(i)}, y_w^{(i)}, y_l^{(i)})$$

取负对数似然，得到 Loss Function：

$$\mathcal{L}_{\text{DPO}}(\theta) = - \sum_{i=1}^N \log \left( \frac{\exp(r(x^{(i)}, y_w^{(i)}))}{\exp(r(x^{(i)}, y_w^{(i)})) + \exp(r(x^{(i)}, y_l^{(i)}))} \right)$$

#### 2.2.2 DPO Loss 的简化形式

DPO 的一个关键假设是，奖励函数 $r(x, y)$ 可以表示为策略模型 $\pi_\theta(y|x)$ 与一个参考策略 $\pi_{\text{ref}}(y|x)$ 的对数比值：

$$r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

其中 $\beta$ 是一个常数，控制着奖励的 scale。将这个假设代入 Loss Function，并进行简化，可以得到 DPO Loss 的最终形式：

$$\mathcal{L}_{\text{DPO}}(\theta) = - \sum_{i=1}^N \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w^{(i)}|x^{(i)})}{\pi_{\text{ref}}(y_w^{(i)}|x^{(i)})} - \log \frac{\pi_\theta(y_l^{(i)}|x^{(i)})}{\pi_{\text{ref}}(y_l^{(i)}|x^{(i)})} \right] \right)$$

其中 $\sigma(z) = \frac{1}{1 + e^{-z}}$ 是 sigmoid 函数。更简洁的形式为：

$$\mathcal{L}_{\text{DPO}}(\theta) = - \sum_{i=1}^N \log \sigma \left( \beta \left[ \log \frac{\pi_\theta(y_w^{(i)}|x^{(i)})}{\pi_{\text{ref}}(y_w^{(i)}|x^{(i)})} - \log \frac{\pi_\theta(y_l^{(i)}|x^{(i)})}{\pi_{\text{ref}}(y_l^{(i)}|x^{(i)})} \right] \right)$$

### 2.3 标准 DPO 的训练过程

标准 DPO 的训练过程主要包括：

1.  **数据准备**: 收集偏好数据集，包含上下文 $x^{(i)}$，chosen 样本 $y_w^{(i)}$ 和 rejected 样本 $y_l^{(i)}$。
2.  **模型初始化**: 初始化策略模型 $\pi_\theta$ 和参考模型 $\pi_{ref}$ (通常 $\pi_{ref}$ 是训练前的策略模型)。
3.  **Loss 计算**: 对于每个 batch 的数据，计算 DPO Loss $\mathcal{L}_{DPO}(\theta)$。
4.  **梯度更新**: 使用梯度下降等优化算法，更新策略模型 $\pi_\theta$ 的参数，最小化 DPO Loss。
5.  **迭代训练**: 重复步骤 3 和 4，直到模型收敛或达到预定的训练步数。

## 3. Learnable Beta DPO 的数学模型

### 3.1 动态 Beta 值的引入：$\beta(x)$ 函数

Learnable Beta DPO 的核心创新在于将 DPO 中的固定超参数 $\beta$ 替换为一个**依赖于策略模型表征的函数 $\beta(x; \pi_\theta)$**。这个函数通过策略模型的隐状态表征和困惑度，来自适应地调整 preference loss 的强度。具体形式为：

$$\beta(x; \pi_\theta) = w \cdot \mathrm{PPL}_{\pi_\theta}(x) \cdot f(h_{\pi_\theta}(x))$$

其中：
- $h_{\pi_\theta}(x)$ 是策略模型 $\pi_\theta$ 处理输入 $x$ 后的最后一层隐状态
- $\mathrm{PPL}_{\pi_\theta}(x)$ 是用策略模型计算的困惑度
- $f(h) = 1 + \epsilon \cdot \tanh(\mathrm{NN}(h))$ 是 BetaHead 网络的调整函数

下面我们详细解释每个组成部分：

#### 3.1.1 可学习参数 $w$ 的作用

$w$ 是一个**可学习的标量参数**，用于调整整体 $\beta$ 值的量级。通过学习 $w$，模型可以自动确定适当的 scale。

#### 3.1.2 策略模型困惑度 $\mathrm{PPL}_{\pi_\theta}(x)$ 的意义

$\mathrm{PPL}_{\pi_\theta}(x)$ 反映了**策略模型对输入的确定性程度**。对于给定的输入序列 $x = (x_1, x_2, ..., x_m)$，困惑度定义为：

$$\mathrm{PPL}_{\pi_\theta}(x) = \exp \left( - \frac{1}{m} \sum_{i=1}^m \log \pi_\theta(x_i | x_{<i}) \right)$$

其中：
- $x = (x_1, x_2, ..., x_m)$ 是输入序列
- $x_{<i}$ 表示 $x_i$ 之前的所有 token
- $\pi_\theta(x_i | x_{<i})$ 是模型对第 $i$ 个 token 的预测概率

- 高困惑度表示策略模型对该输入的预测不确定
- 低困惑度表示策略模型对该输入的预测比较确定

#### 3.1.3 BetaHead 的调整函数 $f(h)$

函数 $f(h) = 1 + \epsilon \cdot \tanh(\mathrm{NN}(h))$ 基于策略模型的隐状态进行细粒度调整：

- **输入**：策略模型的最后一层隐状态 $h_{\pi_\theta}(x)$
- **网络结构**：一个简单的神经网络 $\mathrm{NN}(\cdot)$
- **输出范围**：通过 $\tanh$ 和小参数 $\epsilon$ 确保调整幅度在 $[1-\epsilon, 1+\epsilon]$ 范围内

这种设计使得 $\beta(x; \pi_\theta)$ 能够：
1. 通过隐状态捕捉输入的深层语义特征
2. 通过困惑度反映模型的确定性程度
3. 实现策略模型和 BetaHead 的协同学习

### 3.2 Learnable Beta DPO Loss Function

#### 3.2.1 将 $\beta(x)$ 融入 DPO Loss

在 Learnable Beta DPO 中，我们将标准 DPO Loss 中的固定 $\beta$ 值替换为动态的 $\beta(x)$ 函数。对于每个数据样本 $(x^{(i)}, y_w^{(i)}, y_l^{(i)})$, 我们根据上下文 $x^{(i)}$ 计算 $\beta(x^{(i)})$，然后将其代入 DPO Loss 函数。

#### 3.2.2 Loss Function 的完整形式

Learnable Beta DPO Loss Function 的形式如下：

$$\mathcal{L}_{\text{LearnableBetaDPO}}(\theta) = - \sum_{i=1}^N \log \sigma \left( \beta(x^{(i)}; \pi_\theta) \left[ \log \frac{\pi_\theta(y_w^{(i)}|x^{(i)})}{\pi_{\text{ref}}(y_w^{(i)}|x^{(i)})} - \log \frac{\pi_\theta(y_l^{(i)}|x^{(i)})}{\pi_{\text{ref}}(y_l^{(i)}|x^{(i)})} \right] \right)$$

其中：
- $\beta(x^{(i)}; \pi_\theta) = w \cdot \mathrm{PPL}_{\pi_\theta}(x^{(i)}) \cdot (1 + \epsilon \cdot \tanh(\mathrm{NN}(h_{\pi_\theta}(x^{(i)}))))$
- $h_{\pi_\theta}(x^{(i)})$ 是策略模型的最后一层隐状态
- $\sigma(z)$ 是 sigmoid 函数

## 4. Learnable Beta DPO 的训练过程

### 4.1 模型结构：BetaHead 网络与 DPO 模型

为了实现 Learnable Beta DPO，项目引入了一个与策略模型紧密耦合的 **BetaHead 网络**。这种设计的核心特点是 BetaHead 直接利用策略模型的内部表征，而不是独立处理输入：

1. **共享表征**：
   - BetaHead 直接使用策略模型 $\pi_\theta$ 的最后一层隐状态 $h_{\pi_\theta}(x)$
   - 这种设计使得 BetaHead 能够直接访问策略模型对输入的深层理解
   - 避免了重复的特征提取，提高了计算效率

2. **网络结构**：
   - 一个轻量级的神经网络 $\mathrm{NN}(\cdot)$，直接在策略模型的隐状态空间中操作
   - 可以是简单的线性层或多层感知机（MLP）
   - 输出维度为1，产生用于计算 $f(h_{\pi_\theta}(x))$ 的标量值

3. **协同学习**：
   - 策略模型的参数更新会直接影响 BetaHead 的输入表征
   - BetaHead 的梯度反向传播会影响策略模型的表征学习
   - 这种双向互动促进了两个网络的协同进化

完整的模型架构包含以下组件：
- 策略模型 $\pi_\theta$：核心模型，同时承担生成响应, 计算困惑度PPL和提供隐状态表征的任务
- BetaHead 网络：基于策略模型的隐状态计算动态 $\beta(x; \pi_\theta)$ 值
- 参考模型 $\pi_{\text{ref}}$：提供基准的模型（通常是训练前的基础模型）

#### 4.2 联合训练：Policy Model, BetaHead 的协同优化

Learnable Beta DPO 的训练过程需要协同优化多个组件。具体步骤如下：

1. **数据准备**：与标准 DPO 相同，准备好偏好对数据。

2. **模型初始化**：
   - 初始化策略模型 $\pi_\theta$
   - 加载参考模型 $\pi_{\text{ref}}$（固定参数，不参与训练）
   - 初始化 BetaHead 网络
   - 初始化可学习参数 $w$（例如，初始化为 1）

3. **前向计算流程**：对于每个 batch 的数据 $(x^{(i)}, y_w^{(i)}, y_l^{(i)})$：
   
   a. **获取隐状态表示**：
      - 使用策略模型 $\pi_\theta$ 处理输入文本 $x^{(i)}$
      - 提取 last hidden state $h^{(i)}$ 作为 BetaHead 的输入. 
   
   b. **计算困惑度**：
      - 使用策略模型 $\pi_\theta$ 计算上下文 $x^{(i)}$ 的困惑度 $\mathrm{PPL}_{\pi_\theta}(x^{(i)})$
   
   c. **计算动态 beta**：
      - 通过 BetaHead 网络计算 $\mathrm{NN}(h_{\pi_\theta}(x^{(i)}))$
      - 计算调整函数 $f(h_{\pi_\theta}(x^{(i)})) = 1 + \epsilon \cdot \tanh(\mathrm{NN}(h_{\pi_\theta}(x^{(i)})))$
      - 得到最终的 $\beta(x^{(i)}; \pi_\theta) = w \cdot \mathrm{PPL}_{\pi_\theta}(x^{(i)}) \cdot f(h_{\pi_\theta}(x^{(i)}))$
   
   d. **计算 Loss**：
      - 使用动态 $\beta(x^{(i)})$ 计算 Learnable Beta DPO Loss $\mathcal{L}_{\text{LearnableBetaDPO}}(\theta)$

4. **梯度更新**：
   - 同时更新策略模型 $\pi_\theta$ 的参数
   - 更新 BetaHead 网络的参数
   - 更新可学习参数 $w$

5. **迭代训练**：重复步骤 3 和 4，直到模型收敛或达到预定的训练步数。

这种架构设计确保了：
1. BetaHead 网络能够利用 LLM 提取的深层语义特征
2. PPL 计算使用固定的参考模型，提供稳定的困惑度估计
3. 整个系统可以端到端训练，同时保持参考模型的稳定性

## 5. 理论分析与优势

### 5.1 动态 Beta 的直觉解释：探索与利用的自适应平衡

动态 $\beta(x)$ 的核心优势在于**自适应地平衡探索与利用**。根据信息融合理论，$\beta$ 值控制了模型在不同上下文中如何权衡参考策略和奖励信息：

1. **较大的 $\beta(x)$ 值**：
   - 更倾向于遵循参考策略 $\pi_{\text{ref}}$
   - 适用于模型较为熟悉的领域
   - 例如：对于常见的指令遵循任务，模型已经具有良好的基础能力

2. **较小的 $\beta(x)$ 值**：
   - 更多地利用奖励信息进行探索
   - 适用于模型不太熟悉的领域
   - 例如：对于创新性任务或特殊领域，模型需要更多地从人类偏好中学习

这种动态调整机制使得模型能够：
- 在熟悉的场景中保持稳定性（较大的 $\beta$）
- 在新颖的场景中保持灵活性（较小的 $\beta$）
- 通过 $\mathrm{PPL}_{\pi_\theta}(x)$ 自动感知策略模型对当前输入的确定性程度
- 通过 BetaHead 网络基于隐状态 $h_{\pi_\theta}(x)$ 学习更细粒度的上下文特征

这种自适应机制使得模型能够：
1. 当策略模型对输入较为确定时（低困惑度），倾向于保持当前行为
2. 当策略模型对输入不确定时（高困惑度），更多地利用奖励信息进行学习
3. 通过 BetaHead 网络捕捉深层语义特征，实现更精细的调整

### 5.2 理论挑战与未来研究方向

虽然 Learnable Beta DPO 具有潜在的优势，但也面临一些理论挑战和未来研究方向：

*   **$\beta(x)$ 函数的设计**:  当前的 $\beta(x) = w \cdot \mathrm{PPL}(x) \cdot f(x)$ 形式是一种启发式设计，是否还有更优的设计方案？例如，是否可以考虑更复杂的上下文特征，或者使用更精细的神经网络结构来建模 $\beta(x)$？
*   **理论分析**:  动态 $\beta$ 对 DPO 算法的收敛性、稳定性和泛化性有何影响？是否可以从理论上证明动态 $\beta$ 在某些条件下可以提高 DPO 的性能？
*   **与其他动态 $\beta$ 方法的比较**:  是否存在其他动态调整 DPO 中 $\beta$ 值的方法？本项目的方法与这些方法相比，有什么优势和劣势？

### 6. 总结

Learnable Beta DPO 通过引入动态的、可学习的 $\beta(x)$ 函数，扩展了标准 DPO 算法。$\beta(x)$ 函数的设计结合了可学习参数 $w$、上下文困惑度 $\mathrm{PPL}(x)$ 和神经网络 $f(x)$，旨在使 preference loss 能够根据上下文自适应调整。这种动态 $\beta$ 的方法有望提高 DPO 算法的灵活性、适应性和性能，是 DPO 算法研究的一个有前景的方向。未来的研究可以进一步探索 $\beta(x)$ 函数的更优设计，并进行更深入的理论分析和实验验证。

