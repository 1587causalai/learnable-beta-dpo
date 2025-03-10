# 阅读笔记 LLM Post-Training: A Deep Dive into Reasoning Large Language Models 


飞书笔记文档 [link](https://swze06osuex.feishu.cn/docx/C9cvdvuB3oQjoexwCaPchXmbnSd)

后训练技术核心逻辑
[Awesome-LLM-Post-training](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training)


Task: 
- 从这篇公众号文章开始 [link](https://mp.weixin.qq.com/s/W-ubJWJVmwGo0Ce6Bhr1Rw) 出发，记录LLM后训练技术综述的阅读笔记。
- 记录文章中提到的LLM后训练技术，并记录其原理、应用场景、优缺点等。


![20250310102341](https://s2.loli.net/2025/03/10/EAOu79qgJhKvZlD.png)


1. 后训练三大核心技术体系：
   - 监督微调(SFT)
   - 强化学习优化(RLHF, DPO, ORPO, GRPO等)
   - 测试时扩展(TTS)

2. 奖励建模的创新：
   - 过程奖励与结果奖励的比较
   - 群体相对策略优化(GRPO)

3. 微调与强化学习面临的挑战：
   - 灾难性遗忘
   - 奖励欺骗

4. 未来发展趋势与研究方向：
   - 自适应计算
   - 多智能体协作
   - 隐私保护个性化



## 核心数学公式

监督微调(SFT) loss:

$$
L_{\text{MLE}} = -\sum_{i=1}^{T} \log P_{\theta}(y_t|y_{<t}, x)
$$

其中，$P_{\theta}(y_t|y_{<t}, x)$ 是模型在给定上下文$x$和之前生成的序列 $y_{<t}$ 下，生成下一个token $y_t$的概率。

RL Loss基本形式:

$$
J(\pi_{\theta}) = \mathbb{E}[ \sum_{t=1}^{\infty} \gamma^{t} R(s_t, a_t)]
$$

其中，$\pi_{\theta}$是参数为$\theta$的策略，$s$是从策略$\pi$的状态分布$\rho_{\pi}$中采样的状态，$a$是从策略$\pi_{\theta}$中采样的动作，$R(s, a)$是奖励函数。

价值函数相关定义:

$$
V(s) = \mathbb{E}_{a \sim \pi_{\theta}, s' \sim P}[R(s, a) + \gamma V(s')]
$$

$$
Q(s, a) = R(s, a) + \gamma \mathbb{E}_{s' \sim P}[V(s')]
$$

$$
A(s, a) = Q(s, a) - V(s)
$$

其中，$V(s)$是状态价值函数，$Q(s, a)$是状态-动作价值函数，$A(s, a)$是优势函数，$\gamma$是折扣因子，$P$是环境动态。

PPO (近端策略优化) Loss:

$$
L_{\text{PPO}} = \mathbb{E}_{s,a}[\min(r_t(\theta)A^{\pi_{\text{old}}}(s,a), \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A^{\pi_{\text{old}}}(s,a))]
$$

其中，$r_t(\theta) = \frac{\pi_{\theta}(a|s)}{\pi_{\text{old}}(a|s)}$是概率比率，$A^{\pi_{\text{old}}}$是在旧策略下计算的优势值，$\epsilon$是裁剪参数。

Reward Model Loss (奖励模型损失):

$$
L_{\text{RM}} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}[\log(\sigma(r_{\phi}(x,y_w) - r_{\phi}(x,y_l)))]
$$

其中，$r_{\phi}$是奖励模型，$\phi$是模型参数，$(x,y_w,y_l)$是输入与偏好对比数据，$y_w$是优先选择的回答，$y_l$是较差的回答，$\sigma$是sigmoid函数。


DPO (直接偏好优化) Loss:

$$
L_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_{\theta}(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

