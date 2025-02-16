# SFT 和 DPO Loss 组合训练提案

## 背景

在 DPO (Direct Preference Optimization) 训练中，一种常见的做法是将 SFT (Supervised Fine-Tuning) Loss 和 DPO Loss 结合起来，通过加权求和的方式优化模型。这种方法可能有助于在保持基础生成能力的同时，更好地学习人类偏好。

## 设计目标

1. 在现有 DPO 框架基础上，添加 SFT Loss 支持
2. 通过可调节的权重系数平衡两种 loss
3. 保持代码的可维护性和可扩展性
4. 提供充分的监控和评估能力

## 技术方案

### 1. Loss 设计

结合两种 loss 的数学形式：

```
L_total = α * L_sft + (1-α) * L_dpo

其中：
- L_sft = -log P(chosen|query)                                    # 标准交叉熵损失
- L_dpo = -log(sigmoid(β(x) * (r_chosen - r_rejected)))         # DPO loss
- α ∈ [0,1] 为权重系数
```

### 2. 数据结构设计

```python
class CombinedDataset(Dataset):
    def __init__(
        self,
        queries: List[str],          # 查询/提示词
        chosen_responses: List[str],  # 选中的回答
        rejected_responses: List[str], # 被拒绝的回答
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        # 数据验证和初始化
        ...
        
    def __getitem__(self, idx):
        # 返回用于 SFT 和 DPO 的数据
        return {
            "sft_input_ids": ...,       # 用于 SFT 训练
            "sft_attention_mask": ...,
            "chosen_input_ids": ...,     # 用于 DPO 训练
            "chosen_attention_mask": ...,
            "rejected_input_ids": ...,
            "rejected_attention_mask": ...,
        }
```

### 3. 模型设计

```python
class CombinedDPOModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        beta_head: nn.Module,
        tokenizer: PreTrainedTokenizer,
        sft_weight: float = 0.5,
    ):
        # 模型初始化
        ...
        
    def forward(self, ...):
        # 1. 计算 SFT Loss
        sft_loss = self._compute_sft_loss(...)
        
        # 2. 计算 DPO Loss
        dpo_loss = self._compute_dpo_loss(...)
        
        # 3. 组合 Loss
        total_loss = self.sft_weight * sft_loss + (1 - self.sft_weight) * dpo_loss
        
        return {
            "loss": total_loss,
            "sft_loss": sft_loss,
            "dpo_loss": dpo_loss,
            ...
        }
```

## 实现考虑

### 1. 性能影响
- 需要同时计算三个序列的前向传播
- 可能需要更大的 batch size
- 内存使用会增加

### 2. 训练策略
- 可以实现动态权重调整
- 初期可以给 SFT loss 更大权重
- 后期逐渐增加 DPO loss 权重

### 3. 监控指标
- 分别跟踪 SFT loss 和 DPO loss
- 监控 beta 值的变化
- 评估生成质量和偏好对齐程度

## 潜在风险

1. **复杂度增加**：
   - 代码复杂度显著提高
   - 维护成本增加
   - 调试难度增加

2. **训练不稳定**：
   - 两种 loss 可能相互干扰
   - 需要更多的超参数调优
   - 可能需要更长的训练时间

3. **内存压力**：
   - 每个样本需要更多的内存
   - 可能需要减小 batch size
   - 训练速度可能降低

## 建议

考虑到项目当前的状态和复杂度，建议：

1. 保持当前的纯 DPO 实现作为主分支
2. 在单独的实验分支中尝试 SFT+DPO 组合
3. 通过实验对比两种方法的效果
4. 根据实验结果决定是否合并到主分支

## 后续工作

如果决定实施这个提案，建议按以下步骤进行：

1. 创建实验分支
2. 实现基础版本的组合训练
3. 进行充分的测试和实验
4. 评估效果并决定是否采用

## 参考资料

- DPO 原始论文
- 相关工作中的实践经验
- 社区讨论和反馈 