import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, Optional, Tuple, Union

class DynamicBetaDPOModel(nn.Module):
    """动态Beta DPO模型
    
    结合了基础语言模型和BetaHead网络，实现动态beta DPO训练
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        beta_head: nn.Module,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        super().__init__()
        self.base_model = base_model
        self.beta_head = beta_head
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = next(self.base_model.parameters()).device
        
    def get_dynamic_beta(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算动态beta值
        
        Args:
            input_ids: 输入token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            beta: 动态计算的beta值 [batch_size]
            ppl: 计算的困惑度值 [batch_size]
        """
        # 获取最后一个token的hidden state作为context embedding
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 从outputs中获取hidden_states
        if hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
        else:
            hidden_states = outputs["hidden_states"]
            
        last_hidden = hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        context_embedding = last_hidden[:, -1, :]  # [batch_size, hidden_dim]
        
        # 计算困惑度
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs["logits"]
            
        logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
        labels = input_ids[:, 1:]  # [batch_size, seq_len-1]
        
        # 计算每个位置的loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction='none'
        ).reshape_as(labels)
        
        # 根据attention mask计算有效token的平均loss
        mask = attention_mask[:, 1:].float()
        avg_loss = (loss * mask).sum(dim=1) / mask.sum(dim=1)
        ppl = torch.exp(avg_loss)
        
        # 使用beta head计算动态beta值
        beta = self.beta_head(context_embedding, ppl)
        
        return beta, ppl
        
    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """DPO前向传播
        
        Args:
            chosen_input_ids: 选择的回答的input ids
            chosen_attention_mask: 选择的回答的attention mask
            rejected_input_ids: 拒绝的回答的input ids
            rejected_attention_mask: 拒绝的回答的attention mask
            
        Returns:
            包含loss和其他指标的字典
        """
        # 获取动态beta值
        beta_chosen, ppl_chosen = self.get_dynamic_beta(
            chosen_input_ids,
            chosen_attention_mask
        )
        beta_rejected, ppl_rejected = self.get_dynamic_beta(
            rejected_input_ids,
            rejected_attention_mask
        )
        
        # 计算chosen和rejected的logits
        chosen_outputs = self.base_model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        rejected_outputs = self.base_model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # 获取logits
        if hasattr(chosen_outputs, "logits"):
            chosen_logits = chosen_outputs.logits
            rejected_logits = rejected_outputs.logits
        else:
            chosen_logits = chosen_outputs["logits"]
            rejected_logits = rejected_outputs["logits"]
        
        # 计算policy loss
        chosen_log_probs = F.log_softmax(chosen_logits[:, :-1], dim=-1)
        rejected_log_probs = F.log_softmax(rejected_logits[:, :-1], dim=-1)
        
        chosen_labels = chosen_input_ids[:, 1:]
        rejected_labels = rejected_input_ids[:, 1:]
        
        chosen_log_probs = torch.gather(
            chosen_log_probs,
            dim=-1,
            index=chosen_labels.unsqueeze(-1)
        ).squeeze(-1)
        rejected_log_probs = torch.gather(
            rejected_log_probs,
            dim=-1,
            index=rejected_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 使用动态beta计算DPO loss
        chosen_mask = chosen_attention_mask[:, 1:].float()
        rejected_mask = rejected_attention_mask[:, 1:].float()
        
        chosen_rewards = (chosen_log_probs * chosen_mask).sum(dim=1) / chosen_mask.sum(dim=1)
        rejected_rewards = (rejected_log_probs * rejected_mask).sum(dim=1) / rejected_mask.sum(dim=1)
        
        # 计算每个样本的loss
        loss_per_sample = -F.logsigmoid(beta_chosen * (chosen_rewards - rejected_rewards))
        loss = loss_per_sample.mean()
        
        return {
            "loss": loss,
            "beta_chosen": beta_chosen.mean(),
            "beta_rejected": beta_rejected.mean(),
            "ppl_chosen": ppl_chosen.mean(),
            "ppl_rejected": ppl_rejected.mean(),
        }
