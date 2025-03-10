import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention, DualChunkAttention
from .mlp import MLP


class TransformerLayer(nn.Module):
    """
    Transformer解码器层的实现
    包含注意力机制和前馈网络
    """
    def __init__(
        self,
        hidden_size=896,
        num_q_heads=14,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=4864,
        activation_type="swiglu",
        layer_norm_eps=1e-5,
        dropout=0.0,
        attention_dropout=0.0,
        use_dual_chunk=False,
        max_position_embeddings=32768,
    ):
        super().__init__()
        
        # 第一个层归一化
        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # 注意力层
        if use_dual_chunk:
            self.attention = DualChunkAttention(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dropout=attention_dropout,
                max_position_embeddings=max_position_embeddings,
            )
        else:
            self.attention = GroupedQueryAttention(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dropout=attention_dropout,
                max_position_embeddings=max_position_embeddings,
            )
        
        # 第二个层归一化
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # 前馈网络
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=activation_type,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        # 自注意力层
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        attn_output = attention_outputs[0]
        
        # 残差连接
        hidden_states = residual + self.dropout(attn_output)
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        feed_forward_output = self.mlp(hidden_states)
        
        # 残差连接
        hidden_states = residual + self.dropout(feed_forward_output)
        
        outputs = (hidden_states,)
        
        # 可选输出
        if output_attentions:
            outputs += (attention_outputs[1],)
        
        if use_cache:
            outputs += (attention_outputs[-1],)
        
        return outputs


class QwenTransformer(nn.Module):
    """
    类似于Qwen-0.5B的Transformer模型实现
    """
    def __init__(
        self,
        vocab_size=151646,
        hidden_size=896,
        num_hidden_layers=24,
        num_q_heads=14,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=4864,
        activation_type="swiglu",
        max_position_embeddings=32768,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
        dropout=0.0,
        attention_dropout=0.0,
        use_dual_chunk=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        initializer_range=0.02,
        use_cache=True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.activation_type = activation_type
        self.max_position_embeddings = max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.use_dual_chunk = use_dual_chunk
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                activation_type=activation_type,
                layer_norm_eps=layer_norm_eps,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_dual_chunk=use_dual_chunk,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # 输出层
        if tie_word_embeddings:
            self.lm_head = lambda x: F.linear(x, self.embedding.weight)
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            # 线性层使用截断正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用截断正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # 层归一化使用全1初始化
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        use_cache=None,
    ):
        """
        前向传播实现
        """
        # 处理use_cache参数
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # 处理输入
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("不能同时提供input_ids和inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("必须提供input_ids或inputs_embeds")
        
        # 初始化past_key_values
        if past_key_values is None:
            past_key_values = tuple([None] * self.num_hidden_layers)
        
        # 获取输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        
        # 初始化注意力掩码
        if attention_mask is not None:
            # 创建因果掩码
            attention_mask = self._prepare_attention_mask(attention_mask, seq_length)
        
        # 初始化位置ID
        if position_ids is None:
            # 创建默认位置ID
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        next_cache = () if use_cache else None
        
        # 通过所有Transformer层
        for i, (layer, past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_cache += (layer_outputs[-1],)
            
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        # 最终层归一化
        hidden_states = self.final_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # 计算语言模型头部输出
        logits = self.lm_head(hidden_states)
        
        # 构建返回结果
        if not return_dict:
            outputs = (logits, all_hidden_states, all_self_attentions, next_cache)
            return outputs
        
        # 假设我们有一个特定的返回字典类，这里简化为普通字典
        return {
            "logits": logits,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "past_key_values": next_cache,
        }
    
    def _prepare_attention_mask(self, attention_mask, seq_length):
        """
        准备注意力掩码
        """
        if attention_mask.dim() == 2:
            # 批处理的注意力掩码 [batch_size, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            # 扩展的注意力掩码 [batch_size, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError(f"注意力掩码的维度应该为2或3，得到 {attention_mask.dim()}")
        
        # 创建因果掩码（上三角为-inf）
        device = attention_mask.device
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=device) * float("-inf"),
            diagonal=1,
        )
        
        # 将注意力掩码与因果掩码结合
        extended_attention_mask = extended_attention_mask + causal_mask
        
        return extended_attention_mask
    
    def generate(
        self,
        input_ids,
        max_length=100,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        pad_token_id=None,
        eos_token_id=None,
    ):
        """
        生成文本的简单实现
        """
        pad_token_id = pad_token_id if pad_token_id is not None else self.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        
        # 如果输入是一维的，增加批处理维度
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        past_key_values = None
        
        # 如果当前长度已经达到最大长度，直接返回
        if cur_len >= max_length:
            return input_ids
        
        # 初始化注意力掩码
        attention_mask = torch.ones(batch_size, cur_len, device=input_ids.device)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        while cur_len < max_length:
            # 前向传播
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1].unsqueeze(-1),
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            # 获取下一个token的概率分布
            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 应用top_k筛选
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k, dim=-1)[0][:, -1].unsqueeze(-1) <= next_token_logits
                next_token_logits[indices_to_remove] = float("-inf")
            
            # 应用top_p筛选
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累计超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个token超过阈值
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将排序后的索引映射回原始索引
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float("-inf")
            
            # 采样或贪婪搜索
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 对已完成的序列设置为pad_token_id
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # 将新token添加到input_ids中
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新注意力掩码
            attention_mask = torch.cat([attention_mask, unfinished_sequences.unsqueeze(-1)], dim=-1)
            
            # 更新未完成序列标记
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            
            # 如果所有序列都已完成，提前结束
            if unfinished_sequences.max() == 0:
                break
            
            cur_len = cur_len + 1
        
        return input_ids 