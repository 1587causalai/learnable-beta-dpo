import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码(RoPE)的实现
    """
    def __init__(self, dim, max_position_embeddings=32768, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 创建旋转角度缓存 - 修复重复注册问题
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        
        # 为所有位置计算角度
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # 为所有位置缓存旋转矩阵的cos和sin值
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x, seq_len=None):
        # 如果需要的序列长度超出当前缓存，重新计算
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
            
        # 获取对应位置的旋转矩阵值
        return (
            self.cos_cached[:seq_len].to(x.device),
            self.sin_cached[:seq_len].to(x.device),
        )

def rotate_half(x):
    """
    将向量的一半维度旋转
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    将旋转位置编码应用到查询和键
    """
    if position_ids is None:
        # 默认连续位置
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
    else:
        # 处理自定义位置ID
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
    return q, k

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力(GQA)的实现
    多个查询头共享同一个键值头
    """
    def __init__(
        self,
        hidden_size=896,
        num_q_heads=14,
        num_kv_heads=2,
        head_dim=64,
        dropout=0.0,
        max_position_embeddings=32768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # 确保查询头数量是KV头数量的整数倍
        assert self.num_q_heads % self.num_kv_heads == 0, "q_heads必须是kv_heads的整数倍"
        self.num_queries_per_kv = self.num_q_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化旋转位置编码
        self.rotary_emb = RotaryEmbedding(
            head_dim, 
            max_position_embeddings=max_position_embeddings
        )
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 计算查询、键、值投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑形状
        query_states = query_states.view(batch_size, seq_length, self.num_q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 获取KV缓存
        kv_seq_len = seq_length
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            # 将过去的KV状态连接到当前状态
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # 是否需要返回KV缓存
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # 应用旋转位置编码
        cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # 复制KV头以匹配查询头的数量
        key_states = torch.repeat_interleave(key_states, self.num_queries_per_kv, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_queries_per_kv, dim=1)
        
        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 归一化注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 重塑并投影回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs

class DualChunkAttention(nn.Module):
    """
    双块注意力机制实现，用于更高效地处理长上下文
    """
    def __init__(
        self,
        hidden_size=896,
        num_q_heads=14,
        num_kv_heads=2,
        head_dim=64,
        dropout=0.0,
        max_position_embeddings=32768,
        chunk_size=2048,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.gqa = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dropout=dropout,
            max_position_embeddings=max_position_embeddings,
        )
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 如果序列长度小于块大小，直接使用标准注意力
        if seq_length <= self.chunk_size or not self.training:
            return self.gqa(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        # 实现训练中的双块注意力逻辑
        # 1. 将序列分成多个块
        # 2. 对每个块内部应用局部注意力
        # 3. 应用全局块间注意力
        # 这里只是占位实现，完整实现需要更多代码
        
        # 这里简化为直接使用GQA
        return self.gqa(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        ) 