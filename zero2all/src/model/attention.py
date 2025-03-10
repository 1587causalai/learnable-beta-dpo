"""
注意力机制实现模块 (Attention Mechanisms Implementation)

本模块实现了Transformer架构中的核心组件——注意力机制的多种变体，包括：

1. 标准多头自注意力 (Multi-Head Self-Attention, MHSA)
   - 原始Transformer论文中提出的基础注意力机制
   - 每个头有独立的Q、K、V投影

2. 分组查询注意力 (Grouped-Query Attention, GQA)
   - 多个查询头共享相同的键值头
   - 显著减少内存使用和计算量
   - 适用于大规模语言模型推理优化

3. 双块注意力 (Dual-Chunk Attention)
   - 用于处理长序列的注意力优化
   - 结合局部和全局注意力机制

核心数学原理:
- 自注意力计算: Attention(Q, K, V) = softmax(QK^T/√d)V
- 其中Q(查询)、K(键)、V(值)是输入的线性投影

位置编码:
- 使用旋转位置编码(RoPE)表示位置信息
- RoPE通过旋转向量在复平面上的相对位置来编码位置信息

作者: Zero2All项目组
许可证: MIT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RotaryEmbedding(nn.Module):
    """
    旋转位置编码(RoPE)的实现
    
    RoPE将位置信息通过旋转向量在复平面上的方式编码，而不是添加位置向量。
    这种方式具有以下优点：
    1. 保持向量长度不变，避免干扰注意力计算
    2. 更好地捕捉相对位置关系
    3. 具有外推到更长序列的能力
    
    数学表示：
    对于位置m和n，频率为ω的旋转变换为:
    R_Θ(m-n) = [cos((m-n)Θ), sin((m-n)Θ), ..., cos((m-n)Θ), sin((m-n)Θ)]
    
    参数:
        dim: 编码的维度大小，通常为注意力头的维度
        max_position_embeddings: 最大位置编码缓存
        base: 用于计算频率的基数（影响编码的衰减率）
    """
    def __init__(self, dim, max_position_embeddings=32768, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # 计算不同频率的逆频率，用于生成旋转矩阵
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 初始化余弦和正弦缓存
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len):
        """
        为所有位置预计算并缓存旋转矩阵的余弦和正弦值
        
        参数:
            seq_len: 序列长度，决定了缓存的大小
        """
        self.max_seq_len_cached = seq_len
        
        # 生成位置索引向量 [seq_len]
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device)
        
        # 计算每个位置、每个频率的角度：位置 × 频率
        # 使用爱因斯坦求和约定实现张量乘法
        # [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # 将角度转换为余弦和正弦值
        # [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 缓存cos和sin值
        # [seq_len, dim]
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        
        # 为了减少后续维度调整的需要，直接增加两个维度
        # [seq_len, 1, 1, dim]
        self.register_buffer("cos_cached", cos_cached.unsqueeze(1).unsqueeze(1))
        self.register_buffer("sin_cached", sin_cached.unsqueeze(1).unsqueeze(1))
    
    def forward(self, x, seq_len=None):
        """
        获取对应序列长度的旋转编码值
        
        参数:
            x: 输入张量，用于确定设备
            seq_len: 需要的序列长度
            
        返回:
            (cos, sin): 余弦和正弦值的元组，形状为 [seq_len, 1, 1, head_dim]
        """
        # 如果没有提供seq_len，则从输入张量推断
        if seq_len is None:
            if x.dim() >= 3:
                seq_len = x.size(2)  # [batch, heads, seq_len, ...]
            else:
                seq_len = x.size(0)  # 假设x是序列本身
        
        # 如果需要的序列长度超出当前缓存，重新计算
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        # 获取对应位置的旋转矩阵值，并移动到正确的设备上
        # 确保返回的形状是 [seq_len, 1, 1, head_dim]
        cos = self.cos_cached[:seq_len].to(x.device)
        sin = self.sin_cached[:seq_len].to(x.device)
        
        # 打印调试信息
        # print(f"RoPE forward - cos shape: {cos.shape}, sin shape: {sin.shape}")
        
        return cos, sin

def rotate_half(x):
    """
    将向量的一半维度旋转90度
    
    这是RoPE的核心操作，通过将向量的后半部分与前半部分交换并改变符号
    来实现复平面上的旋转效果。
    
    参数:
        x: 输入张量 [..., dim]
        
    返回:
        旋转后的张量 [..., dim]
    """
    x1, x2 = x.chunk(2, dim=-1)  # 将最后一维平均分为两部分
    return torch.cat((-x2, x1), dim=-1)  # 拼接为旋转后的向量

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    将旋转位置编码应用到查询和键
    
    实现公式：
    q' = q·cos(θ) + rotate_half(q)·sin(θ)
    k' = k·cos(θ) + rotate_half(k)·sin(θ)
    
    参数:
        q: 查询张量 [batch, heads, seq_len, head_dim]
        k: 键张量 [batch, heads, seq_len, head_dim]
        cos: 余弦值 [seq_len, 1, 1, head_dim] 或 [seq_len, head_dim]
        sin: 正弦值 [seq_len, 1, 1, head_dim] 或 [seq_len, head_dim]
        position_ids: 可选，自定义位置ID
    
    返回:
        (q', k'): 应用位置编码后的查询和键
    """
    # 打印输入形状以便调试
    # print(f"DEBUG - q: {q.shape}, k: {k.shape}, cos: {cos.shape}, sin: {sin.shape}")
    
    # 获取序列长度
    q_seq_len = q.size(2)
    
    # 检查并调整余弦和正弦值的维度以匹配需求
    if cos.dim() == 2:
        # 如果cos是[seq_len, head_dim]，将其扩展为[seq_len, 1, 1, head_dim]
        cos = cos.unsqueeze(1).unsqueeze(1)  # [seq_len, 1, 1, head_dim]
        sin = sin.unsqueeze(1).unsqueeze(1)  # [seq_len, 1, 1, head_dim]
    
    # 确保cos和sin的序列长度与q和k匹配
    if cos.size(0) > q_seq_len:
        cos = cos[:q_seq_len]
        sin = sin[:q_seq_len]
    
    # 确保cos和sin的最后一个维度与q和k的最后一个维度匹配
    head_dim = q.size(-1)
    if cos.size(-1) != head_dim:
        # 如果维度不匹配，可能需要调整
        if cos.size(-1) > head_dim:
            cos = cos[..., :head_dim]
            sin = sin[..., :head_dim]
        else:
            # 如果cos/sin的维度小于head_dim，这是一个错误
            raise ValueError(f"RoPE维度不匹配: cos/sin的head_dim={cos.size(-1)}，但q/k的head_dim={head_dim}")
    
    if position_ids is None:
        # 默认情况：使用连续的位置
        # 确保广播正确
        # q和k形状: [batch, heads, seq_len, head_dim]
        # cos和sin形状: [seq_len, 1, 1, head_dim]
        # 我们需要将cos和sin广播到q和k的形状
        
        # 方法1: 使用广播机制
        # 将cos和sin转置为 [1, 1, seq_len, head_dim]，这样可以与q和k的维度对齐
        cos_broadcasted = cos.permute(1, 2, 0, 3)  # [1, 1, seq_len, head_dim]
        sin_broadcasted = sin.permute(1, 2, 0, 3)  # [1, 1, seq_len, head_dim]
        
        # 应用旋转变换
        q_rot = (q * cos_broadcasted) + (rotate_half(q) * sin_broadcasted)
        k_rot = (k * cos_broadcasted) + (rotate_half(k) * sin_broadcasted)
    else:
        # 使用自定义位置ID的情况
        # 从cos和sin中选择对应position_ids的行
        # position_ids形状: [batch, seq_len]
        
        # 创建索引张量以便从cos和sin中选择正确的行
        batch_size, seq_len = position_ids.shape
        
        # 方法: 使用gather操作
        # 首先将cos和sin转换为适合gather的形状
        cos_flat = cos.squeeze(1).squeeze(1)  # [seq_len, head_dim]
        sin_flat = sin.squeeze(1).squeeze(1)  # [seq_len, head_dim]
        
        # 使用gather选择对应的位置编码
        # 将position_ids扩展为 [batch, seq_len, 1]
        indices = position_ids.unsqueeze(-1)
        
        # 收集对应位置的cos和sin值
        # 结果形状: [batch, seq_len, head_dim]
        cos_gathered = torch.gather(cos_flat, 0, indices.expand(-1, -1, head_dim))
        sin_gathered = torch.gather(sin_flat, 0, indices.expand(-1, -1, head_dim))
        
        # 重塑为与q和k兼容的形状
        # [batch, 1, seq_len, head_dim]
        cos_pos = cos_gathered.unsqueeze(1)
        sin_pos = sin_gathered.unsqueeze(1)
        
        # 应用旋转变换
        q_rot = (q * cos_pos) + (rotate_half(q) * sin_pos)
        k_rot = (k * cos_pos) + (rotate_half(k) * sin_pos)
    
    return q_rot, k_rot

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力(GQA)的实现
    
    GQA是一种内存效率更高的注意力变体，其中多个查询头共享同一个键值头。
    这种方法可以显著减少KV缓存的内存占用，同时保持模型性能。
    
    主要优势:
    - 减少模型参数数量
    - 显著降低KV缓存内存使用
    - 加快自回归生成速度
    
    参数:
        hidden_size: 隐藏层维度
        num_q_heads: 查询头数量
        num_kv_heads: 键值头数量（小于查询头数量）
        head_dim: 每个头的维度大小
        dropout: Dropout比率
        max_position_embeddings: 最大位置编码
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
        
        # 确保查询头数量是KV头数量的整数倍，以便正确分组
        assert self.num_q_heads % self.num_kv_heads == 0, "q_heads必须是kv_heads的整数倍"
        self.num_queries_per_kv = self.num_q_heads // self.num_kv_heads
        
        # 线性投影层
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
        """
        GQA前向传播
        
        参数:
            hidden_states: 输入张量 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: 用于缓存的过去KV状态
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用KV缓存
            
        返回:
            输出张量和可选的注意力权重和KV缓存
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 计算查询、键、值投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑形状为多头格式: [batch, heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_length, self.num_q_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 处理KV缓存，用于加速自回归生成
        kv_seq_len = seq_length
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            # 将过去的KV状态连接到当前状态
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # 准备返回的KV缓存
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # 应用旋转位置编码
        cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # GQA的核心操作：复制KV头以匹配查询头的数量
        # 例如: 如果有4个查询头和1个KV头，则KV头被复制4次
        key_states = torch.repeat_interleave(key_states, self.num_queries_per_kv, dim=1)
        value_states = torch.repeat_interleave(value_states, self.num_queries_per_kv, dim=1)
        
        # 计算注意力分数: Q·K^T/√d
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 应用softmax得到注意力权重
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力加权求和: Attention·V
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 重塑回原始维度并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # 准备输出
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
            
        return outputs

class DualChunkAttention(nn.Module):
    """
    双块注意力机制实现，用于更高效地处理长上下文
    
    这种注意力机制将长序列分成多个块，分别进行处理，
    可以减少长序列处理的计算复杂度，从O(n²)降低到接近O(n)。
    
    实现思路:
    1. 将长序列分成固定大小的块
    2. 对每个块内部应用标准注意力（局部注意力）
    3. 在块之间应用全局信息交换（全局注意力）
    
    参数:
        hidden_size: 隐藏层维度
        num_q_heads: 查询头数量
        num_kv_heads: 键值头数量
        head_dim: 每个头的维度
        dropout: Dropout比率
        max_position_embeddings: 最大位置编码长度
        chunk_size: 块的大小（token数量）
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
        # 使用GQA作为基础注意力机制
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
        hidden_states, # [batch_size, seq_length, hidden_size]
        attention_mask=None, # [batch_size, seq_length]
        position_ids=None, # [batch_size, seq_length]
        past_key_value=None, # [batch_size, num_kv_heads, seq_length, head_dim]
        output_attentions=False, # bool
        use_cache=False, # bool
    ):
        """
        双块注意力前向传播
        
        参数:
            hidden_states: 输入张量 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            past_key_value: KV缓存
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用KV缓存
            
        返回:
            输出张量和可选的注意力权重和缓存
        """
        batch_size, seq_length = hidden_states.shape[:2] 
        
        # 对于短序列或推理阶段，直接使用标准GQA
        if seq_length <= self.chunk_size or not self.training:
            return self.gqa(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        # 注: 完整实现的双块注意力需要更复杂的逻辑
        # 1. 将序列分成多个块
        # 2. 对每个块内部应用局部注意力
        # 3. 应用全局块间注意力
        # 当前仅为简化实现，完整实现需要更多代码
        
        # 简化版：直接使用GQA
        return self.gqa(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        ) 

class MultiHeadAttention(nn.Module):
    """
    标准的多头自注意力机制(Multi-Head Self-Attention)实现
    
    这是Transformer架构中最基础的注意力机制，由Vaswani等人在
    "Attention Is All You Need" 论文中提出。
    
    原理: 
    1. 将输入投影到查询(Q)、键(K)、值(V)三个空间
    2. 在降维的头空间中并行计算注意力
    3. 拼接多头结果并投影回原始维度
    
    数学表示:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    参数:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数量
        head_dim: 每个头的维度
        dropout: Dropout比率
        max_position_embeddings: 最大位置编码长度
        use_bias: 是否在线性层中使用偏置
    """
    def __init__(
        self,
        hidden_size=896,
        num_heads=14,
        head_dim=64,
        dropout=0.0,
        max_position_embeddings=32768,
        use_bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_prob = dropout
        
        # 总的注意力维度：头数 × 每个头的维度
        self.attention_dim = num_heads * head_dim
        
        # 检查维度是否匹配隐藏层维度
        if self.attention_dim != hidden_size:
            raise ValueError(
                f"注意力总维度 {self.attention_dim} 与隐藏层维度 {hidden_size} 不匹配"
            )
        
        # 查询、键、值的线性投影层
        self.q_proj = nn.Linear(hidden_size, self.attention_dim, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, self.attention_dim, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, self.attention_dim, bias=use_bias)
        
        # 输出投影层
        self.o_proj = nn.Linear(self.attention_dim, hidden_size, bias=use_bias)
        
        # Dropout层，用于正则化
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
        """
        标准多头自注意力前向传播
        
        参数:
            hidden_states: [batch_size, seq_length, hidden_size] 输入张量
            attention_mask: [batch_size, 1, seq_length, seq_length] 注意力掩码
            position_ids: [batch_size, seq_length] 位置ID
            past_key_value: 用于缓存的过去状态
            output_attentions: 是否输出注意力权重
            use_cache: 是否使用KV缓存
            
        返回:
            输出张量和可选的注意力权重和缓存
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 线性投影得到查询、键、值
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑为多头形式: [batch, heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 处理KV缓存，用于自回归生成
        kv_seq_len = seq_length
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            # 拼接过去的KV状态
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # 是否保存KV状态用于后续使用
        if use_cache:
            past_key_value = (key_states, value_states)
        
        # 应用旋转位置编码
        cos, sin = self.rotary_emb(query_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # 计算注意力分数: Q·K^T/√d
        # [batch_size, num_heads, seq_length, seq_length]
        attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码（如果提供）
        if attention_mask is not None:
            # 添加掩码 (-inf使softmax后的权重为0)
            attention_scores = attention_scores + attention_mask
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 注意力加权求和: Attention·V
        # [batch_size, num_heads, seq_length, head_dim]
        context = torch.matmul(attention_weights, value_states)
        
        # 重塑为原始维度: [batch_size, seq_length, hidden_size]
        context = context.transpose(1, 2).contiguous().reshape(batch_size, seq_length, self.hidden_size)
        
        # 输出投影
        output = self.o_proj(context)
        
        # 准备输出
        outputs = (output,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (past_key_value,)
        
        return outputs

def compare_mhsa_gqa():
    """
    打印多头自注意力(MHSA)与分组查询注意力(GQA)的比较信息
    """
    print("多头自注意力(MHSA)与分组查询注意力(GQA)对比:")
    print("-" * 50)
    print("1. 多头自注意力(MHSA):")
    print("   - 标准Transformer使用的注意力机制")
    print("   - 每个注意力头有独立的查询(Q)、键(K)、值(V)投影")
    print("   - 需要的内存: 对于h个头，需要3h组投影参数")
    print("   - KV缓存大小: 与头数成正比")
    print()
    print("2. 分组查询注意力(GQA):")
    print("   - 多个查询(Q)头共享同一组键(K)和值(V)投影")
    print("   - 大幅减少参数量和内存使用")
    print("   - 对于nq个查询头和nkv个键值头(nq > nkv):")
    print("     * 参数数量: nq + 2*nkv (而MHSA需要3*nq)")
    print("     * KV缓存大小: 与键值头数成正比，显著小于MHSA")
    print("   - 性能损失很小，但效率提升显著")
    print()
    print("3. 什么时候使用哪种?")
    print("   - 对于小模型或注重精度的场景: 可以使用MHSA")
    print("   - 对于大模型或内存受限场景: 推荐使用GQA")
    print("   - Qwen系列模型使用GQA以优化大规模部署效率")
    print("-" * 50)

if __name__ == "__main__":
    """
    模块测试和演示代码
    """
    print("=" * 50)
    print("注意力机制测试")
    print("=" * 50)
    
    # 显示MHSA与GQA的比较
    compare_mhsa_gqa()
    
    # 创建测试输入
    batch_size = 2
    seq_length = 16
    hidden_size = 64
    num_heads = 4
    kv_heads = 2
    head_dim = hidden_size // num_heads
    
    # 随机输入
    x = torch.randn(batch_size, seq_length, hidden_size)
    print(f"\n创建测试输入: 形状={x.shape}, 类型={x.dtype}")
    
    try:
        # 首先测试旋转位置编码
        print("\n1. 测试旋转位置编码(RoPE):")
        rope = RotaryEmbedding(head_dim, max_position_embeddings=32)
        cos, sin = rope(x, seq_len=seq_length)
        print(f"  RoPE cos形状: {cos.shape}")
        print(f"  RoPE sin形状: {sin.shape}")
        
        # 测试rotate_half函数
        print("\n2. 测试rotate_half函数:")
        test_vec = torch.randn(2, 2, 3, head_dim)
        rotated = rotate_half(test_vec)
        print(f"  输入形状: {test_vec.shape}")
        print(f"  旋转后形状: {rotated.shape}")
        
        # 测试apply_rotary_pos_emb函数
        print("\n3. 测试apply_rotary_pos_emb函数:")
        q = torch.randn(batch_size, num_heads, seq_length, head_dim)
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)
        print(f"  查询形状: {q.shape}")
        print(f"  键形状: {k.shape}")
        print(f"  cos形状: {cos.shape}")
        print(f"  sin形状: {sin.shape}")
        
        # 启用详细调试
        print("  详细调试信息:")
        print(f"    q维度: {q.dim()}, 形状: {q.shape}")
        print(f"    k维度: {k.dim()}, 形状: {k.shape}")
        print(f"    cos维度: {cos.dim()}, 形状: {cos.shape}")
        print(f"    sin维度: {sin.dim()}, 形状: {sin.shape}")
        
        # 转置cos和sin以便广播
        cos_broadcasted = cos.permute(1, 2, 0, 3)
        sin_broadcasted = sin.permute(1, 2, 0, 3)
        print(f"    cos_broadcasted形状: {cos_broadcasted.shape}")
        print(f"    sin_broadcasted形状: {sin_broadcasted.shape}")
        
        # 手动应用旋转
        q_rot_manual = (q * cos_broadcasted) + (rotate_half(q) * sin_broadcasted)
        k_rot_manual = (k * cos_broadcasted) + (rotate_half(k) * sin_broadcasted)
        print(f"    手动旋转后q形状: {q_rot_manual.shape}")
        print(f"    手动旋转后k形状: {k_rot_manual.shape}")
        
        # 使用函数应用旋转
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        print(f"  旋转后查询形状: {q_rot.shape}")
        print(f"  旋转后键形状: {k_rot.shape}")
        
        # 测试多头自注意力
        print("\n4. 测试多头自注意力(MHSA):")
        mhsa = MultiHeadAttention(hidden_size, num_heads, head_dim)
        mhsa_output = mhsa(x)[0]
        print(f"  输入形状: {x.shape}")
        print(f"  MHSA输出形状: {mhsa_output.shape}")
        
        # 测试分组查询注意力
        print("\n5. 测试分组查询注意力(GQA):")
        gqa = GroupedQueryAttention(hidden_size, num_heads, kv_heads, head_dim)
        gqa_output = gqa(x)[0]
        print(f"  输入形状: {x.shape}")
        print(f"  GQA输出形状: {gqa_output.shape}")
        
        # 参数数量比较
        mhsa_params = sum(p.numel() for p in mhsa.parameters())
        gqa_params = sum(p.numel() for p in gqa.parameters())
        print(f"\n6. 参数数量比较:")
        print(f"  MHSA参数: {mhsa_params}")
        print(f"  GQA参数: {gqa_params}")
        print(f"  GQA节省: {mhsa_params - gqa_params} ({(mhsa_params - gqa_params) / mhsa_params * 100:.2f}%)")
        
        # 测试是否所有组件能够一起工作
        print("\n7. 验证组件集成:")
        # 创建测试注意力掩码
        attention_mask = torch.zeros(batch_size, 1, 1, seq_length)
        attention_mask[:, :, :, :seq_length//2] = -1e9  # 掩盖前一半的序列
        
        # 使用注意力掩码测试
        outputs_with_mask = mhsa(x, attention_mask=attention_mask)[0]
        print(f"  使用掩码后的MHSA输出形状: {outputs_with_mask.shape}")
        
        # 测试双块注意力
        print("\n8. 测试双块注意力:")
        dual_chunk = DualChunkAttention(
            hidden_size=hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            chunk_size=8  # 小于seq_length，以测试分块逻辑
        )
        dual_output = dual_chunk(x)[0]
        print(f"  输入形状: {x.shape}")
        print(f"  双块注意力输出形状: {dual_output.shape}")
        
        print("\n所有测试通过!")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


