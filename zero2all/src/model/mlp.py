import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU激活函数的实现，类似于GLU但使用SwiSH而不是GELU
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class GeGLU(nn.Module):
    """
    GeGLU激活函数的实现，使用GELU作为门控机制
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.gelu(x1) * x2
        return self.w3(hidden)


class MLP(nn.Module):
    """
    多层感知机实现，支持不同的激活函数
    """
    def __init__(
        self, 
        hidden_size=896, 
        intermediate_size=4864, 
        activation_type="swiglu",
        dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_type = activation_type
        
        # 选择激活函数类型
        if activation_type == "swiglu":
            self.gate = SwiGLU(
                in_features=hidden_size,
                hidden_features=intermediate_size,
                out_features=hidden_size
            )
        elif activation_type == "geglu":
            self.gate = GeGLU(
                in_features=hidden_size,
                hidden_features=intermediate_size,
                out_features=hidden_size
            )
        else:
            # 传统的前馈网络实现
            self.fc1 = nn.Linear(hidden_size, intermediate_size)
            self.activation = F.gelu
            self.fc2 = nn.Linear(intermediate_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if hasattr(self, 'gate'):
            # 使用门控激活函数
            x = self.gate(x)
        else:
            # 传统的前馈网络
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
        
        return self.dropout(x) 