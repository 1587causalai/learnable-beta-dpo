import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaHead(nn.Module):
    """BetaHead网络，用于动态计算beta值
    
    计算公式：β(x) = w * PPL(x) * f(x)
    其中 f(x) = 1 + ε * tanh(NN(x))
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        epsilon: float = 0.1,
        init_weight: float = 1.0,
        head_type: str = "linear"
    ):
        super().__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.tensor(init_weight))
        self.input_dim = input_dim
        
        if head_type == "linear":
            self.nn = nn.Linear(input_dim, 1)
        elif head_type == "mlp":
            self.nn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
            
    def forward(self, context_embedding: torch.Tensor, ppl: torch.Tensor) -> torch.Tensor:
        """计算动态beta值
        
        Args:
            context_embedding: 上下文的向量表示 [batch_size, hidden_dim]
            ppl: 困惑度值 [batch_size]
            
        Returns:
            beta: 动态计算的beta值 [batch_size]
        """
        # 确保输入维度正确
        assert context_embedding.dim() == 2, f"Expected 2D tensor, got {context_embedding.dim()}D"
        assert ppl.dim() == 1, f"Expected 1D tensor, got {ppl.dim()}D"
        assert context_embedding.size(1) == self.input_dim, \
            f"Expected input dimension {self.input_dim}, got {context_embedding.size(1)}"
        assert context_embedding.size(0) == ppl.size(0), \
            f"Batch size mismatch: context_embedding {context_embedding.size(0)} vs ppl {ppl.size(0)}"
        
        # 计算f(x) = 1 + ε * tanh(NN(x))
        nn_output = self.nn(context_embedding).squeeze(-1)  # [batch_size]
        fx = 1 + self.epsilon * torch.tanh(nn_output)
        
        # 计算最终的beta值：β(x) = w * PPL(x) * f(x)
        beta = torch.abs(self.w) * ppl * fx
        
        return beta
