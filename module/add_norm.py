import torch
import torch.nn as nn

class AddAndNorm(nn.Module):
    def __init__(self, norm_dim: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape=norm_dim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        return self.layer_norm(self.dropout(Y) + X)