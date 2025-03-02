import torch
import torch.nn as nn

class PositionWiseFFN(nn.Module):
    def __init__(self, out_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(out_features=out_features),
            nn.SiLU(),
            nn.LazyLinear(out_features=out_features)
        )

    def forward(self, X: torch.Tensor):
        return self.mlp(X)