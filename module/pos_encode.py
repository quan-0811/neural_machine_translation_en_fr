import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, dropout_rate: float, num_steps_max=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.positional_encoding_matrix = torch.zeros(size=(1, num_steps_max, embedding_dim))
        X = torch.arange(num_steps_max, dtype=torch.float32).reshape(-1, 1)/torch.pow(10000, torch.arange(start=0, end=embedding_dim, step=2, dtype=torch.float32) / embedding_dim)
        self.positional_encoding_matrix[:,:, 0::2] = torch.sin(X)
        self.positional_encoding_matrix[:,:, 1::2] = torch.cos(X)

    def forward(self, X: torch.Tensor):
        X = X + self.positional_encoding_matrix[:, :X.shape[1], :]
        return self.dropout(X)