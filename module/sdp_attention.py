import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        