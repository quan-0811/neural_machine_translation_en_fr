import torch
import torch.nn as nn
import math
from utils import masked_softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, valid_lens):
        """
        Attention pooling for q, k, v with:
            Shape of queries: (batch_size, no. of queries, d)
            Shape of keys: (batch_size, no. of key-value pairs, d)
            Shape of values: (batch_size, no. of key-value pairs, value dimension)
            Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        """

        d = queries.shape[-1]

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(attention_scores, valid_lens)

        return torch.bmm(self.dropout(attention_weights), values)
