import torch
import torch.nn as nn
from module.sdp_attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.W_q = nn.LazyLinear(num_hiddens)
        self.W_k = nn.LazyLinear(num_hiddens)
        self.W_v = nn.LazyLinear(num_hiddens)
        self.W_o = nn.LazyLinear(num_hiddens)
        self.attention = ScaledDotProductAttention(dropout_rate=dropout_rate)
        self.num_heads = num_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, valid_lens):

        # Pass q, k, v to FC layer in parallel and reshape for attention
        queries = self.reshape_for_attention(self.W_q(queries))
        keys = self.reshape_for_attention(self.W_k(keys))
        values = self.reshape_for_attention(self.W_v(values))

        # Repeat valid_lens since one single (q, k, v) are pass through multiple heads
        valid_lens = torch.repeat_interleave(valid_lens, self.num_heads, dim=0)

        # Attention pooling, output.shape = (batch_size * num_heads, num_queries, num_hidden // num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # Concatenate heads
        heads_concatenated = self.revert_reshape(output)

        return self.W_o(heads_concatenated)

    def reshape_for_attention(self, X: torch.Tensor):
        """
        Reshape queries, keys, values
        from (batch_size, num_queries or num_keys_values, num_hiddens)
        to (batch_size * num_heads, num_queries or num_keys_values, num_hiddens // num_heads)
        """
        # Change to (batch_size, num_heads, num_queries or num_keys_values, num_hiddens // num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)

        # Change to (batch_size * num_heads, num_queries or num_keys_values, num_hiddens // num_heads)
        X = X.reshape(-1, X.shape[2], X.shape[3])

        return X

    def revert_reshape(self, X: torch.Tensor):
        """
        Revert the changes made by reshape_for_attention()
        """
        # Change to (batch_size, num_heads, num_queries or num_keys_values, num_hiddens // num_heads)
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)

        # Change to (batch_size, num_queries or num_keys_values, num_hiddens)
        X = X.reshape(X.shape[0], X.shape[1], -1)

        return X
