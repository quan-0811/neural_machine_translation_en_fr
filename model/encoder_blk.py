import torch
import torch.nn as nn
from module.mh_attention import MultiHeadAttention
from module.add_norm import AddAndNorm
from module.pos_wise_ffn import PositionWiseFFN

class TransformerEncoderBlock(nn.Module):
    def  __init__(self, num_hiddens: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.mh_attention = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads, dropout_rate=dropout_rate)
        self.add_and_norm_1 = AddAndNorm(norm_dim=num_hiddens, dropout_rate=dropout_rate)
        self.position_wise_ffn = PositionWiseFFN(out_features=num_hiddens)
        self.add_and_norm_2 = AddAndNorm(norm_dim=num_hiddens, dropout_rate=dropout_rate)

    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor):
        mha_output = self.mh_attention(X, X, X, valid_lens)
        add_norm_1_output = self.add_and_norm_1(X, mha_output)
        ffn_output = self.position_wise_ffn(add_norm_1_output)
        add_norm_2_output = self.add_and_norm_2(add_norm_1_output, ffn_output)

        return add_norm_2_output

