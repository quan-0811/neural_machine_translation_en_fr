import torch
import torch.nn as nn
from module.mh_attention import MultiHeadAttention
from module.add_norm import AddAndNorm
from module.pos_wise_ffn import PositionWiseFFN

class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_hiddens: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.masked_mha = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads, dropout_rate=dropout_rate)
        self.add_and_norm_1 = AddAndNorm(norm_dim=num_hiddens, dropout_rate=dropout_rate)
        self.mh_attention = MultiHeadAttention(num_hiddens=num_hiddens, num_heads=num_heads, dropout_rate=dropout_rate)
        self.add_and_norm_2 = AddAndNorm(norm_dim=num_hiddens, dropout_rate=dropout_rate)
        self.position_wise_ffn = PositionWiseFFN(out_features=num_hiddens)
        self.add_and_norm_3 = AddAndNorm(norm_dim=num_hiddens, dropout_rate=dropout_rate)

    def forward(self, X: torch.Tensor, encoder_output: torch.Tensor, valid_enc_lens: torch.Tensor, valid_dec_len: torch.Tensor):
        batch_size, num_steps, hidden_dim = X.shape

        if self.training:
            valid_dec_lens = torch.zeros(size=(batch_size, num_steps), dtype=torch.long)
            for i in range(batch_size):
                valid_dec_lens[i, :valid_dec_len[i]] = torch.arange(start=1, end=valid_dec_len[i].item()+1)
        else:
            valid_dec_lens = torch.tensor([num_steps] * batch_size)

        masked_mha_output = self.masked_mha(X, X, X, valid_dec_lens)
        add_norm_1_output = self.add_and_norm_1(X, masked_mha_output)

        mha_output = self.mh_attention(add_norm_1_output, encoder_output, encoder_output, valid_enc_lens)
        add_norm_2_output = self.add_and_norm_2(add_norm_1_output, mha_output)

        pos_wise_output = self.position_wise_ffn(add_norm_2_output)
        add_norm_3_output = self.add_and_norm_3(add_norm_2_output, pos_wise_output)

        return add_norm_3_output