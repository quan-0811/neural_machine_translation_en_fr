import torch
import torch.nn as nn
from model.decoder import TransformerDecoder
from model.encoder import TransformerEncoder

class Transformers(nn.Module):
    def __init__(self, num_blks: int, num_hiddens: int, num_heads: int, dropout_rate: float, eng_vocab_size: int, fra_vocab_size: int):
        super().__init__()
        self.encoder = TransformerEncoder(num_blks=num_blks, eng_vocab_size=eng_vocab_size, num_hiddens=num_hiddens, num_attention_heads=num_heads, dropout_rate=dropout_rate)
        self.decoder = TransformerDecoder(num_blks=num_blks, fra_vocab_size=fra_vocab_size, num_hiddens=num_hiddens, num_attention_heads=num_heads, dropout_rate=dropout_rate)

    def forward(self, enc_input: torch.Tensor, enc_valid_lens: torch.Tensor, dec_input: torch.Tensor, dec_valid_lens: torch.Tensor):
        encoder_output = self.encoder(enc_input, enc_valid_lens)
        decoder_output = self.decoder(dec_input, encoder_output, enc_valid_lens, dec_valid_lens)

        return decoder_output
