import math
import torch
import torch.nn as nn
from model.decoder_blk import TransformerDecoderBlock
from module.pos_encode import PositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self, num_blks: int, fra_vocab_size: int, num_hiddens: int, num_attention_heads: int, dropout_rate: float):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(num_embeddings=fra_vocab_size, embedding_dim=num_hiddens)
        self.position_encode = PositionalEncoding(embedding_dim=num_hiddens, dropout_rate=dropout_rate)
        blks = [TransformerDecoderBlock(num_hiddens=num_hiddens, num_heads=num_attention_heads, dropout_rate=dropout_rate) for _ in range(num_blks)]
        self.encoder_blocks = nn.Sequential(*blks)
        self.dense = nn.LazyLinear(out_features=fra_vocab_size)

    def forward(self, X: torch.Tensor, enc_valid_lens: torch.Tensor):
        X = self.position_encode(self.embedding(X) * math.sqrt(self.num_hiddens))

        for enc_blk in self.encoder_blocks:
            X = enc_blk(X, enc_valid_lens)

        return self.dense(X)