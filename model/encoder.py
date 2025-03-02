import math
import torch
import torch.nn as nn
from model.encoder_blk import TransformerEncoderBlock
from module.pos_encode import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, num_blks: int, eng_vocab_size: int, num_hiddens: int, num_attention_heads: int, dropout_rate: float):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(num_embeddings=eng_vocab_size, embedding_dim=num_hiddens)
        self.position_encode = PositionalEncoding(embedding_dim=num_hiddens, dropout_rate=dropout_rate)
        blks = [TransformerEncoderBlock(num_hiddens=num_hiddens, num_heads=num_attention_heads, dropout_rate=dropout_rate) for _ in range(num_blks)]
        self.encoder_blocks = nn.Sequential(*blks)

    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor):
        X = self.position_encode(self.embedding(X) * math.sqrt(self.num_hiddens))

        for enc_blk in self.encoder_blocks:
            X = enc_blk(X, valid_lens)

        return X