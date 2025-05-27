import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class _SimpleEmbedding(nn.Module):
    """Simple embedding layer for 1D data."""

    def __init__(self, n_emb: int, d_emb: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = n_emb,
                                      embedding_dim = d_emb)

    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            return self.embedding(x)
        else:
            idx = x.argmax(dim=1)
            return self.embedding(idx)
    
class _PositionalEncoding(nn.Module):
    """Taken from https://github.com/wenhao-gao/synformer/ and modified."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, stidx: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        nbatch, seqlen, _ = x.size()

        idx = torch.arange(seqlen).unsqueeze(1).T.repeat([nbatch, 1])
        idx = idx - stidx.unsqueeze(1)
        mask = idx >= 0
        idx[~mask] = 0
        x = x + self.pe[idx]
        return self.dropout(x)