import torch
import torch.nn as nn
from tensordict import TensorDict

class PositionalEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        return x
    
class SimpleMLP(nn.Module): 
    def __init__(self, d_in, d_out, n_layers=1): # TODO
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_layers = n_layers

    def forward(self, x): # TODO
        return x @ torch.randn((self.d_in, self.d_out))

class SimpleEmbedding(nn.Module):
    def __init__(self, n_embedding, d_embedding):
        super().__init__()
        self.n_embedding = n_embedding
        self.d_embedding = d_embedding
        self.embeddings = nn.Embedding(n_embedding, d_embedding)
    
    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            return self.embeddings.forward(x)
        else:
            idx = x.argmax(dim=1)
            return self.embeddings.forward(idx)

class SequenceEncoder(nn.Module):
    def __init__(self, d_model: int):
        self.d_model = d_model
    
    def forward(self, seq: torch.Tensor, pad_mask: torch.Tensor): # TODO
        return seq