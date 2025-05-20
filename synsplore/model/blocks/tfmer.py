from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .act import ACTIVATION_GENERATORS
from .loss import LOSS_GENERATORS

class _TransformerDecoder(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_feedforward: int, 
                 n_layers: int, 
                 output_norm: bool,
                 has_encoder: bool):
        super().__init__()
        
        if not has_encoder:
            self.dec = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_feedforward,
                    batch_first=True,
                    norm_first=True),
                num_layers=n_layers,
                norm=nn.LayerNorm(d_model) if output_norm else None
            )
        else:
            self.dec = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_feedforward,
                    batch_first=True,
                    norm_first=True
                ),
                num_layers=n_layers,
                norm=nn.LayerNorm(d_model) if output_norm else None
            )
        
        self.has_encoder = has_encoder
    
    def forward(self,
                seqenc: torch.Tensor,
                pad_mask: torch.Tensor,
                cond: torch.Tensor = None):
        
        autoreg_mask = nn.Transformer.generate_square_subsequent_mask(
            sz=seqenc.shape[1],
            dtype=seqenc.dtype,
            device=seqenc.device,
        )
        
        if not self.has_encoder:
            seqenc = self.dec.forward(
                src=seqenc,
                mask=autoreg_mask,
                src_key_padding_mask=pad_mask
            )
        else:
            seqenc = self.dec.forward(
                tgt=seqenc,
                memory=cond,
                tgt_mask=autoreg_mask,
                tgt_key_padding_mask=pad_mask
            )
        
        return seqenc


