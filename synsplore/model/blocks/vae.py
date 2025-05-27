from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .act import ACTIVATION_GENERATORS
from .loss import LOSS_GENERATORS
from .mlp import _SimpleMLP

class _SimpleVAEDecoder(nn.Module):
    def __init__(self, 
                 d_in: int,
                 d_latent: int = None,
                 d_hidden: int = None, 
                 d_out: int = None,
                 n_layers: int = 3,
                 h_act: str = "relu",
                 out_act: str = "none",
                 loss: str = "ce",
                 loss_beta: float = 1.0):
        assert n_layers > 0
        if d_latent is None:
            d_latent = d_in
        if d_hidden is None:
            d_hidden = d_latent
        if d_out is None:
            d_out = d_hidden

        super().__init__()
        
        self.mu = nn.Linear(d_in, d_latent)
        self.logvar = nn.Linear(d_in, d_latent)
        
        self.decoder = _SimpleMLP(
            d_in=d_latent,
            d_hidden=d_hidden,
            d_out=d_out,
            n_layers=n_layers,
            h_act=h_act,
            out_act=out_act,
            loss=loss
        )

        self.loss_beta = loss_beta

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        mu = self.mu(z)
        logvar = self.logvar(z)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(mu)
        return self.decoder(z), mu, logvar
    
    def get_loss(self, z: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred, mu, logvar = self.forward(z)  
        recon_loss = self.decoder.loss_fn(y_pred, y_true)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.loss_beta * kl_loss