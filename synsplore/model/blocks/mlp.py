from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .act import ACTIVATION_GENERATORS
from .loss import LOSS_GENERATORS

class _SimpleMLP(nn.Module):
    def __init__(self, 
                 d_in: int,
                 d_hidden: int = None, 
                 d_out: int = None,
                 n_layers: int = 3,
                 h_act: str = "relu",
                 out_act: str = "none",
                 loss: str = "ce"):
        assert n_layers > 0
        if d_hidden is None:
            d_hidden = d_in
        if d_out is None:
            d_out = d_hidden

        super().__init__()

        hidden_activation = ACTIVATION_GENERATORS[h_act]
        output_activation = ACTIVATION_GENERATORS[out_act]

        self.layers = nn.ModuleDict()
        if n_layers == 1:
            self.layers["lin0"] = nn.Linear(d_in, d_out)
            self.layers["act0"] = output_activation()
        else:
            self.layers["lin0"] = nn.Linear(d_in, d_hidden)
            self.layers["act0"] = hidden_activation()
            for i in range(1, n_layers-1):
                self.layers[f"lin{i}"] = nn.Linear(d_hidden, d_hidden)
                self.layers[f"act{i}"] = hidden_activation()
            self.layers[f"lin{n_layers-1}"] = nn.Linear(d_hidden, d_out)
            self.layers[f"act{n_layers-1}"] = output_activation()
        
        self.loss_fn = LOSS_GENERATORS[loss]()

    def forward(self, x: torch.Tensor):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def get_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred = self.forward(x)
        return self.loss_fn(y_pred, y_true)