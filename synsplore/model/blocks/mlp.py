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
                 loss: str = "ce",
                 metrics: dict = {}):
        assert n_layers > 0
        if d_hidden is None:
            d_hidden = d_in
        if d_out is None:
            d_out = d_hidden

        super().__init__()

        hidden_activation = ACTIVATION_GENERATORS[h_act]
        output_activation = ACTIVATION_GENERATORS[out_act]

        self.metrics = metrics

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
        self.y_pred = x
        return x
    
    def get_loss(self, y_true: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        if x is not None:
            self.y_pred = self.forward(x)

        return self.loss_fn(self.y_pred, y_true)        
    
    def get_metrics(self, y_true: torch.Tensor, x: torch.Tensor = None):
        if x is not None:
            self.y_pred = self.forward(x)

        metrics_outputs = {}
        for m_name, m in self.metrics.items():
            m.reset()
            # print(m)
            # print(self.y_pred)
            # print(self.y_pred.shape)
            # print(y_true)
            # print(y_true.shape)
            metrics_outputs[m_name] = m(self.y_pred, y_true)

        return metrics_outputs 
