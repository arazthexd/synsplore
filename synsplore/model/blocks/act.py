from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_GENERATORS: Dict[str, Type[nn.Module]] = {
    "none": nn.Identity,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "selu": nn.SELU,
    "lrelu": nn.LeakyReLU
}