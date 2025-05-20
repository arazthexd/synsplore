from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_GENERATORS: Dict[str, Type[nn.Module]] = {
    "ce": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss
}