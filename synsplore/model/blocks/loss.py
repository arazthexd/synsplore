from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss:
    def __init__(self, smooth=1):
        self.smooth = smooth
    
    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
LOSS_GENERATORS: Dict[str, Type[nn.Module]] = {
    "none": lambda: None, 
    "ce": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "dice": DiceLoss,
}

