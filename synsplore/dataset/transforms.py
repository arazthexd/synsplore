from typing import Callable

import torch

from .data import SynsploreData, RouteData, PharmData
from .dataset import SynsploreDataset

class BaseTransform:
    pass

class PharmNoiseTransform(BaseTransform):
    def __init__(self, std: float = 1):
        self.std = std

    def __call__(self, data: SynsploreData) -> SynsploreData:
        data = data.clone()
        data.pharm_data.values += \
            torch.randn_like(data.pharm_data.values) * self.std
        return data

class PharmMaskTransform(BaseTransform):
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, data: SynsploreData) -> SynsploreData:
        data = data.clone()
        mask = torch.rand(data.pharm_data.values.shape[0]) < self.p
        data.pharm_data.values = data.pharm_data.values[mask]
        data.pharm_data.typeids = data.pharm_data.typeids[mask]
        return data
        