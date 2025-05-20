from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .act import ACTIVATION_GENERATORS
from .loss import LOSS_GENERATORS