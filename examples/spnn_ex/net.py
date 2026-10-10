import torch
from torch import Tensor
import torch.nn as nn

class ContextNetwork(nn.Module):
    def __init__(self, context_dim: int=3):
        '''
        this network accept three input: (pref, vref, t)
        '''
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(context_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.softplus(self._net(x)) + 1e-3