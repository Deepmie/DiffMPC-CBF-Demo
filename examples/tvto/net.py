import torch
from torch import Tensor
import torch.nn as nn

class ContextNetwork(nn.Module):
    def __init__(self, context_dim: int=3):
        '''
        this network accept three input: (pref, vref, t)
        Pipline: (b, 3) --> (b, 3) 
        '''
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.softplus(self._net(x)) + 1e-3


class ResidualNetwork(nn.Module):
    def __init__(self, nx: int, nu: int):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(nx+nu, 8),
            nn.Tanh(),
            nn.Linear(8, nu),
        )

    def forward(self, x: Tensor) -> Tensor: # (ntau, )
        return self._net(x) # (nu, )


class DeltaNetwork(nn.Module):
    def __init__(self, ipt_dim: int):
        super().__init__()
        self._net = nn.Sequential(
            nn.Linear(ipt_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x: Tensor) -> Tensor: # (3, )
        return 0.02*torch.sigmoid(self._net(x))