from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn

class Dynamic(ABC, nn.Module):
    def __init__(self, nx: int, nu: int, T: int):
        super().__init__()
        self._nx = nx; self._nu = nu; self._T = T; self._ntau = self._nx + self._nu

    @abstractmethod
    def forward(self, ipt: Union[Tensor, Tuple[Tensor]], order: int=0) -> Tensor:
        if len(ipt) == 1: ipt = ipt[0]
        if torch.is_tensor(ipt): ipt = (ipt[:self._nx], ipt[self._nx:])
        self._ipt = ipt

    def jacobian(self, ipt: Union[Tensor, Tuple[Tensor]]):
        '''
        Input:
            ipt have three cases:
            - (tau[T+1, ntau])
            - (x[T+1, nx], u[T, nu])
            - tau[T+1, ntau]
        '''
        if len(ipt) == 1: ipt = ipt[0]
        if torch.is_tensor(ipt): ipt = (ipt[:, :self.nx], ipt[:, self.nx:])
        J = torch.zeros([self._T, self._nx, self._ntau])
        for t in range(self._T):
            J[t, :, :] = self.forward((ipt[0][t], ipt[1][t]), order=1)
        return J
        