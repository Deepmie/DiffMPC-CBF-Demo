from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor
import torch.nn as nn
from ..base.derivatives import DerivativesFunctionWrapper

class Barrier(ABC, nn.Module):
    def __init__(self, nx: int, nu: int, T: int):
        super().__init__()
        self._nx = nx; self._nu = nu; self._T = T; self._ntau = self._nx + self._nu
        self._barrier_function = DerivativesFunctionWrapper(self._define_barrier_function)

    def forward(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Any]], order: int=0) -> Tensor:
        '''
        Input:
            ipt have three cases:
            - (tau[ntau])
            - (x[nx], u[nu])
            - tau[ntau]
        Output:
            n-order cost function value in time t
            0: [nx, 1],
            1: [nx, ntau],
            2: [nx, ntau, ntau]
        '''
        if not torch.is_tensor(ipt):
            if len(ipt) == 2: ipt = torch.concat([ipt[0], ipt[1]]) # (ntau, )
            elif len(ipt) == 1: ipt = ipt[0]

        if order == 0:
            return self._barrier_function(ipt, params)
        elif order == 1 or order == 2:
            return self._barrier_function.derivatives(ipt, params, order=order)

    def get_safety_constraint(
            self,
            ipt_curr: Union[Tensor, Tuple[Tensor]], params_curr: Optional[Dict[str, Any]],
            ipt_next: Union[Tensor, Tuple[Tensor]], params_next: Optional[Dict[str, Any]],
            gamma: float,
    ) -> Tensor:
        return self.forward(ipt_next, params_next) - (1 - gamma) * self.forward(ipt_curr, params_curr) # (nx, 1)
    
    @abstractmethod
    def _define_barrier_function(self, ipt: Tensor, params: Optional[Dict[str, Any]]) ->Tensor:
        '''
        Input:
            ipt: taut[ntau, 1]
        Output:
            f(taut)[nx, 1]
        '''