from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor
import torch.nn as nn
from ..base import DerivativesFunctionWrapper

class Dynamic(ABC, nn.Module):
    def __init__(self, nx: int, nu: int, T: int):
        super().__init__()
        self._nx = nx; self._nu = nu; self._T = T; self._ntau = self._nx + self._nu
        self._state_transition_equation = DerivativesFunctionWrapper(self._define_state_transition_equation)

    @abstractmethod
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
            return self._state_transition_equation(ipt, params)
        elif order == 1 or order == 2:
            return self._state_transition_equation.derivatives(ipt, params, ord=ord)
        

    def jacobian(self, ipt: Union[Tensor, Tuple[Tensor]]) -> Tensor:
        '''
        Input:
            ipt have three cases:
            - (tau[T+1, ntau])
            - (x[T+1, nx], u[T, nu])
            - tau[T+1, ntau]
        Output:
            Jacobian Matrix [T, nx, ntau]
        '''
        if not torch.is_tensor(ipt):
            if len(ipt) == 2:
                ipt = torch.concat([
                    ipt[0],
                    torch.concat([ipt[1], torch.zeros(1, self._nu)], dim=0) # (T+1, nu)
                ])
            elif len(ipt) == 1:
                ipt = ipt[0]

        J: Tensor = torch.zeros([self._T, self._nx, self._ntau])
        for t in range(self._T): J[t] = self.forward(ipt[t], order=1)
        return J

    @abstractmethod
    def _define_state_transition_equation(self, ipt: Tensor, params: Optional[Dict[str, Any]]) -> Tensor:
        '''
        Input:
            ipt: taut
        Output:
            f(taut)[nx, 1]
        '''