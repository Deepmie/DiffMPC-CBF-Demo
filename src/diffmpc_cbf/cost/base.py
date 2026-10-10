from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Callable, Any
import torch
from torch import Tensor
from ..base import DerivativesFunctionWrapper

class Cost(ABC):
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T; self._ntau = self._nx + self._nu
        self._stage_cost_function = DerivativesFunctionWrapper(self._define_stage_cost_function)
        self._terminal_cost_function = DerivativesFunctionWrapper(self._define_terminal_cost_function)

    def get_stage_cost(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Any]]=None, order: int=0) -> Union[Tensor, float]:
        '''
        Input:
            ipt have three cases:
            - (tau[T+1, ntau])
            - (x[T+1, nx], u[T, nu])
            - tau[T+1, ntau]
        Output:
            n-order stage cost function value in time t
            0: [1]
            1: [ntau, 1]
            2: [ntau, ntau]
        '''
        if not torch.is_tensor(ipt):
            if len(ipt) == 2: ipt = torch.concat([ipt[0], ipt[1]]) # (ntau, )
            elif len(ipt) == 1: ipt = ipt[0]

        if order == 0:
            return self._stage_cost_function(ipt, params)
        elif order == 1 or order == 2:
            return self._stage_cost_function.derivatives(ipt, params, order=order)

    def get_terminal_cost(self, xt: Tensor, params: Optional[Dict[str, Any]]=None, order: int=0) -> Union[Tensor, float]:
        '''
        Input:
            xt (nx, ): terminal state
        Output:
            n-order terminal cost function value in time t
            0: [1]
            1: [nx, 1]
            2: [nx, nx]
        '''
        if order == 0:
            return self._terminal_cost_function(xt, params)
        elif order == 1 or order == 2:
            return self._terminal_cost_function.derivatives(xt, params, order=order)

    def jacobian(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Any]]=None) -> Tensor:
        '''
        Input:
            ipt have three cases:
                - (tau[T+1, ntau])
                - (x[T+1, nx], u[T, nu])
                - tau[T+1, ntau]
        '''
        if not torch.is_tensor(ipt):
            if len(ipt) == 2:
                ipt = torch.concat([
                    ipt[0],
                    torch.concat([ipt[1], torch.zeros(1, self._nu)], dim=0) # (T+1, nu)
                ])
            elif len(ipt) == 1:
                ipt = ipt[0]
        
        J: Tensor = torch.zeros([self._T+1, self._ntau])
        for t in range(self._T): J[t] = self.get_stage_cost(ipt[t], params, order=1).flatten()
        J[self._T, :self._nx] = self.get_terminal_cost(ipt[self._T, :self._nx], params, order=1).flatten()
        return J
    
    def hessian(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Any]]=None) -> Tensor:
        H: Tensor = torch.zeros([self._T+1, self._ntau, self._ntau])
        for t in range(self._T):
            H[t, :, :] = self.get_stage_cost((ipt[0][t], ipt[1][t]), params, order=2)
        H[self._T, :self._nx, :self._nx] = self.get_terminal_cost(ipt[0][self._T], params, order=2)
        return H

    @abstractmethod
    def _define_stage_cost_function(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None) -> Tensor:
        '''
        Input:
            ipt: taut
        Output:
            l(taut), 1-dimension
        '''

    def _define_terminal_cost_function(self, xt: Tensor, params: Optional[Dict[str, Any]]=None) -> Tensor:
        '''
        Input:
            ipt: xT
        Output:
            lT(xT), 1-dimension
        '''
        return torch.tensor([0]) # (1, )