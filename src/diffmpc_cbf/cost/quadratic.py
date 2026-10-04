from . import Cost
import torch
from torch import Tensor
from typing import Dict, Tuple, Union, Optional

class QuadCost(Cost):
    def __init__(self, nx: int, nu: int, T: int, C: Tensor, c: Tensor):
        '''
        Input: 
            C[T+1, ntau, ntau]
            c[T+1, ntau]
        '''
        super().__init__(nx, nu, T)
        self._C = C; self._c = c

    def get_stage_cost(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Tensor]]=None, order: int=0) -> Union[Tensor, float]:
        if len(ipt) == 1: ipt = ipt[0]
        if len(ipt) == 2: ipt = torch.concat([ipt[0], ipt[1]])
        t: int = params.get('t')
        taut = ipt.unsqueeze(-1) # (ntau, 1)
        if order == 0: # 1
            return (1/2*taut.T @ self._C[t] @ taut + self._c[t].unsqueeze(-1).T @ taut).squeeze().item()
        elif order == 1: # (ntau, 1)
            return self._C[t] @ taut + self._c[t].unsqueeze(-1)
        elif order == 2: # (ntau, ntau)
            return self._C[t]
    
    def get_terminal_cost(self, xt: Tensor, params: Optional[Dict[str, Tensor]]=None, order: int=0) -> Union[Tensor, float]:
        xt = xt.unsqueeze(-1) # (nx, 1)
        CT = self._C[self._T, :self._nx, :self._nx] # (nx, nx)
        cT = self._c[self._T, :self._nx].unsqueeze(-1) # (nx, 1)
        if order == 0: # 1
            return (1/2*xt.T @ CT @ xt + cT.T @ xt).squeeze().item()
        elif order == 1: # (nx, 1)
            return CT @ xt + cT
        elif order == 2: # (nx, nx)
            return CT

    def jacobian(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Tensor]]=None) -> Tensor:
        '''
        Input:
            ipt have three cases:
                - (tau[T+1, ntau])
                - (x[T+1, nx], u[T, nu])
                - tau[T+1, ntau]
        '''
        if len(ipt) == 1: ipt = ipt[0]
        if len(ipt) == 2:
            _tau = torch.zeros([self._T+1, self._ntau])
            _tau[:, :self._nx] = ipt[0]; _tau[:self._T, self._nx:] = ipt[1]
            ipt = _tau
        return (self._C @ ipt.unsqueeze(-1) + self._c.unsqueeze(-1)).squeeze() # (T+1, ntau)

    def hessian(self) -> Tensor:
        return self._C # (T+1, ntau, ntau)