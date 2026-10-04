from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional
import torch
from torch import Tensor

class Cost(ABC):
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T; self._ntau = self._nx + self._nu

    @abstractmethod
    def get_stage_cost(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Tensor]]=None, order: int=0) -> Union[Tensor, float]:
        '''
        Input:
            ipt have two cases:
                - (tau): just tau composed by x and u
                - (x, u): a two-element tuple
        Output:
            n-order cost function value in time t
        '''
        if len(ipt) == 1: ipt = ipt[0]
        if torch.is_tensor(ipt): ipt = (ipt[:self._nx], ipt[self._nx:])
        self._ipt = ipt

    @abstractmethod
    def get_terminal_cost(self, xt: Tensor, params: Optional[Dict[str, Tensor]]=None, order: int=0) -> Union[Tensor, float]:
        '''
        Input:
            xt (nx, ): terminal state
        Output:
            n-order cost function value in time t
        '''
        pass

    def jacobian(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Tensor]]=None) -> Tensor:
        '''
        Input:
            ipt have three cases:
                - (tau[T+1, ntau])
                - (x[T+1, nx], u[T, nu])
                - tau[T+1, ntau]
        '''
        if len(ipt) == 1: ipt = ipt[0]
        if torch.is_tensor(ipt): ipt = (ipt[:, :self.nx], ipt[:, self.nx:])
        J: Tensor = torch.zeros([self._T+1, self._ntau])
        for t in range(self._T):
            J[t, :] = self.get_stage_cost((ipt[0][t], ipt[1][t]), params, order=1).flatten()
        J[self._T, :self._nx] = self.get_terminal_cost(ipt[0][self._T], params, order=1).flatten()
        return J
    
    def hessian(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Tensor]]=None) -> Tensor:
        H: Tensor = torch.zeros([self._T+1, self._ntau, self._ntau])
        for t in range(self._T):
            H[t, :, :] = self.get_stage_cost((ipt[0][t], ipt[1][t]), params, order=2)
        H[self._T, :self._nx, :self._nx] = self.get_terminal_cost(ipt[0][self._T], params, order=2)
        return H