from . import Dynamic
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor

class LinDynamic(Dynamic):
    def __init__(self, nx: int, nu: int, T: int, F: Tensor):
        '''
        Input:
            F[T, nx, ntau]
        '''
        super().__init__(nx, nu, T)
        self._F = F

    def forward(self, ipt: Union[Tensor, Tuple[Tensor]], params: Optional[Dict[str, Any]]=None, order: int=0) -> Tensor:
        if len(ipt) == 1: ipt = ipt[0]
        if len(ipt) == 2: ipt = torch.concat([ipt[0], ipt[1]])
        t: int = params.get('t')
        if order == 0:
            return (self._F[t] @ ipt.unsqueeze(-1)).squeeze() # (nx, )
        elif order == 1:
            return self._F[t] # (nx, ntau)

    def jacobian(self, ipt: Union[Tensor, Tuple[Tensor]]):
        '''
        Input:
            ipt have three cases:
            - (tau[T+1, ntau])
            - (x[T+1, nx], u[T, nu])
            - tau[T+1, ntau]
        '''
        return self._F # (T, nx, ntau)