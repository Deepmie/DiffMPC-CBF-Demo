from typing import Dict, Optional, Callable, Any
import torch
from torch import Tensor
from torch.func import jacrev, hessian

class DerivativesFunctionWrapper:
    def __init__(self, func: Callable[[Tensor], Tensor]):
        '''
        Arguments:
            - func: must be a function satisfy:
              Input of this func: [N, nx]
              Output of this func: [N, nf] (nf maybe is 1)
        '''
        self._func = func
        self._is_detect: bool = False

    def __call__(self, ipt: Tensor, params: Dict[str, Any]) -> Tensor:
        value = self._func(ipt, params)
        if not self._is_detect: self._detect_function(ipt, value)
        return value

    def derivatives(self, ipt: Tensor, params: Dict[str, Any], order: int=1) -> Tensor:
        '''
        Arguments:
            - ipt[N, nx]
        Returns:
            - Value
            zero: [N, nf]
            one:  [N, nf, nx]
            two:  [N, nf, nx, nx]
        '''
        with torch.enable_grad():
            # ipt = ipt.detach().requires_grad_()
            # if not self._is_detect: self._detect_function(ipt, value)

            # directly return
            if order == 0:
                value = self.__call__(ipt, params)
                return value

            elif order == 1:
                grad: Tensor = jacrev(self._func)(ipt, params)
                return grad.diagonal(dim1=0, dim2=2).permute(2, 0, 1)

            elif order == 2:
                hess: Tensor = hessian(self._func)(ipt, params)
                return hess.diagonal(dim1=0, dim2=2).diagonal(dim1=2, dim2=4).permute(3, 0, 1, 2)
    

    def _detect_function(self, ipt: Tensor, value: Tensor):
        nx = ipt.ndimension()
        if nx != 2:
            raise ValueError(f'ipt must be two-dim value, you give a {nx}-dim, so error.')
        nf = value.ndimension()
        if nf != 2:
            raise ValueError(f'output of func must be two-dim value, you give a {nf}-dim, so error.')
        self._is_detect = True