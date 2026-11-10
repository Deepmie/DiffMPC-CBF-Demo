from typing import Dict, Optional, Callable, Any
import torch
from torch import Tensor
from torch.func import jacrev, vmap

class DerivativesFunctionWrapper:
    def __init__(self, func: Callable):
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
            ipt = ipt.detach().requires_grad_()
            value = self.__call__(ipt, params)
            if not self._is_detect: self._detect_function(ipt, value)

            # directly return
            if order == 0: return value

            grad = jacrev(self._func)(ipt)
            if order == 1: return grad # (N, nf, N, nx)

            hessian = jacrev(jacrev(self._func))(ipt)
            if order == 2: return hessian # (N, nf, N, nx, Nx, nx)

    def _detect_function(self, ipt: Tensor, value: Tensor):
        nx = ipt.ndimension()
        if nx != 2:
            raise ValueError(f'ipt must be two-dim value, you give a {nx}-dim, so error.')
        nf = value.ndimension()
        if nf != 2:
            raise ValueError(f'output of func must be two-dim value, you give a {nf}-dim, so error.')
        self._is_detect = True