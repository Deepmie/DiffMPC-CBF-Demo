from typing import Dict, Optional, Callable, Any
import torch
from torch import Tensor

class DerivativesFunctionWrapper:
    def __init__(self, func: Callable):
        '''
        Input:
            func: is a nx-1 map or nx-nf map;
            nx-1:  input a nx-dim row vector[nx, 1], output a scalar which shape is [1, ];
            nx-nf: input a nx-dim row vector[nx, 1], output a nf-dim row vector which shape is [nf, 1]
        '''
        self._func = func
        self._is_detect: bool = False

    def __call__(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None) -> Tensor:
        value = self._func(ipt) if params is None else self._func(ipt, params)
        if not self._is_detect: self._detect_function(ipt, value)
        return value

    def derivatives(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None, order: int=1) -> Tensor:
        '''
        Input:
            ipt[nx, 1]
        '''
        with torch.enable_grad():
            value = self.__call__(ipt, params)
            if not self._is_detect: self._detect_function(ipt, value)

            if order == 0: return value
            nx: int = ipt.shape[0]
            nf: int = value.shape[0]

            if value.ndimension() == 1: # (1, )
                grad = torch.autograd.grad(value, [ipt, ], create_graph=True)[0] # (nx, 1)
            elif value.ndimension() == 2: # (nf, 1)
                grad = torch.zeros([nf, nx]) # (nf, nx)
                for i in range(nf): grad[i] = torch.autograd.grad(value[i], [ipt, ], create_graph=True)[0].flatten()
            
            if order == 1: return grad # (nx, 1) / (nf, nx)

            if value.ndimension() == 1: # (nx, 1)
                hessian = torch.zeros([nx, nx])
                for i in range(nx): hessian[i] = torch.autograd.grad(grad[i], [ipt, ], retain_graph=True, create_graph=True)[0].flatten()
            elif value.ndimension() == 2: # (nf, nx)
                hessian = torch.zeros([nf, nx, nx])
                for i in range(nf):
                    for j in range(nx):
                        hessian[i, j] = torch.autograd.grad(grad[i, j], [ipt, ], retain_graph=True, create_graph=True)[0].flatten()
            
            if order == 2: return hessian
        
    def _detect_function(self, ipt: Tensor, value: Tensor):
        if value.ndimension() > 2: raise ValueError('Input function must be a n-1 / n-m map.')
        if value.ndimension() == 1:
            if value.shape[0] != 1: raise ValueError('Argument of function just accept a scalar or a row vector')
        self._is_detect = True