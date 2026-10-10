from typing import Dict, Optional, Callable, Any
import torch
from torch import Tensor

class DerivativesFunctionWrapper:
    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None) -> Tensor:
        return self._func(ipt, params)

    def derivatives(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None, ord: int=1) -> Tensor:
        '''
        Input:
            ipt: (nx, )
        '''
        with torch.enable_grad():
            value = self.__call__(ipt, params)
            if ord == 0: return value # (nf, 1)
            
            grad, = torch.autograd.grad(value, [ipt, ], create_graph=True)
            if ord == 1: return grad # (nx, nf)

            nx: int = ipt.shape[0]; nf: int = value.shape[0]
            hessian = torch.zeros([nx, nx, nf])
            for i in range(nx):
                hessian[i] = torch.autograd.grad(grad[i], [ipt, ], retain_graph=True, create_graph=True)[0]
            if ord == 2: return hessian