from typing import Dict, Optional, Callable, Any
import torch
from torch import Tensor

class DerivativesFunctionWrapper:
    def __init__(self, func: Callable):
        self._func = func

    def __call__(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None) -> Tensor:
        return self._func(ipt, params)

    def derivatives(self, ipt: Tensor, params: Optional[Dict[str, Any]]=None, ord: int=1) -> Tensor:
        with torch.enable_grad():
            value = self.__call__(ipt, params)
            if ord == 0:
                return value # (nf, )
            grad, = torch.autograd.grad(value, [ipt, ], create_graph=True)
            if ord == 1:
                return grad # (nx, nf)

            hessian = torch.stack([
                torch.autograd.grad(grad[i], [ipt, ], retain_graph=True, create_graph=True)[0] 
                for i in range(ipt.numel())])
            if ord == 2:
                return hessian