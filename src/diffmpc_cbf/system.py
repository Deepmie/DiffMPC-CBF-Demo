from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Callable, Any
import torch
from torch import Tensor
import torch.nn as nn
from .base import DerivativesFunctionWrapper

class Cost(ABC, nn.Module):
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
        ):
        super().__init__()
        self._nx = nx
        self._nu = nu
        self._nb = nb
        self._T = T
        self._ntau = self._nx + self._nu
        self._device = device
        self._dtype = dtype
        self._cost_func = DerivativesFunctionWrapper(self._define_cost_function)

    def get_cost(self, tau: Tensor, params: Dict[str, Any], order: int=0):
        if order == 1:
            return self._cost_func(tau, params)
        elif order > 1:
            return self._cost_func.derivatives(tau, params, order=order)

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]):
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        return super().to(device, dtype)

    @abstractmethod
    def _define_cost_function(self, tau: Tensor, params: Dict[str, Any]):
        '''
        In here, define the cost function:
            Input: [T+1, ntau]
            Output: [T+1, ]
        '''
    

class Dynamic(ABC, nn.Module):
    def __init__(
            self,
            nx: int,
            nu: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
        ):
        super().__init__()
        self._nx = nx
        self._nu = nu
        self._T = T
        self._ntau = self._nx + self._nu
        self._device = device
        self._dtype = dtype
        self._dym_equ = DerivativesFunctionWrapper(self._define_dynamic_equation)
    
    def forward(self, tau: Union[Tensor, Tuple[Tensor]], params: Dict[str, Any], order: int=0) -> Tensor:
        if order == 0:
            return self._dym_equ(tau)
        elif order == 1 or order == 2:
            return self._dym_equ.derivatives(tau, params, order=order)
    
    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]):
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        return super().to(device, dtype)
    
    @abstractmethod
    def _define_dynamic_equation(self, tau: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        In here, define the dynamic equation:
            Input: [T+1, ntau]
            Output: [T+1, nx]
        '''


class Barrier(ABC, nn.Module):
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
        ):
        super().__init__()
        self._nx = nx
        self._nu = nu
        self._nb = nb
        self._T = T
        self._ntau = self._nx + self._nu
        self._device = device
        self._dtype = dtype
        self._bar_func = DerivativesFunctionWrapper(self._define_barrier_function)

    def forward(self, x: Tensor, params: Dict[str, Any], order: int=0) -> Tensor:
        if order == 0:
            return self._bar_func(x, params)
        elif order == 1 or order == 2:
            return self._bar_func.derivatives(x, params, order=order)

    def to(self, device: Optional[torch.device], dtype: Optional[torch.dtype]):
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        return super().to(device, dtype)
    
    def get_safety_constraint(
            self,
            ipt_curr: Union[Tensor, Tuple[Tensor]], params_curr: Optional[Dict[str, Any]],
            ipt_next: Union[Tensor, Tuple[Tensor]], params_next: Optional[Dict[str, Any]],
            gamma: float,
    ) -> Tensor:
        return self.forward(ipt_next, params_next) - (1 - gamma) * self.forward(ipt_curr, params_curr) # (nx, 1)
    
    @abstractmethod
    def _define_barrier_function(self, ipt: Tensor, params: Optional[Dict[str, Any]]) ->Tensor:
        '''
        In here, define the barrier equation:
            Input: [T+1, ntau]
            Output: [T+1, nx]
        '''