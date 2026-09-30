from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional
import numpy as np
from numpy import ndarray

class Cost(ABC):
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T

    @abstractmethod
    def get_stage_cost(self, ipt: Tuple[ndarray], params: Optional[Tuple[ndarray]]=None, order: int=0) -> Union[ndarray, float]:
        '''
        Input:
            ipt have two cases:
                - (tau): just tau composed by x and u
                - (x, u): a two-element tuple
        Output:
            n-order cost function value in time t
        '''
        if len(ipt) == 1: ipt = (ipt[:self._nx], ipt[self._nx:])

    @abstractmethod
    def get_terminal_cost(self, xt: ndarray, params: Optional[Tuple[ndarray]]=None, order: int=0) -> Union[ndarray, float]:
        '''
        Input:
            xt (nx, ): terminal state
        Output:
            n-order cost function value in time t
        '''