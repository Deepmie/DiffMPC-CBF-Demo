from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from numpy import ndarray

class Dynamic(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, ipt: Tuple[ndarray], order: int=0) -> ndarray:
        if len(ipt) == 1: ipt = (ipt[:self._nx], ipt[self._nx:])
    