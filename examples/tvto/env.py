import numpy as np
from numpy import ndarray

class Env:
    def __init__(self, x0: ndarray, xref: ndarray):
        self._x0 = x0; self._xref = xref

    