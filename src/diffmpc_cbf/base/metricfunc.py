from .tau import Tau
from .. import Cost, Dynamic
from typing import Dict, Optional
import numpy as np
from numpy import ndarray

class MetricFunction:
    def __init__(self, nx: int, nu: int, T: int, cost: Cost, dynamic: Dynamic):
        self._nx = nx; self._nu = nu; self._T = T; self._cost = cost; self._dynamic = dynamic
        self._ntau = self._nx + self._nu
        self.eq_weight: float = 10.0

    def __call__(self, tau: Tau, x0: ndarray, cost_param: Optional[Dict[ndarray]]=None, dvep: Optional[ndarray]=None) -> float: # (T*ntau+nx)
        metric: float = 0.0

        # cost
        for t in range(self._T):
            metric += self._cost.get_stage_cost(tau[t], cost_param)
        metric += self._cost.get_terminal_cost(tau.get_state(self._T), cost_param)

        # equality residual
        h = np.zeros([(self._T+1)*self._nx])
        for t in range(self._T+1):
            if t == 0:
                _dym_term = x0
            else:
                _dym_term = self._dynamic.step(tau[t-1])
            h[t*self._nx: (t+1)*self._nx] = _dym_term.flatten() - tau.get_state(t)

        _eq_weight = self.eq_weight if dvep is None else max(self.eq_weight, 1.1*np.max(np.abs(dvep)))
        metric += _eq_weight * np.linalg.norm(h, ord=1)
        return metric
