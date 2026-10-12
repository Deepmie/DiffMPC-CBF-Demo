import torch
from torch import Tensor
from ..cost import Cost
from ..dynamic import Dynamic
from typing import Optional, Dict

class iLQRMetricFunction:
    def __init__(self, nx: int, nu: int, T: int, cost: Cost, dynamic: Dynamic):
        self._nx = nx; self._nu = nu; self._T = T; self._cost = cost; self._dynamic = dynamic
        self._ntau = self._nx + self._nu

    def __call__(self, x: Tensor, u: Tensor, params: Optional[Dict[str, Tensor]]=None) -> float:
        metric: float = 0.0
        # cost
        for t in range(self._T):
            metric += self._cost.get_stage_cost((x[t], u[t]), {**params.get('cost'), 't': t})
        metric += self._cost.get_terminal_cost(x[self._T], {**params.get('cost'), 't': t})
        return metric