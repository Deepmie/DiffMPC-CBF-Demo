from .tau import Tau
from .. import Cost, Dynamic
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor

class MetricFunction:
    def __init__(self, nx: int, nu: int, T: int, cost: Cost, dynamic: Dynamic):
        self._nx = nx; self._nu = nu; self._T = T; self._cost = cost; self._dynamic = dynamic
        self._ntau = self._nx + self._nu
        self.eq_weight: float = 10.0

    def __call__(
            self,
            tau,
            x0: Tensor,
            params: Optional[Dict[str, Any]]=None,
            dvep: Optional[Tensor]=None
        ) -> float: # (T*ntau+nx)
        metric: float = 0.0

        # cost
        for t in range(self._T):
            metric += self._cost.get_stage_cost(tau[t], {**params.get('cost'), 't': t})
        metric += self._cost.get_terminal_cost(tau.get_state(self._T), {**params.get('cost'), 't': t})

        # equality residual
        h = torch.zeros([(self._T+1)*self._nx])
        for t in range(self._T+1):
            if t == 0:
                _dym_term = x0
            else:
                _dym_term = self._dynamic.forward(tau[t-1], {**params.get('dynamic'), 't': t})
            h[t*self._nx: (t+1)*self._nx] = _dym_term.flatten() - tau.get_state(t)

        _eq_weight = self.eq_weight if dvep is None else max(self.eq_weight, 1.1*torch.max(torch.abs(dvep)))
        metric += _eq_weight * torch.linalg.norm(h, ord=1)
        return metric


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