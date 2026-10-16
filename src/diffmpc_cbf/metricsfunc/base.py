from ..base import Tau
from ..cost import Cost
from ..dynamic import Dynamic
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor

class MetricFunction:
    def __init__(self, nx: int, nu: int, nb: int, T: int, cost: Cost, dynamic: Dynamic, weight: Dict[str, float]):
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._cost = cost; self._dynamic = dynamic
        self._ntau = self._nx + self._nu
        self._weight = weight

    def __call__(
            self,
            tau: Tau,
            h: Tensor, # ((T+1)*nx, )
            g: Tensor, # (T*(nb+2*nu), )
            params: Optional[Dict[str, Any]]=None,
            dvep: Optional[Tensor]=None
        ) -> float: # (T*ntau+nx)
        metric: float = 0.0

        # cost
        for t in range(self._T):
            metric += self._cost.get_stage_cost(tau[t], {**params.get('cost'), 't': t})
        metric += self._cost.get_terminal_cost(tau.get_state(self._T), {**params.get('cost'), 't': t})

        # equation & inequation residual
        res = torch.tensor([
            torch.linalg.norm(h, ord=1),
            torch.linalg.norm(torch.max(), ord=1)
        ])
        

        _eq_weight = self.eq_weight if dvep is None else max(self.eq_weight, 1.1*torch.max(torch.abs(dvep)))
        metric += _eq_weight * torch.linalg.norm(h, ord=1)
        return metric


