from ..base import Tau
from ..cost import Cost
from ..dynamic import Dynamic
from ..barrier import Barrier
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor

class MetricFunction:
    def __init__(self, nx: int, nu: int, nb: int, T: int, cost: Cost, dynamic: Dynamic, barrier: Barrier, umin: Tensor, umax: Tensor, weight: Dict[str, float]):
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._cost = cost; self._dynamic = dynamic; self._barrier = barrier; self._umin = umin; self._umax = umax
        self._ntau = self._nx + self._nu
        self._weight = weight
        if not (('equation' in self._weight) and ('inequation_cbf' in self._weight) and ('inequation_boundary' in self._weight)): raise ValueError('Missing Keywords')

    def __call__(
            self,
            tau: Tau,
            x0: Tensor,
            params: Optional[Dict[str, Any]]=None,
            dveq: Optional[Tensor]=None,
            dvneq: Optional[Tensor]=None,
        ) -> float: # (T*ntau+nx)
        metric: float        = 0.0
        h: Tensor            = torch.zeros([(self._T+1)*self._nx])
        g_cbf: Tensor        = torch.zeros([self._T*self._nb])
        g_boundary: Tensor   = torch.zeros([self._T*2*self._nu])
        barrier_params: Dict = params.get('barrier')
        gamma: float         = barrier_params.get('gamma')
        for t in range(self._T+1):
            # cost
            if t <  self._T: metric += self._cost.get_stage_cost(tau[t], {**params.get('cost'), 't': t})
            if t == self._T: metric += self._cost.get_terminal_cost(tau.get_state(self._T), {**params.get('cost'), 't': self._T})

            # equation
            _state_prev = x0 if t == 0 else self._dynamic.forward(tau[t-1], {**params.get('dynamic'), 't': t-1})
            h[t*self._nx: (t+1)*self._nx] = (_state_prev - tau.get_state(t)).flatten()

            if t < self._T:
                # inequation
                g_cbf[t*self._nb: (t+1)*self._nb] = -self._barrier.forward(tau.get_state(t+1), {**barrier_params, 't': t+1}).flatten() + (1-gamma)*self._barrier.forward(tau.get_state(t), {**barrier_params, 't': t}).flatten()
                g_boundary[t*2*self._nu: (t+1)*2*self._nu] = torch.concat([self._umin-tau.get_control(t).flatten(), tau.get_control(t).flatten()-self._umax]) # (2*nu, )

        # equation & inequation residual
        res = torch.tensor([
            torch.linalg.norm(h, ord=1),
            torch.linalg.norm(torch.clamp(g_cbf, min=0.), ord=1),
            torch.linalg.norm(torch.clamp(g_boundary, min=0.), ord=1),
        ]).unsqueeze(-1) # (3, 1)
        weight = torch.tensor([
            self._get_weight(self._weight.get('equation'), dveq),
            self._get_weight(self._weight.get('inequation_cbf'), dvneq[:self._T*self._nb]),
            self._get_weight(self._weight.get('inequation_boundary'), dvneq[self._T*self._nb:]),
        ]).unsqueeze(0) # (1, 3)
        metric += (weight @ res).item()
        return metric

    def _get_weight(self, w: float, dv: Optional[Tensor]=None) -> float:
        if dv is None:
            return w
        return max(w, 1.1*dv.abs().max().item())


