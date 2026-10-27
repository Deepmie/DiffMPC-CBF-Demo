from ..base import Tau
from ..cost import Cost
from ..dynamic import Dynamic
from ..barrier import Barrier
from ..utils import *
from typing import Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor

class MetricFunction:
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            cost: Cost,
            dynamic: Dynamic,
            barrier: Barrier,
            umin: Tensor,
            umax: Tensor,
            weight: Dict[str, float],
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32,
        ):
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._cost = cost; self._dynamic = dynamic; self._barrier = barrier; self._umin = umin; self._umax = umax
        self._ntau = self._nx + self._nu; self._ns = self._T*self._nb
        self._t_ts: Tensor = torch.arange(0, self._T+1, step=1, device=device, dtype=dtype)
        self._weight = weight; self._device = device; self._dtype = dtype
        if not (('equation' in self._weight) and ('inequation_cbf' in self._weight) and ('inequation_boundary' in self._weight) and ('inequation_slack' in self._weight)): raise ValueError('Missing Keywords')

    def __call__(
            self,
            tau: Tau,
            slack: Tensor, # (ns, )
            dveq: Tensor,
            dvneq: Tensor,
            h: Tensor, # (T+1*nx, )
            g_cbf: Tensor, # (T*nb, )
            g_boundary: Tensor, # (T*2*nu, )
            x0: Tensor,
            params: Optional[Dict[str, Any]],
            slack_weight: List[float],
            include_res: bool=False,
        ) -> Union[Tuple[Tensor], Tensor]: # (T*ntau+nx)
        if not (tau.device == self._device and tau.dtype == self._dtype): raise ValueError(f'device and dtype of tau is {x0.device} and {x0.dtype}, respectively, not match DiffMPC-CBFs device {self._dtype} and dtype {self._dtype}.')
        metric: float        = 0.0
        cost_params          = params.get('cost')

        # cost
        for t in range(self._T+1):
            if t <  self._T: metric += self._cost.get_stage_cost(tau[t], self._tcon(cost_params, t))
            if t == self._T: metric += self._cost.get_terminal_cost(tau.get_state(self._T), self._tcon(cost_params, self._T))
        
        # add slack penalty
        rho1, rho2 = slack_weight
        metric += rho1*slack.sum(dim=0) + rho2/2*slack.square().sum(dim=0)
        
        # equation & inequation residual
        res = torch.stack([
            torch.norm(h, p=float('inf')),
            torch.norm(torch.clamp(g_cbf-slack, min=0.), p=float('inf')),
            torch.norm(torch.clamp(g_boundary, min=0.), p=float('inf')),
            torch.norm(torch.clamp(-slack, min=0.), p=float('inf')),
        ]).unsqueeze(-1) # (4, 1)

        _dvneq_cbf      = dvneq[:self._T*self._nb]; _bias = self._T*self._nb
        _dvneq_boundary = dvneq[_bias: _bias+self._T*2*self._nu]; _bias += self._T*2*self._nu
        _dvneq_slack    = dvneq[_bias: _bias+self._T*self._nb]
        weight = torch.stack([
            self._get_weight(self._weight.get('equation'), dveq),
            self._get_weight(self._weight.get('inequation_cbf'), _dvneq_cbf),
            self._get_weight(self._weight.get('inequation_boundary'), _dvneq_boundary),
            self._get_weight(self._weight.get('inequation_slack'), _dvneq_slack),
        ]).reshape(1, -1) # (1, 4)
        metric += (weight @ res).flatten()
        return (metric, res) if include_res else metric

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Tensor:
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        return self

    def _pcon(self, params: Dict[str, Any], addition_params: Dict[str, Any], device: torch.device):
        return {**params, **convert_tensor_iteration(addition_params, device)}

    def _tcon(self, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        return {**params, **{'t': t, 't_ts': self._t_ts[t]}}

    def _get_weight(self, w: float, dv: Optional[Tensor]) -> Tensor:
        w_ts = convert_tensor_iteration(w, self._device, self._dtype)
        if dv is None:
            return w_ts
        return torch.max(w_ts, 1.1*dv.abs().max())


