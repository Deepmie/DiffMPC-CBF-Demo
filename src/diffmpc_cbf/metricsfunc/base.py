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
        self._ntau = self._nx + self._nu
        self._weight = weight; self._device = device; self._dtype = dtype
        if not (('equation' in self._weight) and ('inequation_cbf' in self._weight) and ('inequation_boundary' in self._weight)): raise ValueError('Missing Keywords')

    def __call__(
            self,
            tau: Tau,
            x0: Tensor,
            params: Optional[Dict[str, Any]]=None,
            dveq: Optional[Tensor]=None,
            dvneq: Optional[Tensor]=None,
            only_res: bool=False,
        ) -> Union[float, Tensor]: # (T*ntau+nx)
        if not (tau.device.type == self._device.type and tau.dtype == self._dtype): raise ValueError(f'device and dtype of tau is {x0.device} and {x0.dtype}, respectively, not match DiffMPC-CBFs device {self._dtype} and dtype {self._dtype}.')
        def _torch_zeros(size): return torch.zeros(size, device=self._device, dtype=self._dtype)
        metric: float        = 0.0
        h: Tensor            = _torch_zeros([(self._T+1)*self._nx])
        g_cbf: Tensor        = _torch_zeros([self._T*self._nb])
        g_boundary: Tensor   = _torch_zeros([self._T*2*self._nu])
        barrier_params: Dict = params.get('barrier')
        gamma: float         = barrier_params.get('gamma')
        cost_params          = params.get('cost')
        dynamic_params       = params.get('dynamic')
        barrier_params       = params.get('barrier')
        
        for t in range(self._T+1):
            # cost
            if t <  self._T: metric += self._cost.get_stage_cost(tau[t], self._tcon(cost_params, t))
            if t == self._T: metric += self._cost.get_terminal_cost(tau.get_state(self._T), self._tcon(cost_params, self._T))

            # equation
            _state_prev = x0 if t == 0 else self._dynamic.forward(tau[t-1], self._tcon(dynamic_params, t-1))
            h[t*self._nx: (t+1)*self._nx] = (_state_prev - tau.get_state(t)).flatten()

            if t < self._T:
                # inequation
                g_cbf[t*self._nb: (t+1)*self._nb] = -self._barrier.forward(tau.get_state(t+1), self._tcon(barrier_params, t+1)).flatten() + (1-gamma)*self._barrier.forward(tau.get_state(t), self._tcon(barrier_params, t)).flatten()
                g_boundary[t*2*self._nu: (t+1)*2*self._nu] = torch.concat([self._umin-tau.get_control(t).flatten(), tau.get_control(t).flatten()-self._umax]) # (2*nu, )
        
        # equation & inequation residual
        res = torch.stack([
            torch.norm(h, p=1),
            torch.norm(torch.clamp(g_cbf, min=0.), p=1),
            torch.norm(torch.clamp(g_boundary, min=0.), p=1),
        ]) # (3, 1)
        if only_res: return res

        weight = torch.stack([
            self._get_weight(self._weight.get('equation'), dveq),
            self._get_weight(self._weight.get('inequation_cbf'), dvneq[:self._T*self._nb]),
            self._get_weight(self._weight.get('inequation_boundary'), dvneq[self._T*self._nb:]),
        ]).reshape(1, -1) # (1, 3)
        metric += (weight @ res).item()
        return metric

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None) -> Tensor:
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        return self

    def _pcon(self, params: Dict[str, Any], addition_params: Dict[str, Any], device: torch.device):
        return {**params, **convert_tensor_iteration(addition_params, device)}

    def _tcon(self, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        t_ts = convert_tensor_iteration(t, self._device, self._dtype)
        return {**params, **{'t': t, 't_ts': t_ts}}

    def _get_weight(self, w: float, dv: Optional[Tensor]) -> Tensor:
        w_ts = convert_tensor_iteration(w, self._device, self._dtype)
        if dv is None:
            return w_ts
        return torch.max(w_ts, 1.1*dv.abs().max())


