from diffmpc_cbf import Cost, Dynamic, Barrier
from typing import List, Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor
from net import *

class TvToCost(Cost):
    def __init__(
            self,
            nx: int,
            nu: int,
            T: int,
            kappa: float=0.5,
            context_dim: Optional[int]=None
        ):
        super().__init__(nx, nu, T)
        self._kappa = kappa
        if context_dim is None: context_dim = 3*2+2+1
        self.context_network = ContextNetwork(context_dim)

    def _define_stage_cost_function(self, ipt: Tensor, params: Dict[str, Tensor]) -> Tensor: # (6, 1)
        _device = ipt.device
        pt, vt, ut = ipt[0: 2], ipt[2: 4], ipt[4: 6]
        pref = params.get('pref') # (2, 1)
        vref = self._kappa * (pref-pt)
        t: int = params.get('t'); t_ts: Tensor = params.get('t_ts')
        _obs_info: Tuple[Tensor, Tensor] = params.get('obs_info'); _op, _or = _obs_info
        ctx = torch.concat([pt-pref, vt, (pt-_op[:, t, :].unsqueeze(-1)).reshape(-1, 1), t_ts.reshape(1, 1)]).flatten()
        hid: Tensor = self.context_network(ctx) # (3, )
        return (0.5*hid[0]*(pt-pref)**2 + 0.5*hid[1]*(vt-vref)**2 + hid[2]*ut**2 + 0.02*(pt-pref)**4).sum(dim=0) # (1, )
    
    def _define_terminal_cost_function(self, xt: Tensor, params: Dict[str, Tensor]) -> Tensor:
        _device = xt.device
        pt, vt = xt[0: 2], xt[2: 4]
        pref = params.get('pref') # (2, 1)
        vref = self._kappa * (pref-pt)
        t: int = params.get('t'); t_ts: Tensor = params.get('t_ts')
        _obs_info: Tuple[Tensor, Tensor] = params.get('obs_info'); _op, _or = _obs_info
        ctx = torch.concat([pt-pref, vt, (pt-_op[:, t, :].unsqueeze(-1)).reshape(-1, 1), t_ts.reshape(1, 1)]).flatten()
        hid: Tensor = self.context_network(ctx) # (3, )
        return (5*hid[0]*(pt-pref)**2 + 5*hid[1]*(vt-vref)**2 + 0.2*(pt-pref)**4).sum(dim=0) # (1, )

class TvToDynamic(Dynamic):
    def __init__(self, nx: int, nu: int, T: int, delta_t: float, eps: float=1e-1):
        super().__init__(nx, nu, T)
        self._delta_t = delta_t; self._eps = eps
        self.residual_network = ResidualNetwork(nx, nu)
    
    def _define_state_transition_equation(self, ipt: Tensor, params=None): # (6, 1)
        pt, vt, ut = ipt[0: 2], ipt[2: 4], ipt[4: 6]
        _res: Tensor = self.residual_network(torch.concat([pt, vt, ut]).flatten()) # (nu, )
        aeff = ut + self._eps*_res.unsqueeze(-1)
        return torch.concat([
            pt + self._delta_t*vt + 1/2*self._delta_t**2*aeff,
            vt + self._delta_t*aeff
        ])

class TvToBarrier(Barrier):
    def __init__(self, nx: int, nu: int, T: int, nb: int):
        super().__init__(nx, nu, T)
        self.delta_network = DeltaNetwork(2+3)
        self._nb = nb

    def _define_barrier_function(self, xt: Tensor, params: Dict[str, Any]) -> Tensor:
        pt = xt[0: 2]
        t: int = params.get('t'); t_ts: Tensor = params.get('t_ts')
        _obs_info: Tuple[Tensor, Tensor] = params.get('obs_info'); _op, _or = _obs_info
        radius_agent: Tensor = params.get('radius_agent')

        pt_delta = (pt - _op[:, t, :].unsqueeze(-1)) # (nb, 2, 1)
        ort = _or[:, t] # (nb, 1)
        res = torch.zeros([self._nb, 1], device=xt.device, dtype=xt.dtype)
        for i in range(self._nb):
            ctx  = torch.concat([pt_delta[i], torch.norm(pt_delta[i], p=2).reshape(1, 1), ort[i].reshape(1, 1), t_ts.reshape(1, 1)]).flatten()
            hid = self.delta_network(ctx) # (1, )
            res[i] = (pt_delta[i] ** 2).sum(dim=0) - (ort[i] + radius_agent + hid) ** 2
        return res