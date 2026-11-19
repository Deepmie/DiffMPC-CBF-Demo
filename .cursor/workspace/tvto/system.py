from diffmpc_cbf_cs import Cost, Dynamic, Barrier
from typing import List, Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor
from tvto.net import *

class TvToCost(Cost):
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
            kappa: float=0.5,
            ctx_dim: Optional[int]=None,
        ):
        super().__init__(nx, nu, nb, T, device, dtype)
        self._kappa = kappa
        self._slackdim = nb * T
        if ctx_dim is None: ctx_dim = 2 + 2 + nb * 2 + 1
        self.ctx_net = ContextNetwork(ctx_dim).to(device, dtype)
        self._buffer_matrix()

    def _define_cost_function(self, tau: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - tau[T+1, ntau]: combination of x and u
            - params: some parameters of stage cost function
        Returns:
            - Cost[T+1, 1]: the cost sequence of T+1 element
        '''
        p, v, u = tau[:, 0: 2], tau[:, 2: 4], tau[:, 4: 6]
        pref: Tensor = params.get('pref').flatten() # (2, )
        vref = self._kappa * (pref - p) # (T+1, 2)
        t: Tensor = params.get('t')
        obs_info: Tuple[Tensor, Tensor] = params.get('obs_info')
        po, ro = obs_info # (nb, T+1, 2), (nb, T+1)

        ctx = torch.concat([
            p - pref, # (T+1, 2)
            v, # (T+1, 2)
            *[poi for poi in (p - po)], # nb * (T+1, 2)
            t.unsqueeze(-1), # (T+1, 1)
        ], dim=1) # (T+1, ctxdim)
        hid: Tensor = self.ctx_net(ctx) # (T+1, 3)
        res = (0.5 * (self._w * hid[:, 0]).unsqueeze(-1) * (p-pref).square()).sum(dim=-1) + \
               (0.5 * (self._w * hid[:, 1]).unsqueeze(-1) * (v-vref).square()).sum(dim=-1) + \
               (t < self._T).to(u.dtype) * (hid[:, 2].unsqueeze(-1) * u.square()).sum(dim=-1) + \
               (0.02 * self._w.unsqueeze(-1) * (p-pref)**4).sum(dim=-1)
        return res.unsqueeze(-1)

    def _buffer_matrix(self):
        self._w: Tensor
        self.register_buffer('_w', torch.tensor([1.0 if t < self._T else 10.0 for t in range(self._T+1)]))


class TvToDynamic(Dynamic):
    def __init__(
            self,
            nx: int,
            nu: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
            delta_t: float,
            eps: float=1e-1
        ):
        super().__init__(nx, nu, T, device, dtype)
        self._delta_t = delta_t; self._eps = eps
        self.res_net = ResidualNetwork(nx, nu).to(device, dtype)

    def _define_dynamic_equation(self, tau: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - tau[T, ntau]: combination of x and u
            - params: some parameters of dynamic equation
        Returns:
            - Next State[T, nx]: the next state sequence of T+1 element
        '''
        p, v, u = tau[:, 0: 2], tau[:, 2: 4], tau[:, 4: 6]
        res: Tensor = self.res_net(tau) # (T, nu)
        aeff = u + self._eps * res # (T, nu)
        return torch.concat([
            p + self._delta_t * v + 0.5 * self._delta_t**2 * aeff, # (T, 2)
            v + self._delta_t * aeff, # (T, 2)
        ], dim=-1) # (T, nx)


class TvToBarrier(Barrier):
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
        ):
        super().__init__(nx, nu, nb, T, device, dtype)
        self.dlt_net = DeltaNetwork(2+3).to(device, dtype)
    
    def _define_barrier_function(self, x: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - x[T, nx]: current state
            - params: some parameters of barrier function
        Returns:
            - Next State[T, nb]: the next state sequence of T+1 element
        '''
        p = x[:, 0: 2]
        t: Tensor = params.get('t')
        obs_info: Tuple[Tensor, Tensor] = params.get('obs_info')
        po, ro = obs_info # (nb, T+1, 2), (nb, T+1)
        ra: Tensor = params.get('radius_agent') # (1, )
        p_delta = p - po[:, t, :] # (nb, T, 2)
        dist = p_delta.norm(p=2, dim=-1, keepdim=True) # (nb, T, 1)
        t_batch = t.unsqueeze(0).expand(self._nb, -1).unsqueeze(-1) # (nb, T, 1)
        ctx = torch.concat([
            p_delta,
            dist,
            ro[:, t].unsqueeze(-1),
            t_batch,
        ], dim=-1) # (nb, T, 5)
        hid: Tensor = self.dlt_net(ctx) # (nb, T, 1)
        res = dist.squeeze(-1) - ((ro[:, t] + ra).unsqueeze(-1) + hid).norm(p=2, dim=-1) # (nb, T)
        return res.permute(1, 0) # (T, nb)

