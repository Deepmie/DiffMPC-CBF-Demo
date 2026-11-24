from diffmpc_cbf import Cost, Dynamic, Barrier
from typing import List, Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor
from net import *

'''
state: [px, py, psi, v] (4)
control: [a, delta] (2)
'''


class HighwayCost(Cost):
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
            vref: float,
            yref: float,
        ):
        super().__init__(nx, nu, nb, T, device, dtype)
        self._vref = vref
        self._yref = yref
        self.ctx_net = ContextNetwork(4, 4).to(device, dtype)
        self._buffer_matrix()
    
    def _define_cost_function(self, tau: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - tau[T+1, ntau]: combination of x and u
            - params: some parameters of stage cost function
        Returns:
            - Cost[T+1, 1]: the cost sequence of T+1 element
        '''
        px, py, psi, v, a, delta = [tau[:, i:i+1] for i in range(self._ntau)]
        t: Tensor = params.get('t')
        ctx: Tensor = torch.concat([
            v - self._vref, # (T+1, 1)
            py - self._yref, # (T+1, 1)
            psi, # (T+1, 1)
            t.unsqueeze(-1),
        ], dim=-1) # (T+1, 4)
        hid: Tensor = self.ctx_net(ctx) # (T+1, 4)

        x = torch.concat([
            (v-self._vref).square(),
            (py-self._yref).square(),
            psi.square(),
            a.square() + delta.square(),
        ], dim=-1) # (T+1, 4)
        return 0.5 * self.w * hid * x

    def _buffer_matrix(self):
        _w: Tensor = torch.ones(self._T+1, 4, device=self._device, dtype=self._dtype)
        _w[self._T, :3] = 10.00
        _w[self._T, 3:] = 0.00
        self.w: Tensor
        self.register_buffer('w', _w.clone())


class HighwayDynamic(Dynamic):
    def __init__(
            self,
            nx: int,
            nu: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
            L: float,
            delta_t: float,
            eps: float=1e-1
        ):
        super().__init__(nx, nu, T, device, dtype)
        self._L = L
        self._delta_t = delta_t
        self._eps = eps
        self.res_net = ResidualNetwork(nx, nu).to(device, dtype)

    def _define_dynamic_equation(self, tau: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - tau[T, ntau]: combination of x and u
            - params: some parameters of dynamic equation
        Returns:
            - Next State[T, nx]: the next state sequence of T+1 element
        '''
        px, py, psi, v, a, delta = [tau[:, i:i+1] for i in range(self._ntau)]
        res: Tensor = self.res_net(tau) # (T, nu)
        a_eff = a + self._eps * res[:, 0:1] # (T, 1)
        delta_eff = delta + self._eps * res[:, 0:1] # (T, 1)
        beta_eff = torch.arctan(0.5 * torch.tan(delta_eff)) # (T, 1)
        return torch.concat([
            px + self._delta_t * v * torch.cos(psi + beta_eff),
            py + self._delta_t * v * torch.sin(psi + beta_eff),
            psi + self._delta_t * v * torch.sin(beta_eff) / (self._L / 2),
            v + self._delta_t * a_eff,
        ], dim=-1) # (T, nx)
        

class HighwayBarrier(Barrier):
    def __init__(
            self,
            nx: int,
            nu: int,
            nb: int,
            T: int,
            device: torch.device,
            dtype: torch.dtype,
            L: float,
            W: float,
            ymin: float,
            ymax: float,
        ):
        '''
        Extra Arguments:
            - L: long axis
            - W: short axis
            - ymin: min value of y
            - ymax: max value of y
        '''
        super().__init__(nx, nu, nb, T, device, dtype)
        self._L = L
        self._W = W
        self._ymin = ymin
        self._ymax = ymax
        self.dlt_net = DeltaNetwork(4).to(device, dtype)
        self._nh = self._nb - 2

    def _define_barrier_function(self, x: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - x[T, nx]: current state
            - params:
              other vehicle position, (nh, 2), [[px1, px2, ...], [py1, py2, ...]]
        Returns:
            - Next State[T, nb]: the next state sequence of T+1 element
        '''
        p, psi, v = x[:, 0:2], x[:, 2:3], x[:, 3:4]
        t: Tensor = params.get('t')
        pvc: Tensor = params.get('vehicle_position') # (T+1, nh, 2)
        t_batch = t.unsqueeze(1).expand(-1, self._nh).unsqueeze(-1) # (T+1, nh, 1)

        # vehicle cbf
        psi_cos = torch.cos(psi) # (T+1, 1)
        psi_sin = torch.sin(psi) # (T+1, 1)
        R = torch.concat([
            torch.concat([ psi_cos, psi_sin], dim=-1).unsqueeze(1), # (T+1, 1, 2)
            torch.concat([-psi_sin, psi_cos], dim=-1).unsqueeze(1), # (T+1, 1, 2)
        ], dim=1) # (T+1, 2, 2)
        dp = pvc[t, ...] - p.view(t.shape[0], 1, 2) # (T+1, nh, 2)
        d = (R.unsqueeze(1) @ dp.unsqueeze(-1)).squeeze(-1) # (T+1, nh, 2)
        dx, dy = d[..., 0], d[..., 1] # (T+1, nh)
        ctx = torch.concat([
            dp, # (T+1, nh, 2)
            dp.norm(p=2, dim=-1, keepdim=True), # (T+1, nh, 1)
            t_batch, # (T+1, nh, 1)
        ], dim=-1) # (T+1, nh, 4)
        hid: Tensor = self.dlt_net(ctx).squeeze(-1) # (T+1, nh)
        hvc = (dx / (self._L + hid)).square() + (dy / (self._W + hid)).square() - 1 # (T+1, nh)

        # road cbf
        py = p[:, 1:2] # (T+1, 1)
        return torch.concat([
            hvc, # (T+1, nh)
            (py - self._W/2) - self._ymin, # (T+1, 1)
            self._ymax - (py + self._W/2), # (T+1, 1)
        ], dim=-1) # (T+1, nh+2)