from diffmpc_cbf import Cost, Dynamic, Barrier
from typing import List, Dict, Tuple, Union, Optional, Any
import torch
from torch import Tensor
from net import *

'''
state: [px, py, psi, v] (4)
control: [a, delta] (2)
'''


class MetaDriveCost(Cost):
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
        vref: Tensor = params.get('vref') # float
        pref: Tensor = params.get('pref') # (T+1, 2)
        psiref: Tensor = params.get('psiref') # (T+1, 1)

        # caculate error
        c, s = torch.cos(psiref), torch.sin(psiref)
        p_delta = torch.concat([px, py], dim=-1) - pref # (T+1, 2)
        es = c * p_delta[:, 0: 1] + s * p_delta[:, 1: 2]
        en = -s * p_delta[:, 0: 1] + c * p_delta[:, 1: 2]
        epsi = torch.atan2(torch.sin(psi - psiref), torch.cos(psi - psiref))

        ctx: Tensor = torch.concat([
            es, # (T+1, 1)
            en, # (T+1, 1)
            epsi, # (T+1, 1)
            t.unsqueeze(-1), # (T+1, 1)
        ], dim=-1).detach() # (T+1, 4)
        hid: Tensor = self.ctx_net(ctx) # (T+1, 4)

        x = torch.concat([
            es.square(),
            en.square(),
            epsi.square(),
            a.square() + delta.square(),
        ], dim=-1) # (T+1, 4)
        return 0.5 * (self.w * hid * x).sum(dim=-1, keepdim=True) + 0.5 * hid[:, 0: 1] * (v - vref).square()

    def _buffer_matrix(self):
        _w: Tensor = torch.ones(self._T+1, 4, device=self._device, dtype=self._dtype)
        _w[self._T, :3] = 10.00
        _w[self._T, 3:] = 0.00
        self.w: Tensor
        self.register_buffer('w', _w.clone())


class MetaDriveDynamic(Dynamic):
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
        delta_eff = delta + self._eps * res[:, 1:2] # (T, 1)
        beta_eff = torch.arctan(0.5 * torch.tan(delta_eff)) # (T, 1)
        return torch.concat([
            px + self._delta_t * v * torch.cos(psi + beta_eff),
            py + self._delta_t * v * torch.sin(psi + beta_eff),
            psi + self._delta_t * v * torch.sin(beta_eff) / (self._L / 2),
            v + self._delta_t * a_eff,
        ], dim=-1) # (T, nx)
        

class MetaDriveBarrier(Barrier):
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
            eps: float,
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
        self._eps = eps
        self.dlt_net = DeltaNetwork(4).to(device, dtype)
        self._nh = self._nb - 2

    def _define_barrier_function(self, x: Tensor, params: Dict[str, Any]) -> Tensor:
        '''
        Arguments:
            - x[T, nx]: current state
            - params:
              other vehicle position, (T+1, nh, 2)
              pref: (T+1, 2)
              psiref: (T+1, 1)
        Returns:
            - Next State[T, nb]: the next state sequence of T+1 element
        '''
        p, psi, v = x[:, 0:2], x[:, 2:3], x[:, 3:4]
        t: Tensor      = params.get('t')
        pvc: Tensor    = params.get('vehicle_position') # (T+1, nh, 2)
        psivc: Tensor  = params.get('vehicle_heading') # (T+1, nh, 1)
        lenvc: Tensor  = params.get('vehicle_length') # (nh, )
        widvc: Tensor  = params.get('vehicle_width') # (nh, )
        maskvc: Tensor = params.get('vehicle_mask') # (nh, ) or None
        pref: Tensor   = params.get('pref') # (T+1, 2)
        psiref: Tensor = params.get('psiref') # (T+1, 1)
        n_left: Tensor = params.get('n_left') # (T+1, 1)
        n_right: Tensor = params.get('n_right') # (T+1, 1)
        
        t_batch = t.unsqueeze(1).expand(-1, self._nh).unsqueeze(-1) # (T+1, nh, 1)

        # vehicle cbf
        psi_cos, psi_sin = torch.cos(psi), torch.sin(psi) # (T+1, 1)
        R = torch.concat([
            torch.concat([ psi_cos, psi_sin], dim=-1).unsqueeze(1), # (T+1, 1, 2)
            torch.concat([-psi_sin, psi_cos], dim=-1).unsqueeze(1), # (T+1, 1, 2)
        ], dim=1) # (T+1, 2, 2)
        # p
        p_delta = pvc[t, ...] - p.view(t.shape[0], 1, 2) # (T+1, nh, 2)
        p_delta_r = (R.unsqueeze(1) @ p_delta.unsqueeze(-1)).squeeze(-1) # (T+1, nh, 2)
        x_delta, y_delta = p_delta_r[..., 0], p_delta_r[..., 1] # (T+1, nh)
        # psi
        psi_delta = psivc[t, ...] - psi.view(t.shape[0], 1, 1) # (T+1, nh, 1)
        c_rel = torch.sqrt(psi_delta.cos().square() + self._eps).squeeze(-1) # (T+1, nh)
        s_rel = torch.sqrt(psi_delta.sin().square() + self._eps).squeeze(-1) # (T+1, nh)
        
        ctx = torch.concat([
            p_delta, # (T+1, nh, 2)
            p_delta.norm(p=2, dim=-1, keepdim=True), # (T+1, nh, 1)
            t_batch, # (T+1, nh, 1)
        ], dim=-1) # (T+1, nh, 4)
        hid: Tensor = self.dlt_net(ctx).squeeze(-1) # (T+1, nh)
        a = self._L / 2 + c_rel * (lenvc / 2) + s_rel * (widvc / 2) + hid # (T+1, nh)
        b = self._W / 2 + s_rel * (lenvc / 2) + c_rel * (widvc / 2) + hid # (T+1, nh)
        hvc = (x_delta / a).square() + (y_delta / b).square() - 1 # (T+1, nh)
        if maskvc is not None: hvc = maskvc * hvc + (1 - maskvc)

        # road cbf
        c, s = torch.cos(psiref[t]), torch.sin(psiref[t]) # (T+1, 1)
        pref_delta = p - pref[t] # (T+1, 2)
        psiref_delta = psi - psiref[t] # (T+1, 1)
        en = -s * pref_delta[:, 0: 1] + c * pref_delta[:, 1: 2] # (T+1, 1)
        epsi = torch.atan2(torch.sin(psiref_delta), torch.cos(psiref_delta)) # (T+1, 1)
        body = 0.5 * self._L * torch.sqrt(epsi.sin().square() + self._eps) + 0.5 * self._W * torch.sqrt(epsi.cos().square() + self._eps) # (T+1, 1)
        return torch.concat([
            hvc, # (T+1, nh)
            n_left[t] - body - en, # (T+1, 1)
            n_right[t] - body + en, # (T+1, 1)
        ], dim=-1) # (T+1, nh+2)