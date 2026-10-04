from .. import Cost, Dynamic, iLQRMetricFunction
from .lqr_step import LQRStep
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional

class DiffMPC:
    def __init__(
        self,
        nx: int,
        nu: int,
        T: int,
        batch_size: int,
        cost: Cost,
        dynamic: Dynamic,
        umin: Tensor, # (nu,)
        umax: Tensor,
        iter_nums: int=10,
        line_search_max_num: int=5,
    ):
        self._nx = nx; self._nu = nu; self._T = T; self._batch_size = batch_size; self._ntau = self._nx + self._nu; self._var_dims = self._T*self._ntau+self._nx
        self._cost = cost; self._dynamic = dynamic; self._umin = umin; self._umax = umax
        self._iter_nums = iter_nums; self._line_search_max_num = line_search_max_num
        self._metricfunc = iLQRMetricFunction(nx, nu, T, self._cost, self._dynamic)
    
    def step(self, x0: Tensor, cost_params: Optional[Dict[str, Tensor]]) -> Tuple[Tensor]:
        with torch.no_grad():
            x: Tensor = torch.zeros([self._T+1, self._nx])
            u: Tensor = torch.zeros([self._T, self._nu])
            x[0] = x0
            for t in range(self._T):
                x[t+1] = self._dynamic.forward((x[t], u[t])).flatten()

            for i in range(self._iter_nums):
                Jf, Jl, Hl = self._build_ilqr_params(x, u, cost_params)
                x, u = LQRStep(x0, x, u, self._dynamic, cost_params, self._metricfunc, self._line_search_max_num)(Jf, Jl, Hl)

        x_star, u_star = x.detach(), u.detach()
        Jf, Jl, Hl = self._build_ilqr_params(x_star, u_star, cost_params)
        x_out, u_out = LQRStep(x0, x_star, u_star, self._dynamic, cost_params, self._metricfunc, self._line_search_max_num)(Jf, Jl, Hl)
        return x_out, u_out

    def _build_ilqr_params(
            self,
            x: Tensor,
            u: Tensor,
            cost_params: Optional[Dict[str, Tensor]],
        ) -> Tuple[Tensor]:
        Jf = torch.zeros([self._T, self._nx, self._ntau])
        Jl = torch.zeros([self._T+1, self._ntau])
        Hl = torch.zeros([self._T+1, self._ntau, self._ntau])

        for t in range(self._T):
            Jf[t, :, :] = self._dynamic.forward((x[t], u[t]), order=1)

        for t in range(self._T):
            Jl[t, :] = self._cost.get_stage_cost((x[t], u[t]), cost_params, order=1).flatten()
        Jl[self._T, :self._nx] = self._cost.get_terminal_cost(x[self._T], cost_params, order=1).flatten()

        for t in range(self._T):
            Hl[t, :, :] = self._cost.get_stage_cost((x[t], u[t]), cost_params, order=2)
        Hl[t, :self._nx, :self._nx] = self._cost.get_terminal_cost(x[self._T], cost_params, order=2)
        return Jf, Jl, Hl