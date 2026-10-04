from .. import Cost, Dynamic, iLQRMetricFunction
from .solve import lqr_step_solve, mpc_solve, build_ilqr_params
from .lqr_step import LQRStep
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, Any

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
        line_search_decay_rate: float=0.5,
    ):
        self._nx = nx; self._nu = nu; self._T = T; self._batch_size = batch_size; self._ntau = self._nx + self._nu; self._var_dims = self._T*self._ntau+self._nx
        self._cost = cost; self._dynamic = dynamic; self._umin = umin; self._umax = umax
        self._iter_nums = iter_nums; self._line_search_max_num = line_search_max_num; self._line_search_decay_rate = line_search_decay_rate
        self._metricfunc = iLQRMetricFunction(nx, nu, T, self._cost, self._dynamic)
    
    def step(self, x0: Tensor, params: Optional[Dict[str, Any]]=None) -> Tuple[Tensor]:
        if params is None: params = {}
        for key in ['cost', 'dynamic']:
            if key not in params:
                params[key] = dict()

        with torch.no_grad():
            x: Tensor = torch.zeros([self._T+1, self._nx])
            u: Tensor = torch.zeros([self._T, self._nu])
            x[0] = x0.flatten()
            for t in range(self._T):
                x[t+1] = self._dynamic.forward((x[t], u[t]), {**params.get('dynamic'), 't': t}).flatten()

            for i in range(self._iter_nums):
                Jf, Jl, Hl = build_ilqr_params(x, u, self._dynamic, self._cost, params)
                x, u = lqr_step_solve(x0, x, u, Jf, Jl, Hl, self._dynamic, params, self._metricfunc, self._line_search_max_num, self._line_search_decay_rate)
        
        x_star, u_star = x.detach(), u.detach()
        Jf, Jl, Hl = build_ilqr_params(x_star, u_star, self._dynamic, self._cost, params)
        x_out, u_out = LQRStep(x0, x_star, u_star, self._dynamic, self._cost, params, self._metricfunc, self._iter_nums, self._line_search_max_num, self._line_search_decay_rate)(Jf, Jl, Hl)
        return x_out, u_out