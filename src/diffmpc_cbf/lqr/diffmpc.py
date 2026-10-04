from .. import Cost, Dynamic, iLQRMetricFunction
from .solve import mpc_solve
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
        return mpc_solve(
            x0,
            self._nx,
            self._nu,
            self._T,
            self._dynamic,
            self._cost, cost_params,
            self._metricfunc,
            self._iter_nums,
            self._line_search_max_num
        )