from . import Cost, Dynamic, Barrier, MetricFunction
from .base import Tau, SQPStatus
from .utils import *
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Any, cast
from qpth.qp import QPFunction


class DiffMPCBF(nn.Module):
    def __init__(
        self,
        nx: int,
        nu: int,
        nb: int,
        T: int,
        batch_size: int,
        cost: Cost,
        dynamic: Dynamic,
        barrier: Barrier,
        umin: Tensor, # (nu, )
        umax: Tensor,
        metric_weight: Dict[str, float] = {'equation': 10.0, 'inequation_cbf': 5.0, 'inequation_boundary': 5.0, 'inequation_slack': 2.0},
        slack_weight: List[float] = [1e2, 1e3],
        min_iter_nums: int = 2,
        max_iter_nums: int = 10,
        ls_alpha_init: float =1.0,
        ls_max_num: int = 5,
        ls_decay_rate: float = 0.5,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._nx = nx
        self._nu = nu
        self._nb = nb
        self._T = T
        self._batch_size = batch_size
        self._cost = cost
        self._dynamic = dynamic
        self._barrier = barrier
        self._slack_weight = slack_weight
        self._min_iter_nums = min_iter_nums
        self._max_iter_nums = max_iter_nums
        self._ls_alpha_init = ls_alpha_init
        self._ls_max_num = ls_max_num
        self._ls_decay_rate = ls_decay_rate
        self._device = device
        self._dtype = dtype

        self._ntau = nx + nu
        self._ng = 2 * (nb + nu)
        self._nd = nb + 2 * nu
        self._nv = self._ntau + nb
        self._taudim = T * self._ntau + nx
        self._slackdim = T * nb
        self._vardim = self._taudim + self._slackdimx
        self._buffer_matrix(umin, umax)

    def forward(self, x0: Tensor, params: Optional[Dict]=None):
        if not (x0.device == self._device and x0.dtype == self._dtype): raise ValueError(f'device and dtype of x is {x0.device} and {x0.dtype} respectively, no match DiffMPC-CBFs device {self._device} and dtype {self._dtype}.')
        if x0.ndimension == 1: x0 = x0.unsqueeze(-1)
        self._preprocess_params(params)
        status: SQPStatus = SQPStatus.INIT

        with torch.no_grad():
            self._reconstruct(x0, params)

            for i in range(self._max_iter_nums):
                ...
        

        self._is_first = False
        
    def reset(self):
        self._is_first: bool = True
        self._zeta = self._zeros(self._T+1, self._ntau + self._nb)
        self._dveq = self._zeros(self._T+1, self._nx)
        self._dvneq = self._zeros(self._T+1, self._ng)
        self._tau = self.zeta[:, :self._ntau]
        self._slack = self._zeta[:self._T, self._ntau:]

    def _solve_qp_subproblem(self):
        ...

    def _reconstruct(self, x0: Tensor, params: Optional[Dict]):
        self._tau[0, :self._nx] = x0
        self._tau[:self._T, self._nx:] = self._tau[:self._T, self._n].clamp(self._umin, self._umax)
        # rollout
        for t in range(self._T):
            self._tau[t+1, :self._nx] = self._dynamic.forward(self._tau[t].unsqueeze(-1), self._tcon(params['dynamic'], t)).flatten()
        if not self._is_first:
            self._dveq.zero_()
            self._dvneq.zero_()

    def _update_zero_order_term(self, _tau: Tensor, x0: Tensor, params: Optional[Dict]):
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        for t in range(self._T+1):
            _xt, _ut = _tau[t, :self._nx], _tau[t, self._nx:]
            _xt_next = _tau[t+1, :self._nx]
            # equation function
            _dymt = x0 if t == 0 else self._dynamic.forward(_tau[t-1], self._tcon(dypm, t-1)) # (nx, 1)
            self._h[t] = _xt_next -_dymt.flatten()
            if t < self._T:
                # inequation of CBF function
                self._g[t, :self._nb] = -self._barrier.forward(_xt_next.unsqueeze(-1), self._tcon(brpm, t+1)).flatten() + \
                (1-gamma) * self._barrier.forward(_xt.unsqueeze(-1), self._tcon(brpm, t)).flatten()
                # inequation of boundary function
                self._g[t, self._nb: self._nd] = torch.concat([
                    self._umin - _ut,
                    _ut - self._umax
                ]) # (2*nu, )

    def _update_one_order_term(self, _tau: Tensor, params: Optional[Dict]):
        ctpm: Dict = params.get('cost')
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        for t in range(self._T+1):
            if t < self._T:



    def _preprocess_params(self, params: Optional[Dict]=None) -> Dict:
        if params is None: params = dict()
        for key in ['cost', 'dynamic', 'barrier']:
            if key not in params: params[key] = dict()
        return params

    def _tcon(self, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        return {**params, **{'t': t, 't_ts': self._t_ts[t]}}

    def _buffer_matrix(self, umin: Tensor, umax: Tensor):
        self._umin: Tensor
        self._umax: Tensor
        self._h: Tensor
        self._g: Tensor
        self._Q: Tensor
        self._p: Tensor
        self._A: Tensor
        self._b: Tensor
        self._G: Tensor
        self._q: Tensor
        self._cx: Tensor
        self._cxT: Tensor
        self._cu: Tensor

        # register
        self.register_buffer('_umin', umin, persistent=False)
        self.register_buffer('_umax', umax, persistent=False)

        self.register_buffer('_h', self._zeros(self._T+1, self._nx))
        self.register_buffer('_g', self._zeros(self._T, self._ng))
        self.register_buffer('_Q', self._zeros(self._vardim, self._vardim))
        self.register_buffer('_p', self._zeros(self._T+1, self.nv))
        self.register_buffer('_A', self._zeros(self._T+1, self._nx, self._vardim))
        self.register_buffer('_b', self._zeros(self._T+1, self._nx))
        self.register_buffer('_G', self._zeros(self._T, self._ng, self._vardim))
        self.register_buffer('_q', self._zeros(self._T, self._ng))

        self.register_buffer('_cx', self._eye(self._ntau, self._nx), persistent=False)
        self.register_buffer('_cxT', self._eye(self._nx), persistent=False)
        self.register_buffer(
            '_cu',
            torch.concat([self._zeros(self._nx, self._nu), self._eye(self._nu)], dim=0),
            persistent=False,
        )


    def _zeros(self, *size): return torch.zeros(size, device=self._device, dtype=self._dtype)
    def _ones(self, *size): return torch.ones(size, device=self._device, dtype=self._dtype)
    def _eye(self, n: int, m: Optional[int]=None):
        if m is not None: return torch.eye(n, m, device=self._device, dtype=self._dtype)
        else: return torch.eye(n, device=self._device, dtype=self._dtype)