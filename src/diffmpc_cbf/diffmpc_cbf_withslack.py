from . import Cost, Dynamic, Barrier, MetricFunction
from .base import Tau
from .utils import *
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, Any, cast
from qpth.qp import QPFunction

class DiffMPCBFs(nn.Module):
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
        metric_weight: Dict[str, float] = {'equation': 10.0, 'inequation_cbf': 5.0, 'inequation_boundary': 5.0},
        slack_weight: List[float]=[0.2, 0.3],
        iter_nums: int = 10,
        line_search_max_num: int = 5,
        line_search_decay_rate: float = 0.5,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._batch_size = batch_size; self._ntau = self._nx + self._nu; self._var_dims = self._T*self._ntau+self._nx
        self._cost = cost; self._dynamic = dynamic; self._barrier = barrier
        self._umin = umin; self._umax = umax
        self._iter_nums = iter_nums; self._line_search_max_num = line_search_max_num; self._line_search_decay_rate = line_search_decay_rate; self._slack_weight = slack_weight
        self._metricfunc = MetricFunction(nx, nu, nb, T, self._cost, self._dynamic, self._barrier, self._umin, self._umax, metric_weight, device, dtype)
        self._tau = Tau(self._nx, self._nu, self._T)
        self._device = device; self._dtype = dtype

    def forward(self, x0: Tensor, params: Optional[Dict[str, Tensor]]=None) -> Tau:
        if not (x0.device == self._device and x0.dtype == self._dtype): raise ValueError(f'device and dtype of x is {x0.device} and {x0.dtype}, respectively, not match DiffMPC-CBFs device {self._dtype} and dtype {self._dtype}.')
        params  = self._preprocess_params(params)
        if x0.ndimension() == 1: x0 = x0.unsqueeze(-1)
        
        # iter for searching fix point tau_star
        with torch.no_grad():
            self._tau.set_state_init(x0.flatten())
            dveq: Tensor = self._get_init_dveq()
            dvneq: Tensor = self._get_init_dvneq()
            for t in range(self._T):
                self._tau.set_state(self._dynamic.forward(self._tau[t], self._tcon(params['dynamic'], t)).flatten(), t+1)
            
            for i in range(self._iter_nums):
                alpha: float = 1.0
                Q, p, G, q, A, b = self._build_sqp_params(x0, self._tau, dveq, dvneq, params)
                zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, q, A, b)
                zeta_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([zeta_delta, dveq_qp, dvneq_qp])
                tau_delta = Tau.build_from_tensor(zeta_delta[:self._var_dims], self._nx, self._ntau).to()
                dveq_delta = dveq_qp - dveq
                dvneq_delta = dvneq_qp - dvneq

                for j in range(self._line_search_max_num):
                    if self._metricfunc(self._tau + alpha*tau_delta, x0, params, dveq_qp, dvneq_qp) < self._metricfunc(self._tau, x0, params, dveq_qp, dvneq_qp):
                        break
                    alpha = self._line_search_decay_rate * alpha
                # update varible
                self._tau += alpha * tau_delta
                dveq += alpha * dveq_delta
                dvneq += alpha * dvneq_delta

        tau_star = self._tau.detach()
        dveq_star = dveq.detach()
        dvneq_star = dvneq.detach()
        # solve the last qp
        Hl, Jl, Jg, g, Jh, h = self._build_sqp_params(x0, tau_star, dveq_star, dvneq_star, params)
        zeta_delta, _, _ = QPFunction(eps=1e-6, verbose=0, maxIter=25)(Hl, Jl, Jg, -g, Jh, -h); zeta_delta = cast(Tensor, zeta_delta)
        tau_delta = Tau.build_from_tensor(zeta_delta.flatten()[:self._var_dims], self._nx, self._ntau).to()
        return tau_star + tau_delta

    def _build_sqp_params(
            self,
            x0: Tensor,
            _tau: Tau,
            dveq: Tensor,
            dvneq: Tensor,
            params: Optional[Dict[str, Dict[str, Any]]]
        ) -> Tuple[Tensor]:
        def _torch_zeros(size): return torch.zeros(size, device=self._device, dtype=self._dtype)
        def _torch_ones(size): return torch.ones(size, device=self._device, dtype=self._dtype)
        def _torch_eyes(n): return torch.eye(n, device=self._device, dtype=self._dtype)

        Jl: Tensor          = _torch_zeros([self._var_dims])
        Hl: Tensor          = _torch_zeros([self._var_dims, self._var_dims])
        Jh: Tensor          = _torch_zeros([(self._T+1)*self._nx, self._var_dims])
        h: Tensor           = _torch_zeros([(self._T+1)*self._nx])
        Jg_cbf: Tensor      = _torch_zeros([self._T*self._nb, self._var_dims])
        g_cbf: Tensor       = _torch_zeros([self._T*self._nb])
        Jg_boundary: Tensor = _torch_zeros([self._T*2*self._nu, self._var_dims])
        g_boundary: Tensor  = _torch_zeros([self._T*2*self._nu])
        
        # build aux variable
        cx: Tensor  = torch.concat([_torch_eyes(self._nx), _torch_zeros([self._nu, self._nx])], dim=0) # (ntau, nx)
        cxT: Tensor = _torch_eyes(self._nx)
        cu: Tensor  = torch.concat([_torch_zeros([self._nx, self._nu]), _torch_eyes(self._nu)], dim=0) # (ntau, nu)
        
        cost_params: Dict = params.get('cost')
        dynamic_params: Dict = params.get('dynamic')
        barrier_params: Dict = params.get('barrier')
        gamma: float = barrier_params.get('gamma')
        for t in range(self._T+1):
            # cost Jacobian
            if t < self._T: Jl[t*self._ntau: (t+1)*self._ntau] = self._cost.get_stage_cost(_tau[t], self._tcon(cost_params, t), order=1).flatten()
            if t == self._T: Jl[self._T*self._ntau: self._var_dims] = self._cost.get_terminal_cost(_tau.get_state(t), self._tcon(cost_params, self._T), order=1).flatten()

            # cost Hessian
            if t < self._T:
                Hl[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] = \
                self._cost.get_stage_cost(_tau[t], self._tcon(cost_params, t), order=2) + \
                (dveq[(t+1)*self._nx: (t+2)*self._nx].reshape(-1, 1, 1) * self._dynamic.forward(_tau[t], self._tcon(dynamic_params, t), order=2)).sum(dim=0)

            if t == self._T:
                Hl[self._T*self._ntau: self._var_dims, self._T*self._ntau: self._var_dims] = self._cost.get_terminal_cost(_tau.get_state(self._T), self._tcon(cost_params, self._T), order=2)

            if t == 0:
                Hl[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] += \
                ((1-gamma) * dvneq[t*self._nb: (t+1)*self._nb].reshape(-1, 1, 1) * \
                cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=2) @ cx.T.unsqueeze(0)).sum(dim=0)
            else:
                _cx = cx if t < self._T else cxT
                Hl[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] += \
                ((-dvneq[(t-1)*self._nb: t*self._nb] + (1-gamma) * dvneq[t*self._nb: (t+1)*self._nb]).reshape(-1, 1, 1) * \
                _cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=2) @ _cx.T.unsqueeze(0)).sum(dim=0)
            
            # equation Jacobian
            Jh[t*self._nx: (t+1)*self._nx, t*self._ntau: (t+1)*self._ntau] = -cx.T if t < self._T else -cxT.T
            if t < self._T: Jh[(t+1)*self._nx: (t+2)*self._nx, t*self._ntau: (t+1)*self._ntau] = self._dynamic.forward(_tau[t], self._tcon(dynamic_params, t), order=1)

            # equation function
            _state_prev = x0 if t == 0 else self._dynamic.forward(_tau[t-1], self._tcon(dynamic_params, t-1))
            h[t*self._nx: (t+1)*self._nx] = (_state_prev - _tau.get_state(t)).flatten()

            if t < self._T:
                # inequation of CBF Jacobian
                Jg_cbf[t*self._nb: (t+1)*self._nb, t*self._ntau: (t+1)*self._ntau] = (1-gamma)*self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=1) @ cx.T
                Jg_cbf[t*self._nb: (t+1)*self._nb, (t+1)*self._ntau: (t+2)*self._ntau] = -self._barrier.forward(_tau.get_state(t+1), self._tcon(barrier_params, t), order=1) @ (cx.T if t < self._T-1 else cxT.T)
                # inequation of CBF function
                g_cbf[t*self._nb: (t+1)*self._nb] = -self._barrier.forward(_tau.get_state(t+1), self._tcon(barrier_params, t+1)).flatten() + (1-gamma)*self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t)).flatten()

                # inequation of boundary Jacobian
                Jg_boundary[t*2*self._nu: (t+1)*2*self._nu, t*self._ntau: (t+1)*self._ntau] = torch.concat([-cu.T, cu.T], dim=0)
                # inequation of boundary function
                g_boundary[t*2*self._nu: (t+1)*2*self._nu] = torch.concat([self._umin-_tau.get_control(t).flatten(), _tau.get_control(t).flatten()-self._umax]) # (2*nu, )

        # post process
        # slack matrix
        rho1, rho2 = self._slack_weight
        ns = self._T*self._nb
        # ensure positive matrix
        Hl = 0.5 * (Hl + Hl.T)
        eig_min = torch.linalg.eigvalsh(Hl).min()
        eps = 1e-4
        if eig_min < eps: Hl += (eps-eig_min) * _torch_eyes(self._var_dims)

        # merge inequation Jacobian & function
        Q = torch.concat([
            torch.concat([Hl, _torch_zeros([self._var_dims, ns])], dim=1),
            torch.concat([_torch_zeros([ns, self._var_dims]), rho2*_torch_eyes(ns)], dim=1),
        ], dim=0)
        p = torch.concat([Jl, rho1*_torch_ones([ns])])
        A = torch.concat([Jh, _torch_zeros([Jh.shape[0], ns])], dim=1)
        b = -h

        G = torch.concat([
            torch.concat([Jg_cbf, -_torch_eyes(ns)], dim=1), # (ns, var_dims+ns)
            torch.concat([Jg_boundary, _torch_zeros([Jg_boundary.shape[0], ns])], dim=1), # (xxx, var_dims+ns)
            torch.concat([_torch_zeros([ns, self._var_dims]), -_torch_eyes(ns)], dim=1), # (ns, var_dims+ns)
        ], dim=0) # (2ns+xxx)
        q = torch.concat([-g_cbf, -g_boundary, _torch_zeros([ns])], dim=0)
        return Q, p, G, q, A, b

    def _preprocess_params(self, params: Optional[Dict[str, Any]]) -> Dict:
        if params is None:
            params = dict()

        for key in ['cost', 'dynamic', 'barrier']:
            if key not in params:
                params[key] = dict()
        # return convert_tensor_iteration(params, device)
        return params

    def _tcon(self, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        t_ts = convert_tensor_iteration(t, self._device, self._dtype)
        return {**params, **{'t': t, 't_ts': t_ts}}

    def _pcon(self, params: Dict[str, Any], addition_params: Dict[str, Any]):
        return {**params, **convert_tensor_iteration(addition_params, self._device, self._dtype)}
    
    def _qp_var_post_process(self, vs: List[Tensor]) -> List[Tensor]:
        return [v.flatten() for v in vs]
    
    def _get_init_dveq(self) -> Tensor:
        return torch.zeros([(self._T+1)*self._nx, ], device=self._device, dtype=self._dtype)

    def _get_init_dvneq(self) -> Tensor:
        return torch.zeros([self._T*(2*self._nb+2*self._nu), ], device=self._device, dtype=self._dtype)

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        self._tau.to(device, dtype)
        return super().to(device, dtype)

    @property
    def cost(self):
        return self._cost

    @property
    def dynamic(self):
        return self._dynamic