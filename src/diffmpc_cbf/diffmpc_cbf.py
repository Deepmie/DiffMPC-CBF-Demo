from . import Cost, Dynamic, Barrier, MetricFunction
from .base import Tau
from .utils import *
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, Any, cast
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
        slack_weight: List[float] = [0.2, 0.3],
        min_iter_nums: int = 2,
        max_iter_nums: int = 10,
        line_search_max_num: int = 3,
        line_search_decay_rate: float = 0.5,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._batch_size = batch_size; self._ntau = self._nx + self._nu; self._var_dims = self._T*self._ntau+self._nx; self._ns = self._T * self._nb
        self._cost = cost; self._dynamic = dynamic; self._barrier = barrier
        self._umin = umin; self._umax = umax
        self._min_iter_nums = min_iter_nums; self._max_iter_nums = max_iter_nums
        self._line_search_max_num = line_search_max_num; self._line_search_decay_rate = line_search_decay_rate; self._slack_weight = slack_weight
        self._metricfunc = MetricFunction(nx, nu, nb, T, self._cost, self._dynamic, self._barrier, self._umin, self._umax, metric_weight, device, dtype)
        self._tau = Tau(self._nx, self._nu, self._T, device=device, dtype=dtype)
        self._t_ts: Tensor = torch.arange(0, self._T+1, step=1, device=device, dtype=dtype)
        self._device = device; self._dtype = dtype
        self.reset()

    def forward(self, x0: Tensor, params: Optional[Dict[str, Tensor]]=None) -> Tau:
        if not (x0.device == self._device and x0.dtype == self._dtype): raise ValueError(f'device and dtype of x is {x0.device} and {x0.dtype}, respectively, not match DiffMPC-CBFs device {self._device} and dtype {self._dtype}.')
        params  = self._preprocess_params(params)
        if x0.ndimension() == 1: x0 = x0.unsqueeze(-1)
        
        # iter for searching fix point tau_star
        with torch.no_grad():
            self._tau.set_state_init(x0.flatten(), need_shift= not self._is_first)
            self._set_init_slack()
            self._set_init_dveq()
            self._set_init_dvneq()
            for t in range(self._T):
                self._tau.set_state(self._dynamic.forward(self._tau[t], self._tcon(params['dynamic'], t)).flatten(), t+1)
            
            for i in range(self._max_iter_nums):
                alpha: float = 1.0
                accept_alpha: bool = False
                Q, p, G, q, A, b = self._build_sqp_params(x0, self._tau, self._slack, self._dveq, self._dvneq, params, zero_order=True, one_order=True, two_order=True)
                zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-5, verbose=0, maxIter=10)(Q, p, G, q, A, b)
                zeta_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([zeta_delta, dveq_qp, dvneq_qp])
                tau_delta = Tau.build_from_tensor(zeta_delta[:self._var_dims], self._nx, self._ntau)
                slack_delta = zeta_delta[self._var_dims:]
                dveq_delta = dveq_qp - self._dveq
                dvneq_delta = dvneq_qp - self._dvneq

                h, g_cbf, g_boundary = self._build_sqp_params(self._tau, self._slack, dveq_qp, dvneq_qp, x0, params, zero_order=True)
                metric_curr, res = self._metricfunc(self._tau, self._slack, dveq_qp, dvneq_qp, h, g_cbf, g_boundary, x0, params, self._slack_weight, include_res=True)
                # Early Stop
                if self._early_stop(i, res, tau_delta, slack_delta, dveq_qp, dvneq_qp, dveq_delta, dvneq_delta, x0, params):
                    break

                # Line Search
                alpha = self._line_search(metric_curr, tau_delta, slack_delta, dveq_qp, dvneq_qp, x0, params)
                
                # update varible
                self._tau   += alpha * tau_delta
                self._slack += alpha * slack_delta
                self._dveq  += alpha * dveq_delta
                self._dvneq += alpha * dvneq_delta

        tau_star = self._tau.detach()
        slack_star = self._slack.detach()
        dveq_star = self._dveq.detach()
        dvneq_star = self._dvneq.detach()
        # solve the last qp
        Q, p, G, q, A, b = self._build_sqp_params(x0, tau_star, slack_star, dveq_star, dvneq_star, params, zero_order=True, one_order=True, two_order=True)
        zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-6, verbose=0, maxIter=25)(Q, p, G, q, A, b)
        zeta_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([zeta_delta, dveq_qp, dvneq_qp])
        
        tau_delta = Tau.build_from_tensor(zeta_delta[:self._var_dims], self._nx, self._ntau); slack_delta = zeta_delta[self._var_dims:]
        tau_final = tau_star + tau_delta
        slack_final = slack_star + slack_delta

        self._tau = tau_final.detach()
        self._slack = slack_final.detach().clone()
        self._dveq = dveq_qp.detach().clone()
        self._dvneq = dvneq_qp.detach().clone()
        if self._is_first: self._is_first = False # change state
        return tau_final

    def _build_sqp_params(
            self,
            _tau: Tau,
            _slack: Tensor,
            dveq: Tensor,
            dvneq: Tensor,
            x0: Tensor,
            params: Optional[Dict[str, Dict[str, Any]]],
            zero_order: bool = False,
            one_order: bool = False,
            two_order: bool = False,
        ) -> Optional[Tuple[Tensor]]:
        if (not zero_order) and (not one_order) and (not two_order):
            return 

        if zero_order: # 0-order item
            h: Tensor           = self._torch_zeros([(self._T+1)*self._nx])
            g_cbf: Tensor       = self._torch_zeros([self._T*self._nb])
            g_boundary: Tensor  = self._torch_zeros([self._T*2*self._nu])

        if one_order:  # 1-order item
            Jl: Tensor          = self._torch_zeros([self._var_dims])
            Jh: Tensor          = self._torch_zeros([(self._T+1)*self._nx, self._var_dims])
            Jg_cbf: Tensor      = self._torch_zeros([self._T*self._nb, self._var_dims])
            Jg_boundary: Tensor = self._torch_zeros([self._T*2*self._nu, self._var_dims])

        if two_order:  # 2-order item
            Hl: Tensor          = self._torch_zeros([self._var_dims, self._var_dims])
        
        # build aux variable
        cx: Tensor    = torch.concat([self._torch_eyes(self._nx), self._torch_zeros([self._nu, self._nx])], dim=0) # (ntau, nx)
        cxT: Tensor   = self._torch_eyes(self._nx)
        cu: Tensor    = torch.concat([self._torch_zeros([self._nx, self._nu]), self._torch_eyes(self._nu)], dim=0) # (ntau, nu)
        cvI: Tensor   = self._torch_eyes(self._var_dims)
        csZv: Tensor  = self._torch_zeros([self._ns, self._var_dims])
        csZh: Tensor  = self._torch_zeros([self._ns, (self._T+1)*self._nx])
        csZgb: Tensor = self._torch_zeros([self._ns, self._T*2*self._nu])
        csI: Tensor   = self._torch_eyes(self._ns)
        csO: Tensor   = self._torch_ones([self._ns])
        
        cost_params: Dict = params.get('cost')
        dynamic_params: Dict = params.get('dynamic')
        barrier_params: Dict = params.get('barrier')
        gamma: float = barrier_params.get('gamma')
        
        for t in range(self._T+1):
            if zero_order:
                # equation function
                _state_prev = x0 if t == 0 else self._dynamic.forward(_tau[t-1], self._tcon(dynamic_params, t-1))
                h[t*self._nx: (t+1)*self._nx] = (_state_prev - _tau.get_state(t)).flatten()
                if t < self._T:
                    # inequation of CBF function
                    g_cbf[t*self._nb: (t+1)*self._nb] = -self._barrier.forward(_tau.get_state(t+1), self._tcon(barrier_params, t+1)).flatten() + (1-gamma)*self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t)).flatten()
                    # inequation of boundary function
                    g_boundary[t*2*self._nu: (t+1)*2*self._nu] = torch.concat([self._umin-_tau.get_control(t).flatten(), _tau.get_control(t).flatten()-self._umax]) # (2*nu, )

            if one_order:
                # cost Jacobian
                if t < self._T: Jl[t*self._ntau: (t+1)*self._ntau] = self._cost.get_stage_cost(_tau[t], self._tcon(cost_params, t), order=1).flatten()
                if t == self._T: Jl[self._T*self._ntau: self._var_dims] = self._cost.get_terminal_cost(_tau.get_state(t), self._tcon(cost_params, self._T), order=1).flatten()
                # equation Jacobian
                Jh[t*self._nx: (t+1)*self._nx, t*self._ntau: (t+1)*self._ntau] = -cx.T if t < self._T else -cxT.T
                if t < self._T: Jh[(t+1)*self._nx: (t+2)*self._nx, t*self._ntau: (t+1)*self._ntau] = self._dynamic.forward(_tau[t], self._tcon(dynamic_params, t), order=1)
                # inequation Jacobian
                if t < self._T:
                    # inequation of CBF Jacobian
                    Jg_cbf[t*self._nb: (t+1)*self._nb, t*self._ntau: (t+1)*self._ntau] = (1-gamma)*self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=1) @ cx.T
                    Jg_cbf[t*self._nb: (t+1)*self._nb, (t+1)*self._ntau: (t+2)*self._ntau] = -self._barrier.forward(_tau.get_state(t+1), self._tcon(barrier_params, t+1), order=1) @ (cx.T if t < self._T-1 else cxT.T)
                    # inequation of boundary Jacobian
                    Jg_boundary[t*2*self._nu: (t+1)*2*self._nu, t*self._ntau: (t+1)*self._ntau] = torch.concat([-cu.T, cu.T], dim=0)

            if two_order:
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
                    _dvneq_t = dvneq[t*self._nb: (t+1)*self._nb] if t < self._T else 0.0
                    Hl[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] += \
                    ((-dvneq[(t-1)*self._nb: t*self._nb] + (1-gamma) * _dvneq_t).reshape(-1, 1, 1) * \
                    _cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=2) @ _cx.T.unsqueeze(0)).sum(dim=0)

        # post process
        # slack matrix
        rho1, rho2 = self._slack_weight

        if two_order:
            # ensure positive matrix
            Hl = 0.5 * (Hl + Hl.T)
            eig_min = torch.linalg.eigvalsh(Hl).min()
            eps = 1e-4
            if eig_min < eps: Hl += (eps-eig_min) * cvI

        # diff usage
        if zero_order and (not one_order) and (not two_order):
            # build for MetricFunction
            return h, g_cbf, g_boundary
        elif (not zero_order) and one_order and (not two_order):
            # build for KKT residual
            return Jl, Jh, Jg_cbf, Jg_boundary 
        elif zero_order and one_order and two_order:
            # build for SQP problem
            # merge inequation Jacobian & function
            Q = torch.concat([
                torch.concat([Hl, csZv.T], dim=1),
                torch.concat([csZv, rho2*csI], dim=1),
            ], dim=0)
            p = torch.concat([Jl, rho1*csO + rho2*_slack])
            A = torch.concat([Jh, csZh.T], dim=1)
            b = -h

            G = torch.concat([
                torch.concat([Jg_cbf, -csI], dim=1), # (ns, var_dims+ns)
                torch.concat([Jg_boundary, csZgb.T], dim=1), # (xxx, var_dims+ns)
                torch.concat([csZv, -csI], dim=1), # (ns, var_dims+ns)
            ], dim=0) # (2ns+xxx)
            q = torch.concat([-g_cbf+_slack, -g_boundary, _slack], dim=0)
            return Q, p, G, q, A, b

    def reset(self):
        self._is_first: bool = True

    def _early_stop(
            self,
            iteration_index: int,
            residual: Tensor,
            tau_delta: Tau,
            slack_delta: Tensor,
            dveq_delta: Tensor,
            dvneq_delta: Tensor,
        ) -> bool:
        con: Tensor = torch.concat([
            torch.stack([
                tau_delta.norm(p=float('inf')),
                slack_delta.norm(p=float('inf')),
                torch.norm(dveq_delta, p=float('inf')),
                torch.norm(dvneq_delta, p=float('inf')),
            ]).unsqueeze(-1),
            residual,
        ], dim=0) # (8, 1)
        eps = torch.concat([
            1e-3 * self._torch_ones([2, 1]),
            1e-2 * self._torch_ones([6, 1]),
        ], dim=0) # (8, 1)

        if iteration_index >= self._min_iter_nums-1 and (con < eps).all():
            return True
        return False

    def _line_search(
            self,
            metric_curr: Tensor,
            tau_delta: Tau,
            slack_delta: Tensor,
            dveq_qp: Tensor,
            dvneq_qp: Tensor,
            x0: Tensor,
            params: Optional[Dict[str, Tensor]],
        ) -> float:
        for j in range(self._line_search_max_num):
            _tau_new = self._tau + alpha*tau_delta; _slack_new = self._slack + alpha*slack_delta
            # compute new metrics
            h, g_cbf, g_boundary = self._build_sqp_params(_tau_new, _slack_new, dveq_qp, dvneq_qp, x0, params, zero_order=True)
            metric_new = self._metricfunc(_tau_new, _slack_new, dveq_qp, dvneq_qp, h, g_cbf, g_boundary, x0, params, self._slack_weight)
            # compare
            if metric_new < metric_curr:
                accept_alpha = True
                break
            alpha = self._line_search_decay_rate * alpha
        if not accept_alpha:
            alpha = 0.0 # not update
        return alpha

    def _preprocess_params(self, params: Optional[Dict[str, Any]]) -> Dict:
        if params is None:
            params = dict()

        for key in ['cost', 'dynamic', 'barrier']:
            if key not in params:
                params[key] = dict()
        # return convert_tensor_iteration(params, device)
        return params

    def _tcon(self, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        return {**params, **{'t': t, 't_ts': self._t_ts[t]}}

    def _pcon(self, params: Dict[str, Any], addition_params: Dict[str, Any]):
        return {**params, **convert_tensor_iteration(addition_params, self._device, self._dtype)}

    def _torch_zeros(self, size):
        return torch.zeros(size, device=self._device, dtype=self._dtype)
    
    def _torch_ones(self, size):
        return torch.ones(size, device=self._device, dtype=self._dtype)
    
    def _torch_eyes(self, n):
        return torch.eye(n, device=self._device, dtype=self._dtype)
    
    def _qp_var_post_process(self, vs: List[Tensor]) -> List[Tensor]:
        return [v.flatten() for v in vs]

    def _set_init_slack(self) -> Tensor:
        if self._is_first:
            self._slack = torch.zeros([self._ns ], device=self._device, dtype=self._dtype)
        else:
            self._slack[:(self._T-1)*self._nb] = self._slack[self._nb: self._T*self._nb].clone()
        return self._slack
    
    def _set_init_dveq(self) -> Tensor:
        if self._is_first:
            self._dveq = torch.zeros([(self._T+1)*self._nx, ], device=self._device, dtype=self._dtype)
        else:
            self._dveq[:self._T*self._nx] = self._dveq[self._nx: (self._T+1)*self._nx].clone()
        return self._dveq

    def _set_init_dvneq(self) -> Tensor:
        if self._is_first:
            self._dvneq = torch.zeros([self._T*(2*self._nb+2*self._nu), ], device=self._device, dtype=self._dtype)
        else:
            self._dvneq[:(self._T-1)*self._nb] = self._dvneq[self._nb: self._T*self._nb].clone(); _bias = self._T*self._nb
            self._dvneq[_bias: _bias+(self._T-1)*2*self._nu] = self._dvneq[_bias+2*self._nu: _bias+self._T*2*self._nu].clone(); _bias += self._T*2*self._nu
            self._dvneq[_bias: _bias+(self._T-1)*self._nb] = self._dvneq[_bias+self._nb: _bias+self._T*self._nb].clone()
        return self._dvneq

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        self._tau.to(device, dtype)
        self._metricfunc.to(device, dtype)
        if device is not None: self._device = device
        if dtype is not None:  self._dtype = dtype
        return super().to(device, dtype)

    @property
    def cost(self):
        return self._cost

    @property
    def dynamic(self):
        return self._dynamic