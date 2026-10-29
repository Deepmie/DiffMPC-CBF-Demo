from . import Cost, Dynamic, Barrier, MetricFunction
from .base import Tau, SQPStatus
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
        start_dv_eps_detection: bool = False,
        line_search_alpha_init: float=1.0,
        line_search_max_num: int = 3,
        line_search_decay_rate: float = 0.5,
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._batch_size = batch_size; self._ntau = self._nx + self._nu; self._var_dims = self._T*self._ntau+self._nx; self._ns = self._T * self._nb
        self._cost = cost; self._dynamic = dynamic; self._barrier = barrier
        self._min_iter_nums = min_iter_nums; self._max_iter_nums = max_iter_nums
        self._start_dv_eps_detection = start_dv_eps_detection
        self._line_search_alpha_init = line_search_alpha_init; self._line_search_max_num = line_search_max_num;
        self._line_search_decay_rate = line_search_decay_rate; self._slack_weight = slack_weight
        self._device = device; self._dtype = dtype
        
        self._get_aux_matrix(umin.clone(), umax.clone())
        self._metricfunc = MetricFunction(nx, nu, nb, T, self._cost, self._dynamic, self._barrier, self._t_ts, metric_weight, device, dtype)
        self._tau = Tau(self._nx, self._nu, self._T, device=device, dtype=dtype)
        self.reset()

    def forward(self, x0: Tensor, params: Optional[Dict[str, Tensor]]=None) -> Tuple[Tau, SQPStatus]:
        if not (x0.device == self._device and x0.dtype == self._dtype): raise ValueError(f'device and dtype of x is {x0.device} and {x0.dtype}, respectively, not match DiffMPC-CBFs device {self._device} and dtype {self._dtype}.')
        params  = self._preprocess_params(params)
        sqp_status: Optional[SQPStatus] = None

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
                accept_alpha: bool = False
                Q, p, G, q, A, b = self._get_sqp_matrix(self._tau, self._slack, self._dveq, self._dvneq, x0, params)
                zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-5, verbose=0, maxIter=10)(Q, p, G, q, A, b)
                zeta_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([zeta_delta, dveq_qp, dvneq_qp])
                tau_delta = Tau.build_from_tensor(zeta_delta[:self._var_dims], self._nx, self._ntau)
                slack_delta = zeta_delta[self._var_dims:]
                dveq_delta = dveq_qp - self._dveq
                dvneq_delta = dvneq_qp - self._dvneq

                h, g_cbf, g_boundary = self._get_zero_order_term(self._tau, x0, params)
                metric_curr, res = self._metricfunc(self._tau, self._slack, dveq_qp, dvneq_qp, h, g_cbf, g_boundary, x0, params, self._slack_weight, include_res=True)
                
                # Early Stop
                els_cdt: Tensor = torch.concat([
                    torch.stack([
                        tau_delta.norm(p=float('inf')),
                        slack_delta.norm(p=float('inf')),
                    ]).unsqueeze(-1),
                    res,
                ], dim=0) # (6, 1)

                if self._start_dv_eps_detection:
                    dv_cdt: Tensor = torch.stack([
                        dveq_delta.norm(p=float('inf')),
                        dvneq_delta.norm(p=float('inf')),
                    ]).unsqueeze(-1) # (2, 1)
                    els_cdt = torch.concat([els_cdt, dv_cdt], dim=0)

                if i >= self._min_iter_nums-1 and (els_cdt < self._els_eps).all():
                    # maybe break
                    # compute KKT residual
                    _tau_new = self._tau + tau_delta; _slack_new = self._slack + slack_delta
                    Jl, Jh, Jg_cbf, Jg_boundary = self._get_one_order_term(_tau_new, params)
                    Jg = torch.concat([Jg_cbf, Jg_boundary, self._csZv], dim=0)
                    rho1, rho2 = self._slack_weight; _n_dvneq = len(dvneq_qp)  
                    kkt_res: Tensor = torch.concat([
                        Jl + (dveq_qp.reshape(1, -1) @ Jh).reshape(-1) + (dvneq_qp.reshape(1, -1) @ Jg).reshape(-1),
                        rho1 + rho2 * _slack_new - dvneq_qp[:self._ns] - dvneq_qp[_n_dvneq-self._ns: _n_dvneq],
                    ], dim=0)
                    if kkt_res.norm(p=float('inf')) < 1e-2:
                        sqp_status = SQPStatus.CONVERGED
                        break

                # Line Search
                _alpha: float = self._line_search_alpha_init
                _line_search_eps: Tensor = 1e-6 + 1e-5 * metric_curr.clamp(max=1.0)
                for j in range(self._line_search_max_num):
                    _tau_new = self._tau + _alpha*tau_delta; _slack_new = self._slack + _alpha*slack_delta
                    # compute new metrics
                    h, g_cbf, g_boundary = self._get_zero_order_term(_tau_new, x0, params)
                    metric_new = self._metricfunc(_tau_new, _slack_new, dveq_qp, dvneq_qp, h, g_cbf, g_boundary, x0, params, self._slack_weight)
                    # compare
                    if metric_new <= metric_curr + _line_search_eps:
                        accept_alpha = True
                        break
                    _alpha = self._line_search_decay_rate * _alpha
                
                # Line Search Failure
                if not accept_alpha:
                    sqp_status = SQPStatus.LINE_SEARCH_FAILED
                    break

                # update varible
                self._tau   += _alpha * tau_delta
                self._slack += _alpha * slack_delta
                self._dveq  += _alpha * dveq_delta
                self._dvneq += _alpha * dvneq_delta

            if sqp_status is None: sqp_status = SQPStatus.MAX_ITERATIONS

        tau_star = self._tau.detach()
        slack_star = self._slack.detach()
        dveq_star = self._dveq.detach()
        dvneq_star = self._dvneq.detach()
        # solve the last qp
        Q, p, G, q, A, b = self._get_sqp_matrix(tau_star, slack_star, dveq_star, dvneq_star, x0, params)
        zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-6, verbose=0, maxIter=25)(Q, p, G, q, A, b)
        zeta_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([zeta_delta, dveq_qp, dvneq_qp])
        
        tau_delta = Tau.build_from_tensor(zeta_delta[:self._var_dims], self._nx, self._ntau); slack_delta = zeta_delta[self._var_dims:]
        _alpha = 1e-3 if sqp_status == SQPStatus.LINE_SEARCH_FAILED else 1.0
        tau_final = tau_star + _alpha*tau_delta
        slack_final = slack_star + _alpha*slack_delta

        self._tau = tau_final.detach()
        self._slack = slack_final.detach().clone()
        self._dveq = dveq_qp.detach().clone()
        self._dvneq = dvneq_qp.detach().clone()
        if self._is_first: self._is_first = False # change state
        return tau_final, sqp_status

    def _get_sqp_matrix(
            self,
            _tau: Tau,
            _slack: Tensor,
            dveq: Tensor,
            dvneq: Tensor,
            x0: Tensor,
            params: Optional[Dict[str, Dict[str, Any]]],
        ) -> Optional[Tuple[Tensor]]:
        rho1, rho2 = self._slack_weight
        h, g_cbf, g_boundary = self._get_zero_order_term(_tau, x0, params)
        Jl, Jh, Jg_cbf, Jg_boundary = self._get_one_order_term(_tau, params)
        Hl = self._get_two_order_term(_tau, dveq, dvneq, params)

        # post process
        # ensure positive matrix
        Hl = 0.5 * (Hl + Hl.T)
        eig_min = torch.linalg.eigvalsh(Hl).min()
        eps = 1e-4
        if eig_min < eps: Hl += (eps-eig_min) * self._cvI

        # merge inequation Jacobian & function
        Q = torch.concat([
            torch.concat([Hl, self._csZv.T], dim=1),
            torch.concat([self._csZv, rho2*self._csI], dim=1),
        ], dim=0)
        p = torch.concat([Jl, rho1*self._csO + rho2*_slack])
        A = torch.concat([Jh, self._csZh.T], dim=1)
        b = -h

        G = torch.concat([
            torch.concat([Jg_cbf, -self._csI], dim=1), # (ns, var_dims+ns)
            torch.concat([Jg_boundary, self._csZgb.T], dim=1), # (xxx, var_dims+ns)
            torch.concat([self._csZv, -self._csI], dim=1), # (ns, var_dims+ns)
        ], dim=0) # (2ns+xxx)
        q = torch.concat([-g_cbf+_slack, -g_boundary, _slack], dim=0)
        return Q, p, G, q, A, b

    def _get_zero_order_term(self, _tau: Tau, x0: Tensor, params: Optional[Dict[str, Dict[str, Any]]]):
        h: Tensor           = self._torch_zeros([(self._T+1)*self._nx])
        g_cbf: Tensor       = self._torch_zeros([self._T*self._nb])
        g_boundary: Tensor  = self._torch_zeros([self._T*2*self._nu])

        dynamic_params: Dict = params.get('dynamic')
        barrier_params: Dict = params.get('barrier')
        gamma: float = barrier_params.get('gamma')
        for t in range(self._T+1):
            # equation function
            _state_prev = x0 if t == 0 else self._dynamic.forward(_tau[t-1], self._tcon(dynamic_params, t-1))
            h[t*self._nx: (t+1)*self._nx] = (_state_prev - _tau.get_state(t)).flatten()
            if t < self._T:
                # inequation of CBF function
                g_cbf[t*self._nb: (t+1)*self._nb] = -self._barrier.forward(_tau.get_state(t+1), self._tcon(barrier_params, t+1)).flatten() + (1-gamma)*self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t)).flatten()
                # inequation of boundary function
                g_boundary[t*2*self._nu: (t+1)*2*self._nu] = torch.concat([self._umin-_tau.get_control(t).flatten(), _tau.get_control(t).flatten()-self._umax]) # (2*nu, )
        return h, g_cbf, g_boundary

    def _get_one_order_term(self, _tau: Tau, params: Optional[Dict[str, Dict[str, Any]]]):
        Jl: Tensor          = self._torch_zeros([self._var_dims])
        Jh: Tensor          = self._torch_zeros([(self._T+1)*self._nx, self._var_dims])
        Jg_cbf: Tensor      = self._torch_zeros([self._T*self._nb, self._var_dims])
        Jg_boundary: Tensor = self._torch_zeros([self._T*2*self._nu, self._var_dims])
                
        cost_params: Dict = params.get('cost')
        dynamic_params: Dict = params.get('dynamic')
        barrier_params: Dict = params.get('barrier')
        gamma: float = barrier_params.get('gamma')
        for t in range(self._T+1):
            # cost Jacobian
            if t < self._T: Jl[t*self._ntau: (t+1)*self._ntau] = self._cost.get_stage_cost(_tau[t], self._tcon(cost_params, t), order=1).flatten()
            if t == self._T: Jl[self._T*self._ntau: self._var_dims] = self._cost.get_terminal_cost(_tau.get_state(t), self._tcon(cost_params, self._T), order=1).flatten()
            # equation Jacobian
            Jh[t*self._nx: (t+1)*self._nx, t*self._ntau: (t+1)*self._ntau] = -self._cx.T if t < self._T else -self._cxT.T
            if t < self._T: Jh[(t+1)*self._nx: (t+2)*self._nx, t*self._ntau: (t+1)*self._ntau] = self._dynamic.forward(_tau[t], self._tcon(dynamic_params, t), order=1)
            # inequation Jacobian
            if t < self._T:
                # inequation of CBF Jacobian
                Jg_cbf[t*self._nb: (t+1)*self._nb, t*self._ntau: (t+1)*self._ntau] = (1-gamma)*self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=1) @ self._cx.T
                Jg_cbf[t*self._nb: (t+1)*self._nb, (t+1)*self._ntau: (t+2)*self._ntau] = -self._barrier.forward(_tau.get_state(t+1), self._tcon(barrier_params, t+1), order=1) @ (self._cx.T if t < self._T-1 else self._cxT.T)
                # inequation of boundary Jacobian
                Jg_boundary[t*2*self._nu: (t+1)*2*self._nu, t*self._ntau: (t+1)*self._ntau] = torch.concat([-self._cu.T, self._cu.T], dim=0)
        return Jl, Jh, Jg_cbf, Jg_boundary


    def _get_two_order_term(self, _tau: Tau, dveq: Tensor, dvneq: Tensor, params: Optional[Dict[str, Dict[str, Any]]]):
        Hl: Tensor        = self._torch_zeros([self._var_dims, self._var_dims])
        cost_params: Dict = params.get('cost')
        dynamic_params: Dict = params.get('dynamic')
        barrier_params: Dict = params.get('barrier')
        gamma: float = barrier_params.get('gamma')
        for t in range(self._T+1):
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
                self._cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=2) @ self._cx.T.unsqueeze(0)).sum(dim=0)
            else:
                _cx = self._cx if t < self._T else self._cxT
                _dvneq_t = dvneq[t*self._nb: (t+1)*self._nb] if t < self._T else 0.0
                Hl[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] += \
                ((-dvneq[(t-1)*self._nb: t*self._nb] + (1-gamma) * _dvneq_t).reshape(-1, 1, 1) * \
                _cx.unsqueeze(0) @ self._barrier.forward(_tau.get_state(t), self._tcon(barrier_params, t), order=2) @ _cx.T.unsqueeze(0)).sum(dim=0)
        return Hl

    def _get_aux_matrix(self, umin: Tensor, umax: Tensor):
        # build aux variable
        _cx: Tensor      = torch.concat([self._torch_eyes(self._nx), self._torch_zeros([self._nu, self._nx])], dim=0) # (ntau, nx)
        _cxT: Tensor     = self._torch_eyes(self._nx)
        _cu: Tensor      = torch.concat([self._torch_zeros([self._nx, self._nu]), self._torch_eyes(self._nu)], dim=0) # (ntau, nu)
        _cvI: Tensor     = self._torch_eyes(self._var_dims)
        _csZv: Tensor    = self._torch_zeros([self._ns, self._var_dims])
        _csZh: Tensor    = self._torch_zeros([self._ns, (self._T+1)*self._nx])
        _csZgb: Tensor   = self._torch_zeros([self._ns, self._T*2*self._nu])
        _csI: Tensor     = self._torch_eyes(self._ns)
        _csO: Tensor     = self._torch_ones([self._ns])
        _els_eps: Tensor = torch.concat([
            1e-3 * self._torch_ones([2, 1]),
            1e-2 * self._torch_ones([4, 1]),
        ], dim=0)
        _t_ts: Tensor = torch.arange(0, self._T+1, step=1, device=self._device, dtype=self._dtype)

        # register
        self.register_buffer('_umin', umin)
        self.register_buffer('_umax', umax)

        self.register_buffer('_cx', _cx)
        self.register_buffer('_cxT', _cxT)
        self.register_buffer('_cu', _cu)
        self.register_buffer('_cvI', _cvI)
        self.register_buffer('_csZv', _csZv)
        self.register_buffer('_csZh', _csZh)
        self.register_buffer('_csZgb', _csZgb)
        self.register_buffer('_csI', _csI)
        self.register_buffer('_csO', _csO)
        self.register_buffer('_els_eps', _els_eps)
        self.register_buffer('_t_ts', _t_ts)

        if self._start_dv_eps_detection:
            _dv_eps: Tensor  = 1e-2 * self._torch_ones([2, 1])
            self.register_buffer('_dv_eps', _dv_eps)

    def reset(self):
        self._is_first: bool = True

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