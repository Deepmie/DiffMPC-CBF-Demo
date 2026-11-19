from .. import Cost, Dynamic, Barrier, MetricFunction
from ..base import Tau, SQPStatus
from ..utils import *
import torch
import torch.nn.functional as F
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
        metric_weight: List[float] = [10.0, 5.0],
        slack_weight: List[float] = [1e2, 1e3],
        min_iter_nums: int = 2,
        max_iter_nums: int = 30,
        qp_config: List[Tuple] = [(1e-5, 20), (1e-7, 30), (1e-8, 50)],
        ls_alpha_init: float =1.0,
        ls_max_num: int = 10,
        ls_decay_rate: float = 0.5,
        use_gauss_newton: bool = True,
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
        self._metric_weight = metric_weight
        self._slack_weight = slack_weight
        self._min_iter_nums = min_iter_nums
        self._max_iter_nums = max_iter_nums
        self._ls_alpha_init = ls_alpha_init
        self._ls_max_num = ls_max_num
        self._ls_decay_rate = ls_decay_rate
        self._use_gauss_newton = use_gauss_newton
        self._device = device
        self._dtype = dtype
        self._qp_config = [{'eps': qc[0], 'maxIter': qc[1]} for qc in qp_config]

        self._ntau = nx + nu
        self._ng = 2 * (nb + nu)
        self._nd = nb + 2 * nu
        self._nv = self._ntau + nb
        self._taudim = (T + 1) * self._ntau
        self._slackdim = T * nb
        self._vardim = self._taudim + self._slackdim
        
        self._buffer_matrix(umin, umax)
        self.reset()

    def forward(self, x0: Tensor, params: Optional[Dict]=None) -> Tuple[Tensor, SQPStatus]:
        if not (x0.device == self._device and x0.dtype == self._dtype): raise ValueError(f'device and dtype of x is {x0.device} and {x0.dtype} respectively, no match DiffMPC-CBFs device {self._device} and dtype {self._dtype}.')
        if x0.ndimension() == 1: x0 = x0.unsqueeze(-1)
        params = self._preprocess_params(params)
        status: SQPStatus = SQPStatus.INIT
        zeta_delta_apt = None

        with torch.no_grad():
            self._reconstruct(x0, params)

            for i in range(self._max_iter_nums):
                zeta_delta, dveq_qp, dvneq_qp = self._solve_qp_subproblem(self._zeta, x0, params)
                zeta_deltaw = F.pad(zeta_delta, (0, self._nb), mode="constant", value=0).view(self._T+1, self._nv)
                
                # is convergence
                pf_ok = self._check_primal_feasibility()
                kkt_ok = self._check_nlp_kkt(dveq_qp, dvneq_qp)
                step_ok = (zeta_delta_apt is None) or (zeta_delta_apt.norm(float('inf')) < 1e-3)
                if pf_ok and kkt_ok and step_ok:
                    status = SQPStatus.CONVERGED
                    break
                
                # direction maybe error
                # model = 0.5 * (zeta_delta @ self._Qv @ zeta_delta) + self._pv @ zeta_delta
                metric_dir = self._metricfunc_direct(zeta_delta, dveq_qp, dvneq_qp)
                if metric_dir >= 0.0:
                    if not pf_ok:
                        status = SQPStatus.NEED_RESTORE
                        break
                    status = SQPStatus.NON_DESCENT
                    break
                
                accept, alpha = self._line_search(zeta_deltaw, dveq_qp, dvneq_qp, x0, params)
                if not accept:
                    status = SQPStatus.LINE_SEARCH_FAILED
                    break
                
                # accept update
                zeta_delta_apt = alpha * zeta_delta
                self._zetav += zeta_delta_apt
                self._dveqv.copy_(dveq_qp)
                self._dvneqv.copy_(dvneq_qp)

            if i == self._max_iter_nums-1 and status == SQPStatus.INIT:
                status = SQPStatus.MAX_ITERATIONS
                self._check_nlp_kkt()

        if status == SQPStatus.CONVERGED:
            zeta_star = self._zeta.detach().clone(); zeta_starv = zeta_star.view(-1)[:self._vardim]
            zeta_delta, dveq_qp, dvneq_qp = self._solve_qp_subproblem(zeta_star, x0, params)

            # check
            if zeta_delta.norm(float('inf')) < 1e-3:
                zeta_final = zeta_starv + zeta_delta
            else:
                zeta_final = None
                status = SQPStatus.FAILED
        else:
            zeta_final = None
            self._check_nlp_kkt(dveq_qp, dvneq_qp)
            raise RuntimeError('SQP could not converged.')
        
        self._Q = self._Q.detach().clone()
        self._p = self._p.detach().clone()
        self._A = self._A.detach().clone()
        self._b = self._b.detach().clone()
        self._G = self._G.detach().clone()
        self._q = self._q.detach().clone()
        self._get2Dview()
        self._first_forward = False
        return zeta_final, status
        
    def reset(self):
        self._first_forward: bool = True
        self._first_one: bool = True
        self._first_two: bool = True
        self._first_one_tmp: bool = True
        self._first_two_tmp: bool = True
        self._zeta.zero_()
        self._dvneq.zero_()
        self._dveq.zero_()

    def _solve_qp_subproblem(self, zeta: Tensor, x0: Tensor, params: Optional[Dict]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        self._update_zero_order_term(zeta, x0, params)
        self._update_one_order_term(zeta, params)
        self._update_two_order_term(zeta, self._dveq, self._dvneq, params)
        
        for qc in self._qp_config:
            eps, maxIter = qc.get('eps'), qc.get('maxIter')
            zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=eps, verbose=0, maxIter=maxIter)(self._Qv.clone(), self._pv.clone(), self._Gv.clone(), self._qv.clone(), self._Av.clone(), self._bv.clone())
            zeta_delta, dveq_qp, dvneq_qp = self._process_qp_result(zeta_delta, dveq_qp, dvneq_qp)
            slack_qp = self._qv - self._Gv @ zeta_delta # (T*ng, )
            r_eq = (self._Av @ zeta_delta - self._bv).norm(float('inf'))
            r_neq = (self._Gv @ zeta_delta - self._qv).clamp_min(0.0).norm(float('inf'))
            stat = (self._Qv @ zeta_delta + self._pv + self._Av.T @ dveq_qp + self._Gv.T @ dvneq_qp).norm(float('inf'))
            r_dv = (-dvneq_qp).clamp_min(0.0).norm(float('inf'))
            r_cp = (dvneq_qp * slack_qp).norm(float('inf'))
            
            qp_ok = torch.isfinite(zeta_delta).all() & \
            torch.isfinite(dveq_qp).all() & \
            torch.isfinite(dvneq_qp).all() & \
            (r_eq < 1e-4) & \
            (r_neq < 1e-4) & \
            (stat < 1e-4) & \
            (r_dv < 1e-4) & \
            (r_cp < 1e-4)

            if qp_ok:
                break
        
        if not qp_ok:
            raise RuntimeError('QP solve not true.')
        return zeta_delta, dveq_qp, dvneq_qp

    def _check_primal_feasibility(self) -> Tensor:
        r_eq = (-self._bv).norm(float('inf'))
        r_neq = (-self._qv).clamp_min(0.0).norm(float('inf'))
        pf_ok = (r_eq < 1e-4) & (r_neq < 1e-4)
        return pf_ok

    def _check_nlp_kkt(self, dveqv: Tensor, dvneqv: Tensor) -> Tensor:
        stat = (self._pv + self._Av.T @ dveqv + self._Gv.T @ dvneqv).norm(float('inf'))
        r_dv = (-dvneqv).clamp_min(0.0).norm(float('inf'))
        r_cp = (dvneqv * self._qv).norm(float('inf'))
        kkt_ok = (stat < 1e-3) & (r_dv < 1e-4) & (r_cp < 1e-4)
        return kkt_ok

    def _line_search(self, zeta_delta: Tensor, dveq: Tensor, dvneq: Tensor, x0: Tensor, params: Optional[Dict]) -> Tuple[bool, float]:
        accept: bool = False
        _alpha: float = self._ls_alpha_init
        _metric_curr = self._metricfunc(self._zeta, dveq, dvneq, x0, params)
        for j in range(self._ls_max_num):
            _zeta_tmp = self._zeta + _alpha * zeta_delta
            _metric_tmp = self._metricfunc(_zeta_tmp, dveq, dvneq, x0, params, mode='temp')
            if _metric_tmp - _metric_curr <= 1e-6:
                accept = True
                break
            _alpha *= self._ls_decay_rate
        return accept, _alpha

    def _metricfunc(self, zeta: Tensor, dveq: Tensor, dvneq: Tensor, x0: Tensor, params: Optional[Dict], mode: str='main') -> Tensor:
        if mode == 'main':
            _bv, _qv = self._bv, self._qv
        elif mode == 'temp':
            self._update_zero_order_term(zeta, x0, params, 'temp')
            _bv, _qv = self._btmpv, self._qtmpv
        
        metric: float = 0.0
        ctpm = params.get('cost')

        # cost
        for t in range(self._T+1):
            _taut = zeta[t, :self._ntau]
            if t < self._T:
                metric += self._cost.get_stage_cost(_taut, self._tcon(ctpm, t))
            elif t == self._T:
                metric += self._cost.get_terminal_cost(_taut, self._tcon(ctpm, t))
        rho1, rho2 = self._slack_weight
        slack = zeta[:, self._ntau:]
        metric += rho1 * slack.sum() + rho2/2 * slack.square().sum()

        weq, wneq = self._metric_weight
        # equation
        metric += (1.1 * dveq.norm(float('inf'))).clamp_min(weq) * (-_bv).norm(1)
        metric += (1.1 * dvneq.norm(float('inf'))).clamp_min(wneq) * (-_qv).clamp_min(0.0).norm(1)
        return metric
    
    def _metricfunc_direct(self, zeta_delta: Tensor, dveq: Tensor, dvneq: Tensor):
        weq, wneq = self._metric_weight
        _weq = (1.1*dveq.norm(float('inf'))).clamp_min(weq)
        _wneq = (1.1*dvneq.norm(float('inf'))).clamp_min(wneq)
        metric_dir: Tensor = self._pv @ zeta_delta + \
        _weq * 2 * ((-self._bv >= 0).to(self._dtype) - 0.5) @ (self._Av @ zeta_delta) + \
        _wneq * (-self._qv > 0).to(self._dtype) @ (self._Gv @ zeta_delta)
        return metric_dir
    
    def _reconstruct(self, x0: Tensor, params: Optional[Dict]):
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        self._zeta[0, :self._nx] = x0.flatten()
        self._zeta[:self._T-1, self._nx: self._ntau] = self._zeta[1:self._T, self._nx: self._ntau].clamp(self._umin, self._umax)
        # rollout
        for t in range(self._T):
            self._zeta[t+1, :self._nx] = self._dynamic.forward(self._zeta[t, :self._ntau].unsqueeze(-1), self._tcon(dypm, t)).flatten()
            g_cbf = self._barrier.forward(self._zeta[t+1, :self._nx], self._tcon(brpm, t+1)).flatten() - \
            (1-gamma) * self._barrier.forward(self._zeta[t, :self._nx], self._tcon(brpm, t)).flatten()
            self._zeta[t, self._ntau:] = (-g_cbf).clamp_min(0.0) + 1e-3
        
        if not self._first_forward:
            self._dveq.zero_()
            self._dvneq.zero_()

    def _update_zero_order_term(self, _zeta: Tensor, x0: Tensor, params: Optional[Dict], mode: str='main'):
        '''
        mode: 'main', 'temp'
        '''
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        if mode == 'main':
            _b, _q = self._b, self._q
        elif mode == 'temp':
            _b, _q = self._btmp, self._qtmp
        
        for t in range(self._T+1):
            _xt, _ut = _zeta[t, :self._nx], _zeta[t, self._nx: self._ntau] 
            # equation function
            _dymt = x0 if t == 0 else self._dynamic.forward(_zeta[t-1, :self._ntau], self._tcon(dypm, t-1)) # (nx, 1)
            _b[t] = _xt - _dymt.flatten()
            if t < self._T:
                _xt_next = _zeta[t+1, :self._nx]
                # inequation of CBF function
                _q[t, :self._nb] = self._barrier.forward(_xt_next, self._tcon(brpm, t+1)).flatten() - \
                (1-gamma) * self._barrier.forward(_xt, self._tcon(brpm, t)).flatten()
                # inequation of boundary function
                _q[t, self._nb: self._nd] = torch.concat([
                    _ut - self._umin,
                    self._umax - _ut,
                ]) # (2*nu, )
        _slack = _zeta[:self._T, self._ntau:]
        _q[:, :self._nb] += _slack
        _q[:, self._nd:] = _slack

    def _update_one_order_term(self, _zeta: Tensor, params: Optional[Dict], mode: str='main'):
        '''
        mode: 'main', 'temp'
        '''
        ctpm: Dict = params.get('cost')
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        if mode == 'main':
            _p, _A, _G = self._p, self._A, self._G
            _first_one = self._first_one
        elif mode == 'temp':
            _p, _A, _G = self._ptmp, self._Atmp, self._Gtmp
            _first_one = self._first_one_tmp
        
        for t in range(self._T+1):
            _taut, _xt = _zeta[t, :self._ntau], _zeta[t, :self._nx]
            if t < self._T:
                _p[t, :self._ntau] = self._cost.get_stage_cost(_taut, self._tcon(ctpm, t), order=1).flatten()
                _A[t+1, :, t, :self._ntau] = self._dynamic.forward(_taut, self._tcon(dypm, t), order=1)
                # inequation of CBF Jacobian
                _G[t, :self._nb, t, :self._ntau] = (1-gamma)*self._barrier.forward(_xt, self._tcon(brpm, t), order=1) @ self._cx.T
                _G[t, :self._nb, t+1, :self._ntau] = -self._barrier.forward(_zeta[t+1, :self._nx], self._tcon(brpm, t+1), order=1) @ self._cx.T
                # inequation of boundary Jacobian
                _G[t, self._nb: self._nd, t, :self._ntau] = torch.concat([-self._cu.T, self._cu.T], dim=0)
            if t == self._T:
                _p[t, :self._nx] = self._cost.get_terminal_cost(_xt, self._tcon(ctpm, t), order=1).flatten()
            _A[t, :, t, :self._ntau] = -self._cx.T
        rho1, rho2 = self._slack_weight
        _p[:self._T, self._ntau:].copy_(rho1 * self._csO + rho2 * _zeta[:self._T, self._ntau:])
        if _first_one:
            _G[:, :self._nb, :self._T, self._ntau:].copy_(-self._csI)
            _G[:, self._nd:, :self._T, self._ntau:].copy_(-self._csI)
            if mode == 'main':
                self._first_one = False
            elif mode == 'temp':
                self._first_one_tmp = False

    def _update_two_order_term(self, _zeta: Tensor, dveq: Tensor, dvneq: Tensor, params: Optional[Dict], mode: str='main'):
        '''
        mode: 'main', 'temp'
        '''
        ctpm: Dict = params.get('cost')
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        if mode == 'main':
            _Q = self._Q
            _first_two = self._first_two
        elif mode == 'temp':
            _Q = self._Qtmp
            _first_two = self._first_two_tmp
        
        for t in range(self._T+1):
            _taut, _xt = _zeta[t, :self._ntau], _zeta[t, :self._nx]
            if t < self._T:
                _Q[t, :self._ntau, t, :self._ntau] = self._cost.get_stage_cost(_taut, self._tcon(ctpm, t), order=2)
                if not self._use_gauss_newton:
                    _Q[t, :self._ntau, t, :self._ntau] += (dveq[t+1].reshape(-1, 1, 1) * self._dynamic.forward(_taut, self._tcon(dypm, t), order=2)).sum(dim=0)
            
            if t == self._T:
                _Q[t, :self._nx, t, :self._nx] = self._cost.get_terminal_cost(_xt, self._tcon(ctpm, t), order=2)
                _Q[t, self._nx: self._ntau, t, self._nx: self._ntau] = 1e-3 * self._cuI
            
            if not self._use_gauss_newton:
                if t == 0:
                    _dvnt = 0.0
                else:
                    _dvnt = -dvneq[t-1, :self._nb]
                _Q[t, :self._ntau, t, :self._ntau] += ((_dvnt + (1-gamma) * dvneq[t, :self._nb]).reshape(-1, 1, 1) * self._cx.unsqueeze(0) @ self._barrier.forward(_xt, self._tcon(brpm, t), order=2) @ self._cx.T.unsqueeze(0)).sum(dim=0)
        # only once
        if _first_two:
            rho2 = self._slack_weight[1]
            _Q[:self._T, self._ntau:, :self._T, self._ntau:].copy_(rho2 * self._csI)
            if mode == 'main':
                self._first_two = False
            elif mode == 'temp':
                self._first_two_tmp = False

    def _get2Dview(self):
        _vdim = (self._T+1) * self._nv
        _xdim = (self._T+1) * self._nx
        _gdim = self._T * self._ng
        self._zetav = self._zeta.view(_vdim)
        self._Qv = self._Q.view(_vdim, _vdim)
        self._pv = self._p.view(_vdim)
        self._Av = self._A.view(_xdim, _vdim)
        self._bv = self._b.view(_xdim)
        self._Gv = self._G.view(_gdim, _vdim)
        self._qv = self._q.view(_gdim)
        self._dveqv = self._dveq.view(_xdim)
        self._dvneqv = self._dvneq.view(_gdim)
        # tmp
        self._Qtmpv = self._Qtmp.view(_vdim, _vdim)
        self._ptmpv = self._ptmp.view(_vdim)
        self._Atmpv = self._Atmp.view(_xdim, _vdim)
        self._btmpv = self._btmp.view(_xdim)
        self._Gtmpv = self._Gtmp.view(_gdim, _vdim)
        self._qtmpv = self._qtmp.view(_gdim)

        # getitem
        self._zetav = self._zetav[:self._vardim]
        self._Qv = self._Qv[:self._vardim, :self._vardim]
        self._pv = self._pv[:self._vardim]
        self._Av = self._Av[:, :self._vardim]
        self._Gv = self._Gv[:, :self._vardim]
        # tmp
        self._Qtmpv = self._Qtmpv[:self._vardim, :self._vardim]
        self._ptmpv = self._ptmpv[:self._vardim]
        self._Atmpv = self._Atmpv[:, :self._vardim]
        self._Gtmpv = self._Gtmpv[:, :self._vardim]

    def _preprocess_params(self, params: Optional[Dict]=None) -> Dict:
        if params is None: params = dict()
        for key in ['cost', 'dynamic', 'barrier']:
            if key not in params: params[key] = dict()
        return params

    def _process_qp_result(self, *vs: Tensor) -> List[Tensor]:
        return [v.flatten() for v in vs]

    def _tcon(self, params: Dict[str, Any], t: int) -> Dict[str, Any]:
        return {**params, **{'t': t, 't_ts': self._ts[t]}}

    def _buffer_matrix(self, umin: Tensor, umax: Tensor):
        self._zeta: Tensor
        self._dveq: Tensor
        self._dvneq: Tensor
        self._umin: Tensor
        self._umax: Tensor
        self._Q: Tensor
        self._p: Tensor
        self._A: Tensor
        self._b: Tensor
        self._G: Tensor
        self._q: Tensor
        self._cx: Tensor
        self._cu: Tensor
        self._csO: Tensor
        self._csI: Tensor
        self._cuI: Tensor
        self._ts: Tensor

        # register
        self.register_buffer('_zeta', self._zeros(self._T+1, self._nv), persistent=False)
        self.register_buffer('_dveq', self._zeros(self._T+1, self._nx), persistent=False)
        self.register_buffer('_dvneq', self._zeros(self._T, self._ng), persistent=False)
        self.register_buffer('_umin', umin, persistent=False)
        self.register_buffer('_umax', umax, persistent=False)
        
        self.register_buffer('_Q', self._zeros(self._T+1, self._nv, self._T+1, self._nv), persistent=False)
        self.register_buffer('_p', self._zeros(self._T+1, self._nv), persistent=False)
        self.register_buffer('_A', self._zeros(self._T+1, self._nx, self._T+1, self._nv), persistent=False)
        self.register_buffer('_b', self._zeros(self._T+1, self._nx), persistent=False)
        self.register_buffer('_G', self._zeros(self._T, self._ng, self._T+1, self._nv), persistent=False)
        self.register_buffer('_q', self._zeros(self._T, self._ng), persistent=False)

        self.register_buffer('_cx', self._eye(self._ntau, self._nx), persistent=False)
        self.register_buffer(
            '_cu',
            torch.concat([self._zeros(self._nx, self._nu), self._eye(self._nu)], dim=0),
            persistent=False,
        )
        self.register_buffer('_csO', self._ones(self._T, self._nb), persistent=False)
        self.register_buffer('_csI', self._eye(self._slackdim).view(self._T, self._nb, self._T, self._nb), persistent=False)
        self.register_buffer('_cuI', self._eye(self._nu), persistent=False)
        self.register_buffer('_ts', self._arange(0, self._T+1), persistent=False)

        # tmp matrix
        self._Qtmp: Tensor
        self._ptmp: Tensor
        self._Atmp: Tensor
        self._btmp: Tensor
        self._Gtmp: Tensor
        self._qtmp: Tensor
        
        self.register_buffer('_Qtmp', self._zeros(self._T+1, self._nv, self._T+1, self._nv), persistent=False)
        self.register_buffer('_ptmp', self._zeros(self._T+1, self._nv), persistent=False)
        self.register_buffer('_Atmp', self._zeros(self._T+1, self._nx, self._T+1, self._nv), persistent=False)
        self.register_buffer('_btmp', self._zeros(self._T+1, self._nx), persistent=False)
        self.register_buffer('_Gtmp', self._zeros(self._T, self._ng, self._T+1, self._nv), persistent=False)
        self.register_buffer('_qtmp', self._zeros(self._T, self._ng), persistent=False)
        self._get2Dview()

    def to(self, device: Optional[torch.device]=None, dtype: Optional[torch.dtype]=None):
        if device is not None: self._device = device
        if dtype is not None: self._dtype = dtype
        res = super().to(device, dtype)
        self._get2Dview()
        return res
    
    def _zeros(self, *size): return torch.zeros(size, device=self._device, dtype=self._dtype)
    def _ones(self, *size): return torch.ones(size, device=self._device, dtype=self._dtype)
    def _eye(self, n: int, m: Optional[int]=None):
        if m is not None: return torch.eye(n, m, device=self._device, dtype=self._dtype)
        else: return torch.eye(n, device=self._device, dtype=self._dtype)
    def _arange(self, start: int, end: int): return torch.arange(start, end, device=self._device, dtype=self._dtype)