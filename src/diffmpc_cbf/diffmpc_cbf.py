from . import Cost, Dynamic, Barrier
from .base import Tau, SQPStatus
from .utils import *
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
        slack_weight: List[float] = [10, 10],
        min_iter_nums: int = 2,
        max_iter_nums: int = 50,
        # qp_config: List[Tuple] = [(1e-5, 20), (1e-7, 30), (1e-8, 50)],
        qp_config: List[Tuple] = [(1e-5, 25)],
        max_reg_nums: int = 5,
        reg_init: float = 1e-6,
        reg_factor: float = 10.0,
        reg_max: float = 1e6,
        ls_alpha_init: float =1.0,
        ls_max_num: int = 10,
        ls_decay_rate: float = 0.5,
        two_order_mode: str = 'nodv', # 'full', 'nodv'
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
        self._two_order_mode = two_order_mode
        self._device = device
        self._dtype = dtype
        self._qp_config = [{'eps': qc[0], 'maxIter': qc[1]} for qc in qp_config]
        self._max_reg_nums = max_reg_nums
        self._reg_init = reg_init
        self._reg_factor = reg_factor
        self._reg_max = reg_max

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
                kkt_ok, stat_ok = self._check_nlp_kkt(dveq_qp, dvneq_qp)
                step_ok = zeta_delta.norm(float('inf')) < 5e-2
                if pf_ok & kkt_ok & step_ok:
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
                    if pf_ok & stat_ok & step_ok:
                        status = SQPStatus.CONVERGED
                        break
                    
                    status = SQPStatus.LINE_SEARCH_FAILED
                    break
                
                # accept update
                zeta_delta_apt = alpha * zeta_delta
                self._zetav += zeta_delta_apt
                self._dveqv.copy_(dveq_qp)
                self._dvneqv.copy_(dvneq_qp)

            if i == self._max_iter_nums-1 and status == SQPStatus.INIT:
                status = SQPStatus.MAX_ITERATIONS
                kkt_ok, stat_ok = self._check_nlp_kkt(dveq_qp, dvneq_qp)

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
            raise RuntimeError(f'SQP could not converged. Reason: {status.name}')
        
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
        
        _Qv = self._Qv.clone()
        _pv = self._pv.clone()
        _Gv = self._Gv.clone()
        _qv = self._qv.clone()
        _Av = self._Av.clone()
        _bv = self._bv.clone()
        # reg Q
        # self._regularize_hessian(_Qv)
        _Qv = _Qv + 0.1 * self._cQI[:self._vardim, :self._vardim]
        # normalization
        scale_g = torch.sqrt(_Gv.square().sum(1) + _qv.square()).clamp_min(1e-8).detach() # (vd,)
        scale_a = torch.sqrt(_Av.square().sum(1) + _bv.square()).clamp_min(1e-8).detach() # (vd,)
        _Gs = _Gv / scale_g.unsqueeze(1)
        _qs = _qv / scale_g
        _As = _Av / scale_a.unsqueeze(1)
        _bs = _bv / scale_a

        for qc in self._qp_config:
            eps, maxIter = qc.get('eps'), qc.get('maxIter')
            zeta_delta, dveq_qp, dvneq_qp = QPFunction(eps=eps, verbose=0, notImprovedLim=20, maxIter=maxIter)(_Qv, _pv, _Gs, _qs, _As, _bs)
            zeta_delta, dveq_qp, dvneq_qp = self._process_qp_result(zeta_delta, dveq_qp, dvneq_qp)
            dveq_qp = dveq_qp / scale_a
            dvneq_qp = dvneq_qp / scale_g

            # check
            slack_qp = _qv - _Gv @ zeta_delta # (T*ng, )
            r_eq = (_Av @ zeta_delta - _bv).norm(float('inf'))
            r_neq = (_Gv @ zeta_delta - _qv).clamp_min(0.0).norm(float('inf'))
            stat = (_Qv @ zeta_delta + _pv + _Av.T @ dveq_qp + _Gv.T @ dvneq_qp).norm(float('inf'))
            r_dv = (-dvneq_qp).clamp_min(0.0).norm(float('inf'))
            r_cp = (dvneq_qp * slack_qp).norm(float('inf'))
            qp_ok = (torch.isfinite(zeta_delta).all()) & \
                    (torch.isfinite(dveq_qp).all()) & \
                    (torch.isfinite(dvneq_qp).all()) & \
                    (r_eq < 1e-4) & \
                    (r_neq < 1e-4) & \
                    (stat < 5e-2) & \
                    (r_dv < 1e-3) & \
                    (r_cp < 1e-3)
            if qp_ok:
                break

        if not qp_ok:
            raise RuntimeError((
                'QP solve failed. Var Tables:\n'
                f'r_eq: {r_eq:.3e}\n'
                f'r_neq: {r_neq:.3e}\n'
                f'stat: {stat:.3e}\n'
                f'r_dv: {r_dv:.3e}\n'
                f'r_cp: {r_cp:.3e}\n'
            ))
        return zeta_delta, dveq_qp, dvneq_qp

    def _check_primal_feasibility(self) -> Tensor:
        r_eq = (-self._bv).norm(float('inf'))
        r_neq = (-self._qv).clamp_min(0.0).norm(float('inf'))
        pf_ok = (r_eq < 1e-3) & \
                (r_neq < 1e-3)
        return pf_ok

    def _check_nlp_kkt(self, dveqv: Tensor, dvneqv: Tensor) -> Tuple[Tensor, Tensor]:
        stat = (self._pv + self._Av.T @ dveqv + self._Gv.T @ dvneqv).norm(float('inf'))
        r_dv = (-dvneqv).clamp_min(0.0).norm(float('inf'))
        r_cp = (dvneqv * self._qv).norm(float('inf'))

        stat_ok = stat < 5e-2
        kkt_ok  = stat_ok & \
                  (r_dv < 1e-3) & \
                  (r_cp < 1e-3)
        return kkt_ok, stat_ok

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
        else:
            raise NameError('mode must include two options: [main, temp.]')
        
        metric: float = 0.0
        ctpm = params.get('cost')
        rho1, rho2 = self._slack_weight
        weq, wneq = self._metric_weight

        # cost
        metric += self._cost.get_cost(zeta[:, :self._ntau], self._tcon(ctpm, 'full')).sum(0) # (1, )
        slack = zeta[:, self._ntau:]
        metric += rho1 * slack.sum() + rho2/2 * slack.square().sum()

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
            self._zeta[t+1, :self._nx].copy_(self._dynamic.forward(self._zeta[t, :self._ntau].unsqueeze(0), self._tcon(dypm, t)).squeeze(0))
            g_cbf = self._barrier.forward(self._zeta[t+1, :self._nx].unsqueeze(0), self._tcon(brpm, t+1)) - \
            (1-gamma) * self._barrier.forward(self._zeta[t, :self._nx].unsqueeze(0), self._tcon(brpm, t))
            self._zeta[t, self._ntau:] = (-g_cbf.squeeze(0)).clamp_min(0.0) + 1e-3
        
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

        u = _zeta[self._tdx, self._nx: self._ntau] # (T, nu)
        _slack = _zeta[self._tdx, self._ntau:] # (T, nb)

        # get b
        _b[0] = _zeta[0, :self._nx] - x0.flatten()
        _b[self._tdx+1] = _zeta[self._tdx+1, :self._nx] - self._dynamic.forward(_zeta[self._tdx, :self._ntau], self._tcon(dypm))
        
        # get q
        _q[:, :self._nb] = self._barrier.forward(_zeta[self._tdx+1, :self._nx], self._tcon(brpm, 'next')) \
                           - (1-gamma) * self._barrier.forward(_zeta[self._tdx, :self._nx], self._tcon(brpm)) + _slack # (T, nb)
        _q[:, self._nb: self._nd] = torch.concat([
            u - self._umin,
            self._umax - u
        ], dim=-1) # (T, 2*nu)
        _q[:, self._nd:] = _slack

    def _update_one_order_term(self, _zeta: Tensor, params: Optional[Dict], mode: str='main'):
        '''
        mode: 'main', 'temp'
        '''
        ctpm: Dict = params.get('cost')
        dypm: Dict = params.get('dynamic')
        brpm: Dict = params.get('barrier')
        gamma: float = brpm.get('gamma')
        rho1, rho2 = self._slack_weight

        if mode == 'main':
            _p, _A, _G = self._p, self._A, self._G
            _first_one = self._first_one
        elif mode == 'temp':
            _p, _A, _G = self._ptmp, self._Atmp, self._Gtmp
            _first_one = self._first_one_tmp

        _p[:, :self._ntau] = self._cost.get_cost(_zeta[:, :self._ntau], self._tcon(ctpm, 'full'), order=1).squeeze(1)
        _p[self._T, self._nx: self._ntau] = 0.0
        _p[self._tdx, self._ntau:] = rho1 * self._csO + rho2 * _zeta[self._tdx, self._ntau:]

        # equation jacobian
        _A[self._tdx+1, :, self._tdx, :self._ntau] = self._dynamic.forward(_zeta[self._tdx, :self._ntau], self._tcon(dypm), order=1)
        _A[self._tf, :, self._tf, :self._ntau] = -self._cx.T
        
        # inequation jacobian
        _G[self._tdx, :self._nb, self._tdx, :self._ntau] = (1-gamma) * self._barrier.forward(_zeta[self._tdx, :self._nx], self._tcon(brpm), order=1) @ self._cx.T
        _G[self._tdx, :self._nb, self._tdx+1, :self._ntau] = -self._barrier.forward(_zeta[self._tdx+1, :self._nx], self._tcon(brpm, 'next'), order=1) @ self._cx.T
        _G[self._tdx, self._nb: self._nd, self._tdx, :self._ntau] = torch.concat([-self._cu.T, self._cu.T], dim=0)

        if _first_one:
            _G[:, :self._nb, :self._T, self._ntau:] = -self._csI
            _G[:, self._nd:, :self._T, self._ntau:] = -self._csI
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
        rho2 = self._slack_weight[1]

        if mode == 'main':
            _Q = self._Q
            _first_two = self._first_two
        elif mode == 'temp':
            _Q = self._Qtmp
            _first_two = self._first_two_tmp

        _Q[self._tf, :self._ntau, self._tf, :self._ntau] = self._cost.get_cost(_zeta[self._tf, :self._ntau], self._tcon(ctpm, 'full'), order=2).squeeze(1)

        if self._two_order_mode == 'full':
            # equation dual varible
            _Q[self._tdx, :self._ntau, self._tdx, :self._ntau] += \
            (dveq[self._tdx+1].unsqueeze(-1).unsqueeze(-1) * self._dynamic.forward(_zeta[self._tdx, :self._ntau], self._tcon(dypm), order=2)).sum(dim=1)
            # inequation dual varible
            coeff = torch.concat([
                (1-gamma) * dvneq[0, :self._nb].unsqueeze(0), # (1, nb)
                -dvneq[:self._T-1, :self._nb] + (1-gamma) * dvneq[1:self._T, :self._nb], # (T-1, nb)
                -dvneq[self._T-1, :self._nb].unsqueeze(0), # (1, nb)
            ], dim=0).unsqueeze(-1).unsqueeze(-1) # (T+1, nb, 1, 1)
            _Q[self._tf, :self._ntau, self._tf, :self._ntau] += \
            (coeff * (self._cx @ self._barrier.forward(_zeta[:, :self._nx], self._tcon(brpm, 'full'), order=2) @ self._cx.T)).sum(dim=1)

        _Q[self._T, self._nx: self._ntau, self._T, :] = 0.0
        _Q[self._T, :, self._T, self._nx: self._ntau] = 0.0
        _Q[self._T, self._nx: self._ntau, self._T, self._nx: self._ntau] = 1e-3 * self._cuI

        # only once
        if _first_two:
            _Q[:self._T, self._ntau:, :self._T, self._ntau:].copy_(rho2 * self._csI)
            if mode == 'main':
                self._first_two = False
            elif mode == 'temp':
                self._first_two_tmp = False

    def _regularize_hessian(self, H: Tensor) -> float:
        _H = H.clone()
        _H = 0.5 * (_H + _H.T)
        if not torch.isfinite(_H).all():
            raise RuntimeError('Non-finite Hessian detected.')

        reg: float = 0.0
        for _ in range(self._max_reg_nums):
            Q = _H + reg * self._cQI[:self._vardim, :self._vardim]
            _, info = torch.linalg.cholesky_ex(Q, check_errors=False)
            if (info == 0).all():
                break
            reg = self._reg_init if reg == 0.0 else reg * self._reg_factor
            if reg > self._reg_max:
                raise RuntimeError(f'No right reg find. now reg_max={self._reg_max:e}')
        H.copy_(Q)
        return reg

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

    def _tcon(self, params: Dict[str, Any], t: Optional[Union[int, str]]=None) -> Dict[str, Any]:
        '''
        t options: 'curr', 'next', 'full', any int and None
        '''
        if t is None or t == 'curr':
            _t = self._tdx
        elif t == 'next':
            _t = self._tdx + 1
        elif t == 'full':
            _t = self._tf
        elif isinstance(t, int):
            _t = t * self._t
        else:
            raise ValueError('param of t not in option list.')
        return {**params, **{'t': _t}}

    def tcon(self, params: Dict[str, Any], t: Optional[Union[int, str]]=None) -> Dict[str, Any]:
        return self._tcon(params, t)

    def _buffer_matrix(self, umin: Tensor, umax: Tensor):
        _vdim = (self._T+1) * self._nv
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
        self._cQI: Tensor
        self._tdx: Tensor
        self._tf: Tensor
        self._t: Tensor

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
        self.register_buffer('_cQI', self._eye(_vdim), persistent=False)
        self.register_buffer('_tdx', self._arange(0, self._T), persistent=False)
        self.register_buffer('_tf', self._arange(0, self._T+1), persistent=False)
        self.register_buffer('_t', torch.ones(1, device=self._device, dtype=torch.int), persistent=False)

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
    def _arange(self, start: int, end: int): return torch.arange(start, end, device=self._device, dtype=torch.int)
    def _tensor(self, v): return torch.as_tensor(v, device=self._device, dtype=self._dtype)