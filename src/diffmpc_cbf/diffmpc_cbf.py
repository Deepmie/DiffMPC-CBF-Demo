from . import Cost, Dynamic, Barrier, MetricFunction
from .base import Tau
from .utils import *
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, Any, cast
from qpth.qp import QPFunction

class DiffMPCBF:
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
        umin: Tensor, # (nu,)
        umax: Tensor,
        iter_nums: int=10,
        line_search_max_num: int=5,
        line_search_decay_rate: float=0.5,
    ):
        self._nx = nx; self._nu = nu; self._nb = nb; self._T = T; self._batch_size = batch_size; self._ntau = self._nx + self._nu; self._var_dims = self._T*self._ntau+self._nx
        self._cost = cost; self._dynamic = dynamic; self._barrier = barrier; self._umin = umin; self._umax = umax
        self._iter_nums = iter_nums; self._line_search_max_num = line_search_max_num; self._line_search_decay_rate = line_search_decay_rate
        self._metricfunc = MetricFunction(nx, nu, T, self._cost, self._dynamic)
        self._tau = Tau(self._nx, self._nu, self._T)

    def step(self, x0: Tensor, params: Optional[Dict[str, Tensor]]=None) -> Tau:
        if params is None: params = {}
        for key in ['cost', 'dynamic']:
            if key not in params:
                params[key] = dict()

        if x0.ndimension() == 1: x0 = x0.unsqueeze(-1)
        
        # iter for searching fix point tau_star
        with torch.no_grad():
            self._tau.set_state_init(x0.flatten())
            dveq: Tensor = self._get_init_dveq()
            for t in range(self._T):
                self._tau.set_state(self._dynamic.forward(self._tau[t], {**params.get('dynamic'), 't': t}).flatten(), t+1)
            
            for i in range(self._iter_nums):
                alpha: float = 1.0
                Q, p, G, h, A, b = self._build_sqp_params(x0, self._tau, dveq, params)
                tau_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, h, A, b)
                tau_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([tau_delta, dveq_qp, dvneq_qp])
                tau_delta = Tau.build_from_tensor(tau_delta, self._nx, self._ntau)
                dveq_delta = dveq_qp - dveq

                for j in range(self._line_search_max_num):
                    if self._metricfunc(self._tau + alpha*tau_delta, x0, params, dveq_qp) < \
                    self._metricfunc(self._tau, x0, params, dveq_qp):
                        break
                    alpha = self._line_search_decay_rate * alpha
                # update varible
                self._tau += alpha * tau_delta
                dveq += alpha * dveq_delta

        tau_star = self._tau.detach()
        dveq_star = dveq.detach()
        # solve the last qp
        Q, p, G, h, A, b = self._build_sqp_params(x0, tau_star, dveq_star, params)
        tau_delta, _, _ = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, h, A, b); tau_delta = cast(Tensor, tau_delta)
        tau_delta = Tau.build_from_tensor(tau_delta.flatten(), self._nx, self._ntau)
        return tau_star + tau_delta

    def _build_sqp_params(
            self,
            x0: Tensor,
            _tau: Tau,
            dveq: Tensor,
            params: Optional[Dict[str, Dict[str, Any]]]
        ) -> Tuple[Tensor]:
        Jl: Tensor          = torch.zeros([self._var_dims])
        Hl: Tensor          = torch.zeros([self._var_dims, self._var_dims])
        Jh: Tensor          = torch.zeros([(self._T+1)*self._nx, self._var_dims])
        h: Tensor           = torch.zeros([(self._T+1)*self._nx])
        Jg_cbf: Tensor      = torch.zeros([self._T*self._nb, self._var_dims])
        g_cbf: Tensor       = torch.zeros([self._T*self._nb])
        Jg_boundary: Tensor = torch.zeros([self._T*2*self._nu, self._var_dims])
        g_boundary: Tensor  = torch.zeros([self._T*2*self._nu])

        # cost Jacobian
        Jl = self._cost.jacobian(_tau.tau, params.get('cost')).reshape(-1)[:self._var_dims] # (_var_dim, )

        # cost Hessian
        for t in range(self._T):
            Hl[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] = \
            self._cost.get_stage_cost(_tau[t], {**params.get('cost'), 't': t}, order=2) + \
            (dveq[(t+1)*self._nx: (t+2)*self._nx].reshape(-1, 1, 1) * self._dynamic.forward(_tau[t], {**params.get('dynamic'), 't': t}, order=2)).sum(axis=0)    
        Hl[self._T*self._ntau: self._T*self._ntau+self._nx, self._T*self._ntau: self._T*self._ntau+self._nx] = self._cost.get_terminal_cost(_tau.get_state(self._T), {**params.get('cost'), 't': self._T}, order=2)
        # ensure positive matrix
        Hl = 0.5 * (Hl + Hl.T)
        eig_min = torch.linalg.eigvalsh(Hl).min()
        eps = 1e-6
        if eig_min < eps: Hl += (eps-eig_min) * torch.eye(self._var_dims)
        
        # equation Jacobian
        c: Tensor  = torch.concat([torch.eye(self._nx), torch.zeros([self._nu, self._nx])], axis=0) # (ntau, nx)
        cT: Tensor = torch.eye(self._nx)
        for t in range(self._T+1):
            Jh[t*self._nx: (t+1)*self._nx, t*self._ntau: (t+1)*self._ntau] = -c.T if t < self._T else -cT.T
            if t < self._T:
                Jh[(t+1)*self._nx: (t+2)*self._nx, t*self._ntau: (t+1)*self._ntau] = self._dynamic.forward(_tau[t], {**params.get('dynamic'), 't': t}, order=1)
        
        # equation function
        for t in range(self._T+1):
            _state_prev = x0 if t == 0 else self._dynamic.forward(_tau[t-1], {**params.get('dynamic'), 't': t-1})
            h[t*self._nx: (t+1)*self._nx] = (_state_prev - _tau.get_state(t)).flatten()

        # inequation of CBF Jacobian
        barrier_params: Dict = params.get('barrier')
        gamma: float = barrier_params.get('gamma')
        for t in range(self._T):
            Jg_cbf[t*self._nb: (t+1)*self._nb, t*self._ntau: (t+1)*self._ntau] = -(1-gamma) * self._barrier.forward(_tau.get_state(t), {**barrier_params, 't': t}, order=1) @ c.T
            Jg_cbf[t*self._nb: (t+1)*self._nb, (t+1)*self._ntau: (t+2)*self._ntau] = self._barrier.forward(_tau.get_state(t+1), {**barrier_params, 't': t+1}, order=1) @ (c.T if t < self._T-1 else cT.T)

        # inequation of CBF function
        for t in range(self._T):
            g_cbf[t*self._nb: (t+1)*self._nb] = self._barrier.forward(_tau.get_state(t), {**barrier_params, 't': t}).flatten()
        
        # set control boundary conditions
        for t in range(self._T):
            Jg_boundary[t*self._nu: (t+1)*self._nu, t*self._ntau+self._nx: (t+1)*self._ntau] = torch.eye(self._nu)
            Jg_boundary[(self._T+t)*self._nu: (self._T+t+1)*self._nu, t*self._ntau+self._nx: (t+1)*self._ntau] = -torch.eye(self._nu)
            g_boundary[t*self._nu: (t+1)*self._nu] = self._umax - _tau.get_control(t).flatten()
            g_boundary[(self._T+t)*self._nu: (self._T+t+1)*self._nu] = _tau.get_control(t).flatten() - self._umin
        
        return Hl, Jl, G, g, Jh, -h

    def _qp_var_post_process(self, vs: List[Tensor]) -> List[Tensor]:
        return [v.flatten() for v in vs]
    
    def _get_init_dveq(self) -> Tensor:
        return torch.zeros([(self._T+1)*self._nx, ])

    @property
    def cost(self):
        return self._cost

    @property
    def dynamic(self):
        return self._dynamic