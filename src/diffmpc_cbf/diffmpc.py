from . import Cost, Dynamic
from .base import Tau, MetricFunction
from .utils import *
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, cast
from qpth.qp import QPFunction

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
        self._metricfunc = MetricFunction(nx, nu, T, self._cost, self._dynamic)

    def step(self, x0: Tensor, cost_params: Optional[Dict[str, Tensor]]) -> Tau:
        # iter for searching fix point tau_star
        with torch.no_grad():
            _tau: Tau    = Tau(self._nx, self._nu, self._T)
            _tau.set_state_init(x0.flatten())
            dveq: Tensor = self._get_init_dveq()
            for t in range(self._T):
                _tau.set_state(self._dynamic.forward(_tau[t]).flatten(), t+1)
            
            for i in range(self._iter_nums):
                alpha: float = 1.0
                Q, p, G, h, A, b = self._build_sqp_params(x0, _tau, dveq, cost_params)
                tau_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, h, A, b)
                tau_delta, dveq_qp, dvneq_qp = self._qp_var_post_process([tau_delta, dveq_qp, dvneq_qp])
                tau_delta = Tau.build_from_tensor(tau_delta, self._nx, self._ntau)
                dveq_delta = dveq_qp - dveq

                for j in range(self._line_search_max_num):
                    if self._metricfunc(_tau + alpha*tau_delta, x0.flatten(), cost_params, dveq_qp) < \
                    self._metricfunc(_tau, x0.flatten(), cost_params, dveq_qp):
                        break
                    alpha = self._line_search_decay_rate * alpha
                # update varible
                _tau += alpha * tau_delta
                dveq += alpha * dveq_delta

        tau_star = _tau.detach()
        dveq_star = dveq.detach()
        # solve the last qp
        Q, p, G, h, A, b = self._build_sqp_params(x0, tau_star, dveq_star, cost_params)
        tau_delta, _, _ = QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, h, A, b); tau_delta = cast(Tensor, tau_delta)
        tau_delta = Tau.build_from_tensor(tau_delta.flatten(), self._nx, self._ntau)
        return tau_star + tau_delta

    def _build_sqp_params(
            self,
            x0: Tensor,
            _tau: Tau,
            dveq: Tensor,
            cost_params: Optional[Dict[str, Tensor]]
        ) -> Tuple[Tensor]:
        J: Tensor  = torch.zeros([self._var_dims])
        H: Tensor  = torch.zeros([self._var_dims, self._var_dims])
        Jh: Tensor = torch.zeros([(self._T+1)*self._nx, self._var_dims])
        h: Tensor  = torch.zeros([(self._T+1)*self._nx])
        G: Tensor  = torch.zeros([2*self._nu*self._T, self._var_dims])
        g: Tensor  = torch.zeros([2*self._nu*self._T])

        # get Jacobian Vector
        for t in range(self._T+1):
            if t < self._T:
                J[t*self._ntau: (t+1)*self._ntau] = self._cost.get_stage_cost(_tau[t], cost_params, order=1).flatten()
            elif t == self._T:
                J[t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(_tau.get_state(t), cost_params, order=1).flatten()

        # get Hessian Matrix
        for t in range(self._T+1):
            if t < self._T:
                H[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] = \
                self._cost.get_stage_cost(_tau[t], cost_params, order=2) + \
                (dveq[(t+1)*self._nx: (t+2)*self._nx].reshape(-1, 1, 1) * self._dynamic.forward(_tau[t], order=2)).sum(axis=0)
            elif t == self._T:
                H[t*self._ntau: t*self._ntau+self._nx, t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(_tau.get_state(t), cost_params, order=2)
        # ensure positive matrix
        H = 0.5 * (H + H.T)
        eig_min = torch.linalg.eigvalsh(H).min()
        eps = 1e-6
        if eig_min < eps: H += (eps-eig_min) * torch.eye(self._var_dims)

        # get Jacobian Matrix for equation
        c  = torch.concat([torch.eye(self._nx), torch.zeros([self._nu, self._nx])], axis=0) # (ntau, nx)
        cT = torch.eye(self._nx)
        for t in range(self._T+1):
            Jh[t*self._nx: (t+1)*self._nx, t*self._ntau: (t+1)*self._ntau] = -c.T if t < self._T else -cT.T
            if t < self._T:
                Jh[(t+1)*self._nx: (t+2)*self._nx, t*self._ntau: (t+1)*self._ntau] = self._dynamic.forward(_tau[t], order=1)

        # get equation Vector
        for t in range(self._T+1):
            if t == 0:
                _dym_term = x0
            else:
                _dym_term = self._dynamic.forward(_tau[t-1])
            h[t*self._nx: (t+1)*self._nx] = _dym_term.flatten() - _tau.get_state(t)

        # set control boundary conditions
        for t in range(self._T):
            G[t*self._nu: (t+1)*self._nu, t*self._ntau+self._nx: (t+1)*self._ntau] = torch.eye(self._nu)
            G[(self._T+t)*self._nu: (self._T+t+1)*self._nu, t*self._ntau+self._nx: (t+1)*self._ntau] = -torch.eye(self._nu)
            g[t*self._nu: (t+1)*self._nu] = self._umax - _tau.get_control(t)
            g[(self._T+t)*self._nu: (self._T+t+1)*self._nu] = _tau.get_control(t)-self._umin
        
        return H, J, G, g, Jh, -h
        

    def _qp_var_post_process(self, vs: List[Tensor]) -> List[Tensor]:
        return [v.flatten() for v in vs]
    
    def _get_init_dveq(self) -> Tensor:
        return torch.zeros([(self._T+1)*self._nx, ])