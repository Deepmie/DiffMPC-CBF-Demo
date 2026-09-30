from ..config import SQPConfig
from .. import Tau
from typing import Tuple, Union, Optional
import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from qpth.qp import QPFunction


class SQP:
    def __init__(self, config: SQPConfig):
        self._nx, self._nu, self._ntau = config.nx, config.nu, config.ntau
        self._cost, self._dynamic = config.cost, config.dynamic

    def solve(self, x0: ndarray, cost_params: Optional[Tuple[ndarray]]=None, total_step: int=50):
        xs = np.zeros([self._nx, total_step+1])
        xt = x0.copy(); xs[:, 0] = x0.flatten()
        for i in range(total_step):
            _x, _u = self._step(xt, cost_params)
            u0 = _u[0, :]
            xt = self._dynamic.step(xt.flatten(), u0)
            xs[:, i+1] = xt.flatten()
        return xs

    def _step(self, x0: ndarray, cost_params: Optional[Tuple[ndarray]]=None):
        _tau: Tau    = Tau(self._nx, self._nu, self._T)
        _tau.set_state_init(x0.flatten())
        dveq = self._get_init_dveq()
        for t in range(self._T):
            _tau.set_state(self._dynamic.step(_tau[t]).flatten(), t+1)

        for i in range(self._iter_nums):
            alpha: float = 1.0
            var_dims: int = self._T*self._ntau+self._nx
            J: ndarray  = np.zeros([var_dims])
            H: ndarray  = np.zeros([var_dims, var_dims])
            Jh: ndarray = np.zeros([(self._T+1)*self._nx, var_dims])
            h: ndarray  = np.zeros([(self._T+1)*self._nx])
            G: ndarray  = np.zeros([2*self._nu*self._T, var_dims])
            g: ndarray  = np.zeros([2*self._nu*self._T])

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
                    (dveq[(t+1)*self._nx: (t+2)*self._nx].reshape(-1, 1, 1) * self._dynamic.step(_tau[t], order=2)).sum(axis=0)
                elif t == self._T:
                    H[t*self._ntau: t*self._ntau+self._nx, t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(_tau.get_state(t), cost_params, order=2)
            
            # ensure positive matrix
            H = 0.5 * (H + H.T)
            eig_min = np.linalg.eigvalsh(H).min()
            eps = 1e-6
            if eig_min < eps: H += (eps-eig_min) * np.eye(var_dims)

            # get Jacobian Matrix for equation
            c  = np.concatenate([np.eye(self._nx), np.zeros([self._nu, self._nx])], axis=0) # (ntau, nx)
            cT = np.eye(self._nx)
            for t in range(self._T+1):
                Jh[t*self._nx: (t+1)*self._nx, t*self._ntau: (t+1)*self._ntau] = -c.T if t < self._T else -cT.T
                if t < self._T:
                    Jh[(t+1)*self._nx: (t+2)*self._nx, t*self._ntau: (t+1)*self._ntau] = self._dynamic.step(_tau[t], order=1)

            # get equation Vector
            for t in range(self._T+1):
                if t == 0:
                    _dym_term = x0
                else:
                    _dym_term = self._dynamic.step(_tau[t-1])
                h[t*self._nx: (t+1)*self._nx] = _dym_term.flatten() - _tau.get_state(t)

            # set control boundary conditions
            for t in range(self._T):
                G[t*self._nu: (t+1)*self._nu, t*self._ntau+self._nx: (t+1)*self._ntau] = torch.eye(self._nu)
                G[(self._T+t)*self._nu: (self._T+t+1)*self._nu, t*self._ntau+self._nx: (t+1)*self._ntau] = -torch.eye(self._nu)
                g[t*self._nu: (t+1)*self._nu] = self._umax*np.ones(self._nu) - _tau.get_control(t)
                g[(self._T+t)*self._nu: (self._T+t+1)*self._nu] = _tau.get_control(t)-self._umin*np.ones(self._nu)

            # convert numpy to tensor
            J, H, Jh, h, G, g = self._convert_numpy_to_tensor_batch([J, H, Jh, h, G, g])
            tau_delta, dveq_qp, dvneq_qp = QPFunction(eps=1e-12, verbose=0, maxIter=20)(H, J, G, g, Jh, -h)
            tau_delta, dveq_qp, dvneq_qp = self._convert_tensor_to_numpy_batch([tau_delta, dveq_qp, dvneq_qp])
            tau_delta = Tau.build_from_numpy(tau_delta, self._nx, self._ntau)
            dveq_delta = dveq_qp - dveq

            for j in range(self._line_search_max_num+1):
                if self._metricfunc(_tau + alpha*tau_delta, x0.flatten(), cost_params, dveq_qp) < \
                self._metricfunc(_tau, x0.flatten(), cost_params, dveq_qp):
                    break
                alpha = self._decay * alpha

            # update varible
            _tau += alpha * tau_delta
            dveq += alpha * dveq_delta
        return _tau.x, _tau.u


    def _get_init_dveq(self) -> ndarray:
        return np.zeros([(self._T+1)*self._nx, ])