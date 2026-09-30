import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from typing import List, Optional, Union, Callable, Any
import matplotlib.pyplot as plt
from qpth.qp import QPFunction
from .type import Tau

class Cost:
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T

    def get_stage_cost(self, x_taut: ndarray, ut: Optional[ndarray]=None, xref: Optional[ndarray]=None, order: int=0) -> Union[ndarray, float]:
        if ut is None:
            pt, vt, ut = x_taut[0], x_taut[1], x_taut[2]
        else:
            pt, vt, ut = x_taut[0], x_taut[1], ut[0]
        pref, vref = xref[0], xref[1]
        if order == 0: # raw function
            return 1/2*(pt-pref)**2 + 1/2*(vt-vref)**2 + 0.05*ut**2 + 0.02*(pt-pref)**4
        elif order == 1: # (3, 1)
            return np.array([[(pt-pref)+0.08*(pt-pref)**3], [vt-vref], [0.1*ut]])
        elif order == 2: # (3, 3)
            return np.array([
                [1+0.24*(pt-pref)**2, 0, 0],
                [0                  , 1, 0],
                [0                  , 0, 0.1],
            ])

    def get_terminal_cost(self, xt: ndarray, xref: Optional[ndarray], order: int=0) -> Union[ndarray, float]:
        pt, vt = xt[0], xt[1]
        pref, vref = xref[0], xref[1]
        if order == 0:
            return 5*(pt-pref)**2 + 5*(vt-vref)**2 + 0.5*(pt-pref)**4
        elif order == 1: # (2, 1)
            return np.array([[10*(pt-pref) + 2*(pt-pref)**3], [10*(vt-vref)]])
        elif order == 2: # (2, 2)
            return np.array([
                [10 + 6*(pt-pref)**2, 0],
                [0                  , 10]
            ])
        

class Dynamic:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t

    def step(self, x_taut: ndarray, ut: Optional[ndarray]=None, order: int=0) -> ndarray:
        if ut is None:
            pt, vt, ut = x_taut[0], x_taut[1], x_taut[2]
        else:
            pt, vt, ut = x_taut[0], x_taut[1], ut[0]

        if order == 0: # (2, 1)
            return np.array([[pt+self.delta_t*vt], [vt+self.delta_t*(ut-0.1*vt**3)]])
        elif order == 1:
            return np.array([ # (2, 3)
                [1.0, self.delta_t, 0.0],
                [0.0, 1-0.3*self.delta_t*vt**2, self.delta_t]
            ])
        elif order == 2:
            return np.array([ # (2, 3, 3)
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, -0.6*self.delta_t*vt, 0],
                 [0, 0, 0]]
            ])


class MetricFunction:
    def __init__(self, nx: int, nu: int, T: int, cost: Cost, dynamic: Dynamic):
        self._nx = nx; self._nu = nu; self._T = T; self._cost = cost; self._dynamic = dynamic
        self._ntau = self._nx + self._nu
        self.eq_weight: float = 10.0

    def __call__(self, tau: Tau, x0: ndarray, xref: Optional[ndarray]=None, dvep: Optional[ndarray]=None) -> float: # (T*ntau+nx)
        metric: float = 0.0

        # cost
        for t in range(self._T):
            metric += self._cost.get_stage_cost(tau[t], xref=xref)
        metric += self._cost.get_terminal_cost(tau.get_state(self._T), xref=xref)

        # equality residual
        h = np.zeros([(self._T+1)*self._nx])
        for t in range(self._T+1):
            if t == 0:
                _dym_term = x0
            else:
                _dym_term = self._dynamic.step(tau[t-1])
            h[t*self._nx: (t+1)*self._nx] = _dym_term.flatten() - tau.get_state(t)

        _eq_weight = self.eq_weight if dvep is None else max(self.eq_weight, 1.1*np.max(np.abs(dvep)))
        metric += _eq_weight * np.linalg.norm(h, ord=1)
        
        return metric


class SQP:
    def __init__(self, verbose: int = 1):
        self._verbose = verbose
        self._iter_nums: int = 10
        self._decay: float  = 0.5
        self._line_search_max_num: int = 5
        self._nx: int = 2 # (p, v)
        self._nu: int = 1 # (u, )
        self._ntau: int = self._nx + self._nu
        self._T: int  = 10
        self._batch_size: int = 1
        self._delta_t: float = 0.1
        self._umin: float = -1.0
        self._umax: float = 1.0

        self._cost        = Cost(self._nx, self._nu, self._T)
        self._dynamic     = Dynamic(self._delta_t)
        self._metricfunc  = MetricFunction(self._nx, self._nu, self._T, self._cost, self._dynamic)

    def solve(self, x0: ndarray, xref: ndarray, total_step: int=50):
        '''
        Args:
            x0: current state, (nx, 1)
            xref: reference state, (nx, 1)
            total_step: total step of iteration, int
        Outputs:
            xs: sequence of states, (nx, total_step+1)
        '''
        xs = np.zeros([self._nx, total_step+1])
        xt = x0.copy(); xs[:, 0] = x0.flatten()
        for i in range(total_step):
            _x, _u = self._step(xt, xref)
            u0 = _u[0, :]
            xt = self._dynamic.step(xt.flatten(), u0)
            xs[:, i+1] = xt.flatten()
        return xs

    def _step(self, x0: ndarray, xref: ndarray, dveq0: Optional[ndarray]=None):
        '''
        Args:
            x0: current state, (nx, 1)
            xref: reference state, (nx, 1)
            dveq0: dual-varable for equation, ((T+1)*nx, 1)
        Outputs:
            xs: sequence of states, (T+1, nx)
            us: sequence of controls, (T, nu)
        '''
        _tau: Tau    = Tau(self._nx, self._nu, self._T)
        _tau.set_state_init(x0.flatten())
        dveq = dveq0.copy() if dveq0 is not None else self._get_init_dveq0()
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
                    J[t*self._ntau: (t+1)*self._ntau] = self._cost.get_stage_cost(_tau[t], xref=xref.flatten(), order=1).flatten()
                elif t == self._T:
                    J[t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(_tau.get_state(t), xref=xref.flatten(), order=1).flatten()

            # get Hessian Matrix
            for t in range(self._T+1):
                if t < self._T:
                    H[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] = \
                    self._cost.get_stage_cost(_tau[t], xref=xref.flatten(), order=2) + \
                    (dveq[(t+1)*self._nx: (t+2)*self._nx].reshape(-1, 1, 1) * self._dynamic.step(_tau[t], order=2)).sum(axis=0)
                elif t == self._T:
                    H[t*self._ntau: t*self._ntau+self._nx, t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(_tau.get_state(t), xref=xref.flatten(), order=2)
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
                if self._metricfunc(_tau + alpha*tau_delta, x0.flatten(), xref.flatten(), dveq_qp) < \
                self._metricfunc(_tau, x0.flatten(), xref.flatten(), dveq_qp):
                    break
                alpha = self._decay * alpha

            # update varible
            _tau += alpha * tau_delta
            dveq += alpha * dveq_delta
        return _tau.x, _tau.u

    def _convert_tensor_to_numpy_batch(self, ts: List[Tensor]) -> List[ndarray]:
        return [self._convert_tensor_to_numpy(t) for t in ts]

    def _convert_tensor_to_numpy(self, t: Tensor) -> ndarray:
        return t.cpu().detach().numpy().flatten()

    def _convert_numpy_to_tensor_batch(self, ns: List[ndarray]) -> List[Tensor]:
        return [self._convert_numpy_to_tensor(n) for n in ns]

    def _convert_numpy_to_tensor(self, n: ndarray) -> Tensor:
        if isinstance(n, ndarray):
            return torch.from_numpy(n).requires_grad_()
        raise TypeError('Please input varible type is `ndarray`')

    def _get_init_dveq0(self) -> ndarray:
        return np.zeros([(self._T+1)*self._nx, ])


