import casadi as ca
from casadi import DM
import numpy as np
from numpy import ndarray
from typing import Optional, Union, Callable, Any
import matplotlib.pyplot as plt

class Cost:
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T

    def get_cost(self, X: Union[DM, ndarray], U: Union[DM, ndarray], XRef: Optional[Union[DM, ndarray]]=None) -> float: # get cost value from X and U
        cost = 0
        if XRef is not None: XRef = [0, 0]
        # Stage Cost
        for t in range(self._T):
            cost += self.get_stage_cost(X[:, t], U[:, t], XRef)
        # Terminal Cost
        cost += self.get_terminal_cost(X[:, self._T], XRef)
        return cost

    def get_stage_cost(self, xt: Union[DM, ndarray], ut: Union[DM, ndarray], xref: Union[DM, ndarray], order: int=0) -> Union[DM, ndarray, float]:
        pt, vt, ut = xt[0], xt[1], ut[0]
        pref, vref = xref[0], xref[1]
        if order == 0: # raw function
            return 1/2*(pt-pref)**2 + 1/2*(vt-vref)**2 + 0.05*ut**2 + 0.02*(pt-pref)**4
        elif order == 1: # 1-order
            return self._func(xt)([[(pt-pref)+0.08*(pt-pref)**3], [vt-vref], [0.1*ut]])
        elif order == 2: # 2-order
            return self._func(xt)([
                [1+0.24*(pt-pref)**2, 0, 0],
                [0                  , 1, 0],
                [0                  , 0, 0.1],
            ])

    def get_terminal_cost(self, xt: Union[DM, ndarray], xref: Union[DM, ndarray], order: int=0) -> Union[DM, ndarray, float]:
        pt, vt = xt[0], xt[1]
        pref, vref = xref[0], xref[1]
        if order == 0:
            return 5*(pt-pref)**2 + 5*(vt-vref)**2 + 0.5*(pt-pref)**4
        elif order == 1:
            return self._func(xt)([[10*(pt-pref) + 2*(pt-pref)**3], [10*(vt-vref)]])
        elif order == 2:
            return self._func(xt)([
                [10 + 6*(pt-pref)**2, 0],
                [0                  , 10]
            ])

    def _func(self, xt: Union[DM, ndarray]) -> Callable[[Any], Union[DM, ndarray]]:
        return ca.DM if isinstance(xt, DM) else np.array
        

class Dynamic:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t

    def step(self, xt: Union[DM, ndarray], ut: Union[DM, ndarray], order: int=0) -> Union[DM, ndarray]:
        pt, vt, ut = xt[0], xt[1], ut[0]
        if order == 0:
            return self._func(xt)([[pt+self.delta_t*vt], [vt+self.delta_t*(ut-0.1*vt**3)]]) # (2, 1)
        elif order == 1:
            return self._func(xt)([ # (2, 3)
                [1.0, self.delta_t, 0.0],
                [0.0, 1-0.3*self.delta_t*vt**2, self.delta_t]
            ])
        elif order == 2:
            return self._func(xt)([ # (2, 3, 3)
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, -0.6*self.delta_t*vt, 0],
                 [0, 0, 0]]
            ])

    def _func(self, xt: Union[DM, ndarray]):
        return ca.DM if isinstance(xt, DM) else np.array



class SQP:
    def __init__(self, verbose: int = 1):
        self._verbose = verbose
        self._nx: int = 2 # (p, v)
        self._nu: int = 1 # (u, )
        self._ntau: int = self._nx + self._nu
        self._T: int  = 10
        self._delta_t: float = 0.1
        self._cost    = Cost(self._nx, self._nu, self._T)
        self._dynamic = Dynamic(self._delta_t)

    def _step(self, x0: ndarray, xref: ndarray, dveq0: ndarray, iter_nums: int):
        '''
        Args:
            x0: current state, (nx, 1)
            xref: reference state, (nx, 1)
            dveq0: dual-varable for equation, (nx*T, 1)
        Outputs:
            xs: sequence of states, (nx, total_step+1)
        '''
        x = np.zeros([self._nx, self._T+1]); x[:, 0] = x0.flatten()
        u = np.zeros([self._nu, self._T])
        dveq = dveq0.copy()
        for t in range(self._T-1):
            x[:, t+1] = self._dynamic.step(x[:, t], u[:, t]).flatten()

        for i in range(iter_nums-1):
            alpha: float = 1.0
            var_nums: int = self._T*self._ntau+self._nx
            J: ndarray = np.zeros(var_nums, 1)
            H: ndarray = np.zeros([var_nums, var_nums])

            # get Jacobian Vector
            for t in range(self._T+1):
                if t < self._T:
                    J[t*self._ntau: (t+1)*self._ntau] = self._cost.get_stage_cost(x[:, t], u[:, t], xref.flatten(), order=1)
                elif t == self._T:
                    J[t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(x[: t], xref.flatten(), order=1)

            # get Hessian Matrix
            for t in range(self._T+1):
                if t < self._T:
                    H[t*self._ntau: (t+1)*self._ntau, t*self._ntau: (t+1)*self._ntau] = \
                    self._cost.get_stage_cost(x[:, t], u[:, t], xref.flatten(), order=2) + \
                    (dveq[t*self._nx: (t+1)*self._nx].reshape(-1, 1, 1) * self._dynamic.step(x[:, t], u[:, t], order=2)).sum(axis=0)
                elif t == self._T:
                    H[t*self._ntau: t*self._ntau+self._nx, t*self._ntau: t*self._ntau+self._nx] = self._cost.get_terminal_cost(x[:, t], xref.flatten(), order=2)

            # get Jacobian Matrix for equation
            