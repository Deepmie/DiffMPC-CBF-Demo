import casadi as ca
from casadi import DM
import numpy as np
from numpy import ndarray
from typing import Tuple, Union, Callable, Optional, Any
import matplotlib.pyplot as plt

class Cost:
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T

    def get_cost(self, X: Union[DM, ndarray], U: Union[DM, ndarray], XRef: Optional[Union[DM, ndarray]]=None) -> float: # get cost value from X and U
        cost = 0
        if XRef is None: XRef = np.zeros(self._nx)
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

    def _func(self, xt: Union[DM, ndarray]):
        return ca.DM if isinstance(xt, DM) else np.array


class iLQR:
    def __init__(self, iter_num: int = 10, verbose: int = 1):
        self._iter_num: int = iter_num
        self._decay: float  = 0.5
        self._line_search_max_num: int = 5
        self._verbose: int = verbose
        self._nx: int = 2 # (p, v)
        self._nu: int = 1 # (u, )
        self._ntau: int = self._nx + self._nu
        self._T: int  = 10
        self._delta_t: float = 0.1
        self._cost    = Cost(self._nx, self._nu, self._T)
        self._dynamic = Dynamic(self._delta_t)

    def step(self, x0: ndarray, xref: ndarray) -> Tuple:
        '''
        Args:
            x0: current state, (nx, 1)
            xref: reference state, (nx, 1)
        Outputs:
            _x: sequence of states, (nx, T+1)
            _u: sequence of states, (nu, T)
        '''
        x = np.zeros([self._nx, self._T+1]); x[:, 0] = x0.flatten()
        u = np.zeros([self._nu, self._T])
        for t in range(0, self._T):
            x[:, t+1] = self._dynamic.step(x[:, t], u[:, t]).flatten()

        for i in range(self._iter_num):
            alpha: int = 1
            ks = np.zeros([self._nu, self._T])
            Ks = np.zeros([self._nu, self._nx, self._T])
            # Terminal Value Function
            Vx  = self._cost.get_terminal_cost(x[:, self._T], xref.flatten(), order=1) # (nx, 1)
            Vxx = self._cost.get_terminal_cost(x[:, self._T], xref.flatten(), order=2) # (nx, nx)

            for t in range(self._T-1, -1, -1):
                F = self._dynamic.step(x[:, t], u[:, t], order=1) # (nx, ntau)
                Qtau = self._cost.get_stage_cost(x[:, t], u[:, t], xref.flatten(), order=1) + F.T @ Vx # (ntau, 1)
                Qtautau = self._cost.get_stage_cost(x[:, t], u[:, t], xref.flatten(), order=2) + F.T @ Vxx @ F # (ntau, ntau)
                Qx, Qu = Qtau[:self._nx, :], Qtau[self._nx:, :]
                Qxx, Qxu, Qux, Quu = Qtautau[:self._nx, :self._nx], Qtautau[:self._nx, self._nx:], Qtautau[self._nx:, :self._nx], Qtautau[self._nx:, self._nx:]
                k = np.linalg.solve(Quu, -Qu) # (nu, 1)
                K = np.linalg.solve(Quu, -Qux) # (nu, nx)
                Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qxu @ k # (nx, 1)
                Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qxu @ K # (nx, nx)

                # record k, K
                ks[:, t] = k.flatten()
                Ks[:, :, t] = K
            
            _x = np.zeros_like(x); _x[:, 0] = x0.flatten()
            _u = np.zeros_like(u)
            j  = 0
            while (self._cost.get_cost(_x, _u, xref.flatten()) >= self._cost.get_cost(x, u, xref.flatten())) and (j <= self._line_search_max_num):
                for t in range(self._T-1):
                    _u[:, t] = u[:, t] + alpha*ks[:, t] + (K @ (_x[:, t] - x[:, t]).reshape(-1, 1)).flatten()
                    _x[:, t+1] = self._dynamic.step(_x[:, t], _u[:, t]).flatten()
                alpha *= self._decay
                j += 1
        return _x, _u

    def solve(self, x0: ndarray, xref: ndarray, total_step: int=50) -> ndarray:
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
            _x, _u = self.step(xt, xref)
            u0 = _u[:, 0]
            xt = self._dynamic.step(xt, u0)
            xs[:, i+1] = xt.flatten()
        return xs

    def plot(self, xs: ndarray):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xs[0, :], xs[1, :], marker='x', color='k', markerfacecolor='r')
        ax.set_xlabel('p'); ax.set_ylabel('v')
        plt.show()
            


if __name__ == '__main__':
    x0   = np.array([[0.], [0.]])
    xref = np.array([[1.], [0.]])
    ilqr = iLQR()
    xs = ilqr.solve(x0, xref)
    ilqr.plot(xs)