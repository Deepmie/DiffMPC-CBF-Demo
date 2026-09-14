import casadi as ca
from casadi import MX
import numpy as np
from numpy import ndarray
from typing import Optional, Union
import matplotlib.pyplot as plt

class Cost:
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T

    def get_cost(self, X: MX, U: MX, XRef: Optional[MX]=None) -> MX: # get cost value from X and U
        cost = 0
        if XRef is not None:
            pref, vref = XRef[0, 0], XRef[1, 0]
        else:
            pref, vref = 0, 0
        # Stage Cost
        for t in range(self._T):
            pt, vt = X[0, t], X[1, t]; ut = U[:, t]
            cost += 1/2*(pt-pref)**2 + 1/2*(vt-vref)**2 + 0.05*ut**2 + 0.02*(pt-pref)**4
        # Terminal Cost
        pT, vT = X[0, self._T], X[1, self._T]
        cost += 5*(pT-pref)**2 + 5*(vT-vref)**2+0.5*(pT-pref)**4
        return cost

class Dynamic:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t

    def step(self, xt: Union[MX, ndarray], ut: Union[MX, ndarray]) -> Union[MX, ndarray]: # run one step
        pt, vt = xt
        pt_next = pt + self.delta_t*vt
        vt_next = vt + self.delta_t*(ut-0.1*vt**3)
        # Result Dimension: (2, 1)
        return ca.vertcat(pt_next, vt_next) if isinstance(xt, MX) else np.concat([pt_next.reshape(-1, 1), vt_next.reshape(-1, 1)])
    
class MPC:
    def __init__(self, verbose: int = 1):
        self._verbose: int = verbose
        self._nx: int = 2 # (p, v)
        self._nu: int = 1 # (u, )
        self._T: int  = 10
        self._delta_t: float = 0.1
        self._cost    = Cost(self._nx, self._nu, self._T)
        self._dynamic = Dynamic(self._delta_t)
        self._build_problem()

    def _build_problem(self):
        self._opti = ca.Opti()
        self.X    = self._opti.variable(self._nx, self._T+1)
        self.U    = self._opti.variable(self._nu, self._T)
        self.X0   = self._opti.parameter(self._nx, 1)
        self.XRef = self._opti.parameter(self._nx, 1)

        # Cost Function
        cost = self._cost.get_cost(self.X, self.U, self.XRef)
        self._opti.minimize(cost)

        # Constraints
        self._opti.subject_to(self.X[:, 0] == self.X0)
        for t in range(self._T):
            x_next = self._dynamic.step(self.X[:, t], self.U[:, t])
            self._opti.subject_to(self.X[:, t+1] == x_next)

        # Solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self._opti.solver('ipopt', opts)

    def solve(self, x0: ndarray, xref: ndarray, total_step: int=100) -> ndarray:
        '''
        Args:
            x0: current state, (nx, 1)
            xref: reference state, (nx, 1)
            total_step: total step of iteration, int
        Output:
            xs: sequence of state, (nx, T+1)
        '''
        xs = np.zeros(self._nx, self._T+1)
        xt = x0.copy(); xs[:, 0] = x0.copy()
        for i in range(total_step):
            self._opti.set_value(self.X0, xt)
            self._opti.set_value(self.XRef, xref)
            # warm start
            self._opti.set_initial(self.X, np.tile(xt, (1, self._T+1)))
            self._opti.set_initial(self.U, np.zeros([self._nu, self._T]))
            sol = self._opti.solve()
            x_opt, u_opt = self._opti.value(self.X), self._opti.value(self.U)
            u0 = u_opt[:, 0]
            xt = self._dynamic.step(xt, u0)

            xs[:, i+1] = xt
            if self._verbose: print(f'step: {i+1} | state: {xt} | control: {u0}')
        return xs

    def plot(self, xs: ndarray):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(xs, marker='x', color='k', markerfacecolor='r')
        ax.set_xlabel('p'); ax.set_ylabel('v')
        plt.show()


if __name__ == '__main__':
    x0   = np.array([[0.], [0.]])
    xref = np.array([[1.], [0.]])
    mpc  = MPC()
    xs   = mpc.solve(x0, xref)
    mpc.plot(xs)