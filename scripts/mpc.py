import casadi as ca
from casadi import MX
import numpy as np

class Cost:
    def __init__(self, nx: int, nu: int, T: int):
        self._nx = nx; self._nu = nu; self._T = T

    def get_cost(self, X: MX, U: MX) -> MX: # get cost value from X and U
        cost = 0
        # Stage Cost
        for t in range(self._T):
            pt, vt = X[:, t]; ut = U[:, t]
            cost += 1/2*(pt-1)**2 + 1/2*vt**2 + 0.05*ut**2 + 0.02*(pt-1)**4
        # Terminal Cost
        pT, vT = X[:, self._T]
        cost += 5*(pT-1)**2 + 5*vT**2+0.5*(pT-1)**4
        return cost

class Dynamic:
    def __init__(self, delta_t: float):
        self.delta_t = delta_t

    def step(self, X: MX, U: MX, t: int) -> MX: # run one step
        pt, vt = X[:, t]; ut = U[:, t]
        pt_next = pt + self.delta_t*vt
        vt_next = vt + self.delta_t*(ut-0.1*vt**3)
        return ca.vertcat(pt_next, vt_next) # (2, 1)
    
class MPC:
    def __init__(self):
        self._nx: int = 2 # (p, v)
        self._nu: int = 1 # (u, )
        self._T: int  = 10
        self._delta_t: float = 0.1
        self._cost    = Cost(self._nx, self._nu, self._T)
        self._dynamic = Dynamic(self._delta_t)
        self._build_problem()

    def _build_problem(self):
        self._opti = ca.Opti()
        self.X  = self._opti.variable(self._nx, self._T+1)
        self.U  = self._opti.variable(self._nu, self._T)
        self.X0 = self._opti.parameter(self._nx, 1)

        # Cost Function
        cost = self._cost.get_cost(self.X, self.U)
        self._opti.minimize(cost)

        # Constraints
        self._opti.subject_to(self.X[:, 0] == self.X0)
        for t in range(self._T):
            x_next = self._dynamic.step(self.X, self.U, t)
            self._opti.subject_to(self.X[:, t+1] == x_next)

        # Solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self._opti.solver('ipopt', opts)

    def solve(self):
        


if __name__ == '__main__':
    a = ca.MX.sym('a', 3, 2)
    b = ca.MX.sym('b', 2, 2)
    print(ca.vertcat(a[:, 0], b[:, 0]).shape)