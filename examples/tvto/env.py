import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

class CircObs:
    def __init__(
            self,
            p: ndarray,
            A: ndarray, # (4, )
            B: ndarray, # (4, )
            omega: ndarray,
            phi: ndarray,
        ): # (2, )
        self._p = p; self._A = A; self._B = B; self._omega = omega; self._phi = phi

    def step(self, t: int) -> ndarray:
        Ax, Ay, Ax_delta, Ay_delta = self._A
        Bx, By, Bx_delta, By_delta = self._B
        omegax, omegay, omegax_delta1, omegax_delta2, omegay_delta1, omegay_delta2 = self._omega
        phix, phiy, phix_delta1, phix_delta2, phiy_delta1, phiy_delta2 = self._phi
        self._p = np.array([
            Ax+Bx*np.cos(omegax*t+phix),
            Ay+By*np.sin(omegay*t+phiy),
        ]) + np.array([
            Ax_delta*np.sin(omegax_delta1*t+phix_delta1) + Bx_delta*np.sin(omegax_delta2*t+phix_delta2),
            Ay_delta*np.sin(omegay_delta1*t+phiy_delta1) + By_delta*np.sin(omegay_delta2*t+phiy_delta2),
        ])

    @property
    def state(self) -> ndarray:
        return self._p

    def _delta(self, t: int, A1: ndarray, A2: ndarray, omega1: ndarray, omega2: ndarray) -> ndarray:
        phi = np.random.uniform(0, 2*np.pi, [4])
        return 


class Env:
    def __init__(self, x0: ndarray, xref: ndarray):
        self._x0 = x0; self._xref = xref

    def step(self):
        ...

    def set_obs(self):
        ...

    def show(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self._x0[0], self._x0[1], s=15, c='k')
        ax.scatter(self._xref[0], self._xref[1], marker='*', s=45, c='r', alpha=0.5)
        ax.text(self._x0[0], self._x0[1], 'start')
        ax.text(self._xref[0], self._xref[1], 'target')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        plt.show()


if __name__ == '__main__':
    env = Env()