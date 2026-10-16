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
        return np.array([
            self._A[0]+self._B[0]*np.cos(self._omega[0]*t+self._phi[0]),
            self._A[1]+self._B[1]*np.sin(self._omega[1]*t+self._phi[1]),
        ]) + np.array([
            self._A[0]*np.sin(omega1[0]*t+phi[0]) + A2[0]*np.sin(omega2[0]*t+phi[1]),
            self._A[0]*np.sin(omega1[1]*t+phi[2]) + A2[1]*np.sin(omega2[1]*t+phi[3]),
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