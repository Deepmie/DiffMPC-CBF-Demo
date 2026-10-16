import numpy as np
from numpy import ndarray
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Env:
    def __init__(
            self,
            x0: ndarray, # (nx,)
            xref: ndarray, # (nx,)
            obs: List[ndarray], # [(nx+2,), ...]
            delta_t: float=0.1,
        ):
        self._x0 = x0; self._xref = xref; self._obs = obs; self._delta_t = delta_t
        self.reset()

    def step(self, ut: ndarray) -> ndarray: # (nu, )
        ut = ut[0]
        pt, vt = self._x
        self._x = np.array([
            pt+self._delta_t*vt,
            vt+self._delta_t*(ut-0.1*vt**3),
        ])
        self._xs.append(self._x.copy())
        return self._x

    def reset(self):
        self._x: ndarray = self._x0.copy() # (nx, )
        self._xs: List[ndarray] = [self._x0.copy(), ]

    def show(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        # ax.axis('off')
        ax.scatter(self._x0[0], self._x0[1], s=15, c='k')
        ax.scatter(self._xref[0], self._xref[1], marker='*', s=45, c='r', alpha=0.5)
        ax.text(self._x0[0], self._x0[1], 'start')
        ax.text(self._xref[0], self._xref[1], 'target')

        for o in self._obs:
            circ1 = Circle((o[0], o[1]), radius=o[2]+o[3], facecolor='skyblue', alpha=0.4, linestyle='--', edgecolor='b')
            circ2 = Circle((o[0], o[1]), radius=o[2], facecolor='green', alpha=0.3)
            ax.add_patch(circ1)
            ax.add_patch(circ2)

        _xs = np.stack(self._xs) # (N, nx)
        ax.scatter(_xs[:, 0], _xs[:, 1], marker='h', c='k')

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        plt.show()
        


if __name__ == '__main__':
    x0: ndarray   = np.array([0., 0.])
    xref: ndarray = np.array([1., 0.6])
    obs: List[ndarray] = [np.array([0.5, 0.5, 0.3, 0.02]), ]
    env = Env(x0, xref, obs)
    env.show()