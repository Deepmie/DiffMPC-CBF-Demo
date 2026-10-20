import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Tuple, Optional

class CircObs:
    def __init__(
            self,
            r: float,
            dr: float,
            # motion params
            A: ndarray, # (4, )
            B: ndarray, # (4, )
            omega: ndarray, # (6, )
            phi: ndarray, # (2, )
            delta_t: float,
        ):
        self._r = r; self._dr = dr
        self._A = A; self._B = B; self._omega = omega
        self._phi = phi; self._delta_t = delta_t

    def step(self) -> ndarray:
        _t = self._num * self._delta_t
        Ax, Ay, Ax_delta, Ay_delta = self._A
        Bx, By, Bx_delta, By_delta = self._B
        omegax, omegay, omegax_delta1, omegax_delta2, omegay_delta1, omegay_delta2 = self._omega
        phix, phiy = self._phi
        phix_delta1, phix_delta2, phiy_delta1, phiy_delta2 = self._phi_delta
        self._p = np.array([
            Ax+Bx*np.cos(omegax*_t+phix),
            Ay+By*np.sin(omegay*_t+phiy),
        ]) + np.array([
            Ax_delta*np.sin(omegax_delta1*_t+phix_delta1) + Bx_delta*np.sin(omegax_delta2*_t+phix_delta2),
            Ay_delta*np.sin(omegay_delta1*_t+phiy_delta1) + By_delta*np.sin(omegay_delta2*_t+phiy_delta2),
        ])
        self._num += 1
        return self._p

    def predict(self, T: int) -> List[ndarray]:
        _num = self._num
        Ax, Ay, Ax_delta, Ay_delta = self._A
        Bx, By, Bx_delta, By_delta = self._B
        omegax, omegay, omegax_delta1, omegax_delta2, omegay_delta1, omegay_delta2 = self._omega
        phix, phiy = self._phi
        phix_delta1, phix_delta2, phiy_delta1, phiy_delta2 = self._phi_delta
        _ps: List[ndarray] = []
        # iteration
        for i in range(T):
            _t = _num * self._delta_t
            _p = np.array([
                Ax+Bx*np.cos(omegax*_t+phix),
                Ay+By*np.sin(omegay*_t+phiy),
            ]) + np.array([
                Ax_delta*np.sin(omegax_delta1*_t+phix_delta1) + Bx_delta*np.sin(omegax_delta2*_t+phix_delta2),
                Ay_delta*np.sin(omegay_delta1*_t+phiy_delta1) + By_delta*np.sin(omegay_delta2*_t+phiy_delta2),
            ])
            _ps.append({'state': _p.reshape(-1, 1), 'radius': self.radius_ws})
            _num += 1
        return _ps

    def reset(self):
        self._phi_delta = np.random.uniform(0, 2*np.pi, [4])
        self._num: int = 0
        self.step()

    @property
    def state(self) -> ndarray:
        return self._p

    @property
    def radius(self) -> float:
        return self._r

    @property
    def radius_ws(self) -> float:
        return self._r + self._dr


class Agent:
    def __init__(self, r: float, delta_t: float, p0: Optional[ndarray]=None):
        self._r = r; self._delta_t = delta_t
        self._p0 = p0

    def step(self, u: ndarray) -> ndarray: # (nu, 1)
        p, v = self._z[0:2], self._z[2:4]
        self._z = np.concat([
            p + self._delta_t*v + 1/2*self._delta_t**2*u,
            v + self._delta_t*u,
        ], axis=0)
        return self._z

    def reset(self):
        self._z: ndarray = np.zeros([4, 1])
        if self._p0 is not None: self._z[0: 2] = self._p0.reshape(-1, 1)

    @property
    def state(self) -> ndarray:
        return self._z

    @property
    def position(self) -> ndarray:
        return self._z[0: 2]

    @property
    def radius(self) -> float:
        return self._r


class Env:
    def __init__(
            self,
            p0: ndarray, # agents' init point
            pref: ndarray, # agents' target point
            delta_t: float=0.01, # env run delta t
        ):
        self._p0 = p0; self._pref = pref
        self._obs = [
            CircObs(
                r  = 0.06,
                dr = 0.02,
                A  = [0.50, 0.30,  0.012, 0.008],
                B  = [0.24, 0.025, 0.008, 0.006],
                omega = [2*np.pi/6, 2*np.pi/7, 2*np.pi/5, 2*np.pi/3.7, 2*np.pi/4.8, 2*np.pi/3.9],
                phi = [0.0, 0.0],
                delta_t = delta_t,
            ),
            CircObs(
                r  = 0.06,
                dr = 0.02,
                A = [0.70,  0.50, 0.008, 0.012],
                B = [0.025, 0.24, 0.006, 0.008],
                omega = [2*np.pi/7, 2*np.pi/6.8, 2*np.pi/4.6, 2*np.pi/3.8, 2*np.pi/5.2, 2*np.pi/4.0],
                phi = [0.0, np.pi/3],
                delta_t = delta_t,
            ),
        ]
        self._delta_t = delta_t
        self._agent = Agent(r = 0.02, delta_t = delta_t, p0 = p0)
        self.reset()
    
    def step(self, u: ndarray) -> Tuple[ndarray, bool]:
        _done: bool = False
        self._num += 1
        self._t = self._num * self._delta_t

        # obstacale move
        _obs_state = []
        for o in self._obs:
            o.step()
            _obs_state.append(o.state)
        self._obs_states.append(_obs_state)

        # agent move
        _agent_state = self._agent.step(u)
        self._agent_states.append(_agent_state.flatten())

        if self._detect_collision(): _done = True
        if self._detect_finish_game(): _done = True
        return _agent_state, _done

    def reset(self):
        self._agent.reset()
        for o in self._obs: o.reset()
        # record history sequence
        self._num = 0 # set abs clock
        self._obs_states: List[List[ndarray]] = []
        self._obs_states.append([o.state for o in self._obs])
        self._agent_states: List[ndarray] = []
        self._agent_states.append(self._agent.state.flatten())

    def show(self):
        self._circs: List[Tuple[Circle]] = []
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self._p0[0], self._p0[1], s=15, c='k')
        ax.scatter(self._pref[0], self._pref[1], marker='*', s=45, c='r', alpha=0.5)
        ax.text(self._p0[0], self._p0[1], 'start')
        ax.text(self._pref[0], self._pref[1], 'target')

        for o in self._obs:
            po = o.state
            circ1 = Circle((po[0], po[1]), radius=o.radius_ws, facecolor='w', alpha=0.4, linestyle='--', edgecolor='b')
            circ2 = Circle((po[0], po[1]), radius=o.radius, facecolor='green', alpha=0.3)
            ax.add_patch(circ1); ax.add_patch(circ2)
            self._circs.append((circ1, circ2))

        pa = self._agent.position.flatten()
        self._circ_agt = Circle((pa[0], pa[1]), radius=self._agent.radius, facecolor='r', alpha=0.7)
        ax.add_patch(self._circ_agt)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')

        ani = FuncAnimation(fig, self.update, frames=self._num, interval=50, blit=True)
        ani.save('imgs/tvto.gif', writer=PillowWriter(fps=20))

    def update(self, frame: int) -> List[Tuple[Circle]]:
        _circs: List[Tuple[Circle]] = []
        for po, (circ1, circ2) in zip(self._obs_states[frame], self._circs):
            circ1.center = (po[0], po[1])
            circ2.center = (po[0], po[1])
            _circs.extend([circ1, circ2])
        sa = self._agent_states[frame]
        self._circ_agt.center = (sa[0], sa[1])
        _circs.append(self._circ_agt)
        return _circs

    def get_obs_info(self, T: int) -> List[List[Dict]]:
        obs_info_predicted  = [o.predict(T) for o in self._obs]
        obs_info = []
        for op in obs_info_predicted:
            r = []
            for opt in op:
                _opt = opt.copy()
                _opt['radius'] = opt['radius'] + self._agent.radius
                r.append(_opt)
            obs_info.append(r)
        return obs_info

    def _detect_collision(self) -> bool:
        for o in self._obs:
            po = o.state
            pa = self._agent.position.flatten()
            if ((po-pa)**2).sum() - (o.radius_ws+self._agent.radius)**2 < 0:
                return True
        return False

    def _detect_finish_game(self) -> bool:
        pa = self._agent.position.flatten()
        if ((pa-self._pref)**2).sum() < 1e-3: return True
        return False

    @property
    def state(self) -> ndarray: # (4, 1)
        return self._agent.state


if __name__ == '__main__':
    p0 = np.array([0, 0])
    pref = np.array([1, 1])
    env = Env(p0, pref)
    for _ in range(100):
        u = 0.5*np.random.random([2, 1])
        env.step(u)
    env.show()