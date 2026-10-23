import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Tuple, Optional

class Clock:
    def __init__(self, delta_t: float):
        self._delta_t = delta_t
        self.reset()

    def step(self):
        self._num += 1

    def reset(self):
        self._num = 0

    @property
    def t(self) -> float:
        return self._num * self._delta_t

    @property
    def num(self) -> int:
        return self._num

    @property
    def delta_t(self) -> float:
        return self._delta_t

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
            clock: Clock,
        ):
        self._r = r; self._dr = dr
        self._A = A; self._B = B; self._omega = omega
        self._phi = phi; self._clock = clock

    def step(self) -> ndarray:
        _t = self._clock.t
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
        return self._p

    def predict(self, T: int) -> List[Dict]:
        _num = self._clock.num
        Ax, Ay, Ax_delta, Ay_delta = self._A
        Bx, By, Bx_delta, By_delta = self._B
        omegax, omegay, omegax_delta1, omegax_delta2, omegay_delta1, omegay_delta2 = self._omega
        phix, phiy = self._phi
        phix_delta1, phix_delta2, phiy_delta1, phiy_delta2 = self._phi_delta
        _ps: List[Dict] = []
        # iteration
        for i in range(T):
            _t = _num * self._clock.delta_t
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
    def __init__(self, r: float, b: float, clock: Clock, p0: Optional[ndarray]=None):
        self._r = r; self._b = b; self._clock = clock; self._delta_t = self._clock.delta_t
        self._p0 = p0

    def step(self, u: ndarray) -> ndarray: # (nu, 1)
        p, v = self._z[0:2], self._z[2:4]
        d_true = -self._b * v
        aeff = u + d_true
        self._z = np.concat([
            p + self._delta_t*v + 1/2*self._delta_t**2*aeff,
            v + self._delta_t*aeff,
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
        self._clock = Clock(delta_t)
        self._p0 = p0; self._pref = pref
        self._obs = [
            CircObs(
                r  = 0.06,
                dr = 0.02,
                A  = [0.50, 0.30,  0.012, 0.008],
                B  = [0.24, 0.025, 0.008, 0.006],
                omega = [2*np.pi/6, 2*np.pi/7, 2*np.pi/5, 2*np.pi/3.7, 2*np.pi/4.8, 2*np.pi/3.9],
                phi = [0.0, 0.0],
                clock=self._clock,
            ),
            CircObs(
                r  = 0.06,
                dr = 0.02,
                A = [0.70,  0.50, 0.008, 0.012],
                B = [0.025, 0.24, 0.006, 0.008],
                omega = [2*np.pi/7, 2*np.pi/6.8, 2*np.pi/4.6, 2*np.pi/3.8, 2*np.pi/5.2, 2*np.pi/4.0],
                phi = [0.0, np.pi/3],
                clock=self._clock,
            ),
        ]
        self._delta_t = delta_t
        self._agent = Agent(r = 0.02, b = 0.5, clock = self._clock, p0 = p0)
        self.reset()
    
    def step(self, u: ndarray) -> Tuple[ndarray, bool]:
        _done: bool = False
        self._clock.step()

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
        self._clock.reset()
        self._agent.reset()
        for o in self._obs: o.reset()
        # record history sequence
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

        ani = FuncAnimation(fig, self.update, frames=self._clock.num, interval=50, blit=True)
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
        return [o.predict(T) for o in self._obs]

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

    @property
    def radius_agent(self) -> float:
        return self._agent.radius


if __name__ == '__main__':
    p0 = np.array([0, 0])
    pref = np.array([1, 1])
    env = Env(p0, pref)
    for _ in range(100):
        u = 0.5*np.random.random([2, 1])
        env.step(u)
    env.show()