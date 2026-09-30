from diffmpc_cbf import Cost, Dynamic
from typing import Tuple, Union, Optional
import numpy as np
from numpy import ndarray

class SpCost(Cost):
    def get_stage_cost(self, ipt: Tuple[ndarray], xref: Optional[Tuple[ndarray]]=None, order: int=0) -> Union[ndarray, float]:
        super().get_stage_cost(ipt, xref, order)
        xt, ut = ipt
        pt, vt, ut = xt[0], xt[1], ut[0]
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

    def get_terminal_cost(self, xt: ndarray, xref: Optional[Tuple[ndarray]]=None, order: int=0) -> Union[ndarray, float]:
        super().get_terminal_cost(xt, xref, order)
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


class SpDynamic(Dynamic):
    def __init__(self, delta_t: float):
        super().__init__()
        self._delta_t = delta_t

    def step(self, ipt: Tuple[ndarray], order: int=0) -> ndarray:
        xt, ut = ipt
        pt, vt, ut = xt[0], xt[1], ut[0]
        if order == 0: # (2, 1)
            return np.array([[pt+self._delta_t*vt], [vt+self._delta_t*(ut-0.1*vt**3)]])
        elif order == 1:
            return np.array([ # (2, 3)
                [1.0, self._delta_t, 0.0],
                [0.0, 1-0.3*self._delta_t*vt**2, self._delta_t]
            ])
        elif order == 2:
            return np.array([ # (2, 3, 3)
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, -0.6*self._delta_t*vt, 0],
                 [0, 0, 0]]
            ])