from diffmpc_cbf import Cost, Dynamic
from typing import Dict, Tuple, Union, Optional
import torch
from torch import Tensor
from net import ContextNetwork


class SpnnCost(Cost):
    def __init__(self, nx: int, nu: int, T: int, context_dim: int=3):
        super().__init__(nx, nu, T)
        self.context_network = ContextNetwork(context_dim)

    def get_stage_cost(self, ipt: Tuple[Tensor], params: Dict[str, Tensor], order: int=0) -> Union[Tensor, float]:
        super().get_stage_cost(ipt, params, order)
        xref = params.get('xref')
        xt, ut = self._ipt
        pt, vt, ut = xt[0], xt[1], ut[0]
        pref, vref = xref[0], xref[1]
        if order == 0: # raw function
            return 1/2*(pt-pref)**2 + 1/2*(vt-vref)**2 + 0.05*ut**2 + 0.02*(pt-pref)**4

    def get_terminal_cost(self, xt: Tensor, params: Dict[str, Tensor], order: int=0) -> Union[Tensor, float]:
        super().get_terminal_cost(xt, params, order)
        xref = params.get('xref')
        pt, vt = xt[0], xt[1]
        pref, vref = xref[0], xref[1]
        if order == 0:
            return 5*(pt-pref)**2 + 5*(vt-vref)**2 + 0.5*(pt-pref)**4
        elif order == 1: # (2, 1)
            return torch.tensor([[10*(pt-pref) + 2*(pt-pref)**3], [10*(vt-vref)]])
        elif order == 2: # (2, 2)
            return torch.tensor([
                [10 + 6*(pt-pref)**2, 0],
                [0                  , 10]
            ])



class SpnnDynamic(Dynamic):
    def __init__(self, nx: int, nu: int, T: int, delta_t: float):
        super().__init__(nx, nu, T)
        self._delta_t = delta_t

    def forward(self, ipt: Tuple[Tensor], params: Optional[Dict[str, Tensor]],order: int=0) -> Tensor:
        super().forward(ipt, order)
        xt, ut = self._ipt
        pt, vt, ut = xt[0], xt[1], ut[0]
        if order == 0: # (2, 1)
            return torch.tensor([[pt+self._delta_t*vt], [vt+self._delta_t*(ut-0.1*vt**3)]])
        elif order == 1:
            return torch.tensor([ # (2, 3)
                [1.0, self._delta_t, 0.0],
                [0.0, 1-0.3*self._delta_t*vt**2, self._delta_t]
            ])
        elif order == 2:
            return torch.tensor([ # (2, 3, 3)
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, -0.6*self._delta_t*vt, 0],
                 [0, 0, 0]]
            ])