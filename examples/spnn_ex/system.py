from diffmpc_cbf import Cost, Dynamic
from typing import Dict, Tuple, Union, Optional
import torch
from torch import Tensor
from net import ContextNetwork

class SpnnCost(Cost):
    def __init__(self, nx: int, nu: int, T: int, context_dim: int=3):
        super().__init__(nx, nu, T)
        self.context_network = ContextNetwork(context_dim)

    def _define_stage_cost_function(self, ipt: Tensor, params: Dict[str, Tensor]) -> Tensor:
        pt, vt, ut = ipt[0], ipt[1], ipt[2]
        pref, vref = params.get('xref')
        t = torch.tensor([params.get('t'), ])
        ctx = torch.stack([pref, vref, t]).flatten()
        hid: Tensor = self.context_network(ctx) # (3, )
        return hid[0]*1/2*(pt-pref)**2 + hid[1]*1/2*(vt-vref)**2 + hid[2]*0.05*ut**2 + 0.02*(pt-pref)**4
    
    def _define_terminal_cost_function(self, xt: Tensor, params: Dict[str, Tensor]) -> Tensor:
        pt, vt = xt[0], xt[1]
        pref, vref = params.get('xref')
        t = torch.tensor([params.get('t'), ])
        ctx = torch.stack([pref, vref, t]).flatten()
        hid: Tensor = self.context_network(ctx) # (3, )
        return hid[0]*5*(pt-pref)**2 + hid[1]*5*(vt-vref)**2 + 0.2*(pt-pref)**4


class SpnnDynamic(Dynamic):
    def __init__(self, nx: int, nu: int, T: int, delta_t: float):
        super().__init__(nx, nu, T)
        self._delta_t = delta_t
    
    def _define_state_transition_equation(self, ipt: Tensor, params=None):
        pt, vt, ut = ipt
        return torch.stack([pt+self._delta_t*vt, vt+self._delta_t*(ut-0.1*vt**3)])
