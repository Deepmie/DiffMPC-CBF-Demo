from diffmpc_cbf import Cost, Dynamic
from typing import Dict, Tuple, Union, Optional
import torch
from torch import Tensor

class SpCost(Cost):
    def _define_stage_cost_function(self, ipt: Tensor, params: Dict[str, Tensor]) -> Tensor:
        return self.get_stage_cost_numerical(ipt, params, order=0)

    def _define_terminal_cost_function(self, xt: Tensor, params: Dict[str, Tensor]) -> Tensor:
        return self.get_terminal_cost_numerical(xt, params, order=0)
    
    def get_stage_cost_numerical(self, ipt: Tensor, params: Dict[str, Tensor], order: int = 0):
        pt, vt, ut = ipt
        pref, vref = params.get('xref')
        if order == 0:
            return 1/2*(pt-pref)**2 + 1/2*(vt-vref)**2 + 0.05*ut**2 + 0.02*(pt-pref)**4
        elif order == 1:
            return torch.tensor([[(pt-pref)+0.08*(pt-pref)**3], [vt-vref], [0.1*ut]])
        elif order == 2:
            return torch.tensor([
                [1+0.24*(pt-pref)**2, 0, 0],
                [0                  , 1, 0],
                [0                  , 0, 0.1],
            ])

    def get_terminal_cost_numerical(self, xt: Tensor, params: Dict[str, Tensor], order: int=0) -> Union[Tensor, float]:
        pt, vt = xt
        pref, vref = params.get('xref')
        if order == 0:
            return 5*(pt-pref)**2 + 5*(vt-vref)**2 + 0.5*(pt-pref)**4
        elif order == 1: # (2, 1)
            return torch.tensor([[10*(pt-pref) + 2*(pt-pref)**3], [10*(vt-vref)]])
        elif order == 2: # (2, 2)
            return torch.tensor([
                [10 + 6*(pt-pref)**2, 0],
                [0                  , 10]
            ])

class SpDynamic(Dynamic):
    def __init__(self, nx: int, nu: int, T: int, delta_t: float):
        super().__init__(nx, nu, T)
        self._delta_t = delta_t

    def _define_state_transition_equation(self, ipt, params=None):
        return self.forward_numerical(ipt, None, order=0)

    def forward_numerical(self, ipt: Tuple[Tensor], params=None, order: int=0) -> Tensor:
        pt, vt, ut = ipt
        if order == 0: # (2, 1)
            return torch.stack([pt+self._delta_t*vt, vt+self._delta_t*(ut-0.1*vt**3)])
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

def print_error(t1: Tensor, t2: Tensor) -> float:
    e = torch.sum((t1 - t2)**2).sqrt().item()
    print(f'{t1.numel()}-dim tensor have {e} error.')
    return e

def test():
    nx: int = 2
    nu: int = 1
    T: int = 5
    delta_t: float = 0.1
    cost = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, delta_t)

    ipt = torch.rand([3, 1], requires_grad=True)
    params = {'xref': torch.rand([2, 1])}

    # test cost
    gs = cost.get_stage_cost(ipt, params, order=1)
    gs_real = cost.get_stage_cost_numerical(ipt, params, order=1)
    hs = cost.get_stage_cost(ipt, params, order=2)
    hs_real = cost.get_stage_cost_numerical(ipt, params, order=2)
    print_error(gs, gs_real)
    print_error(hs, hs_real)

    # test dynamic
    gf = dynamic.forward(ipt, None, order=1)
    gf_real = dynamic.forward_numerical(ipt, None, order=1)
    hf = dynamic.forward(ipt, None, order=2)
    hf_real = dynamic.forward_numerical(ipt, None, order=2)
    print_error(gf, gf_real)
    print_error(hf, hf_real)

if __name__ == '__main__':
    test()