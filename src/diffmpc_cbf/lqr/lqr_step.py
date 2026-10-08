from torch.autograd.function import Function, BackwardCFunction
import torch
from torch import Tensor
from typing import Dict, Tuple, Optional, Union, Callable, cast
from .solve import lqr_step_solve, mpc_solve
from .. import Cost, QuadCost, Dynamic, LinDynamic, iLQRMetricFunction

def get_lambdas(
        ipt: Union[Tensor, Tuple[Tensor]],
        nx: int,
        nu: int,
        T: int,
        C: Tensor, # (T+1, ntau, ntau)
        c: Tensor, # (T+1, ntau)
        F: Tensor, # (T, nx, ntau)
    ) -> Tensor:
    ntau = nx + nu
    if not torch.is_tensor(ipt):
        if len(ipt) == 1: ipt = ipt[0]
        elif len(ipt) == 2:
            _tau = torch.zeros([T+1, ntau])
            _tau[:, :nx] = ipt[0]; _tau[:T, nx:] = ipt[1]
            ipt = _tau # (T+1, ntau)
    
    lambdas: Tensor = torch.zeros([T+1, nx]) # (T+1, nx)
    for t in range(T, -1, -1):
        lambdas[t] = (C[t, :nx, :nx] @ ipt[t, :nx].unsqueeze(-1)).squeeze() + c[t, :nx] # (nx, )
        if t < T: lambdas[t] += (F[t, :, :nx].permute(1, 0) @ lambdas[t+1].unsqueeze(-1)).squeeze() # (nx, )
    return lambdas # (T+1, nx)

def LQRStep(
        x0: Tensor,
        x: Tensor,
        u: Tensor,
        dynamic: Dynamic,
        cost: Cost,
        params: Optional[Dict[str, Tensor]],
        metricfunc: iLQRMetricFunction,
        iter_nums: int,
        line_search_max_num: int,
        line_search_decay_rate: float,
) -> Callable:
    '''
    Input:
        x0[nx, ]: init state point
        x[T+1, nx]: state in current iteration
        u[T, nu]: control in current iteration
    '''
    nx, nu, T = x.shape[1], u.shape[1], x.shape[0]-1
    ntau = nx + nu
    
    class LQRStep(Function):
        @staticmethod
        def forward(
            ctx: BackwardCFunction,
            Jf: Tensor, # (T, nx, ntau)
            Jl: Tensor, # (T+1, ntau)
            Hl: Tensor, # (T+1, ntau, ntau)
        ) -> Tuple[Tensor]:
            _x, _u = lqr_step_solve(x0, x, u, Jf, Jl, Hl, dynamic, params, metricfunc, line_search_max_num, line_search_decay_rate)
            ctx.save_for_backward(Jf, Jl, Hl, _x, _u)
            return _x, _u

        @staticmethod
        def backward(
            ctx: BackwardCFunction,
            dl_dx: Tensor, # (T+1, nx)
            dl_du: Tensor, # (T, nu)
        ) -> Tuple[Tensor]:
            Jf, Jl, Hl, x, u = ctx.saved_tensors
            Jf, Jl, Hl, x, u = [cast(Tensor, v) for v in [Jf, Jl, Hl, x, u]]
            tau = torch.zeros([T+1, ntau])
            tau[:, :nx] = x; tau[:T, nx:] = u
            dl_dtau = torch.zeros([T+1, ntau]) # (T+1, ntau)
            dl_dtau[:, :nx] = dl_dx; dl_dtau[:T, nx:] = dl_du
            dx0 = torch.tensor([nx])
            dynamic_back = LinDynamic(nx, nu, T, Jf)
            cost_back = QuadCost(nx, nu, T, Hl, -dl_dtau)
            params_back = None
            dmetricfunc = iLQRMetricFunction(nx, nu, T, cost_back, dynamic_back)
            dx, du = mpc_solve(dx0, nx, nu, T, dynamic_back, cost_back, params_back, dmetricfunc, iter_nums, line_search_max_num)
            dtau = torch.zeros([T+1, ntau])
            dtau[:, :nx] = dx; dtau[:T, nx:] = du
            lams: Tensor  = get_lambdas((x, u), nx, nu, T, Hl, Jl, Jf)
            dlams: Tensor = get_lambdas((dx, du), nx, nu, T, Hl, -dl_dtau, Jf)
            dl_dHl: Tensor = -0.5 * (dtau.unsqueeze(-1) @ tau.unsqueeze(-2) + tau.unsqueeze(-1) @ dtau.unsqueeze(-2)) # (T+1, ntau, ntau)
            dl_dJl: Tensor = -dtau # (T+1, ntau)
            dl_dJf: Tensor = -(dlams[1:].unsqueeze(-1) @ tau[:-1].unsqueeze(-2) + lams[1:].unsqueeze(-1) @ dtau[:-1].unsqueeze(-2)) # (T, nx, ntau)
            return dl_dJf, dl_dJl, dl_dHl
    
    return LQRStep.apply
