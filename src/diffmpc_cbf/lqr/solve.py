import torch
from torch import Tensor
from .. import Cost, Dynamic, iLQRMetricFunction
from typing import Dict, Tuple, Optional

def lqr_step_solve(
        x0: Tensor,
        x: Tensor,
        u: Tensor,
        Jf: Tensor,
        Jl: Tensor,
        Hl: Tensor,
        dynamic: Dynamic,
        params: Optional[Dict[str, Tensor]],
        metricfunc: iLQRMetricFunction,
        line_search_max_num: int=10,
        line_search_decay_rate: float=0.5,
    ) -> Tuple[Tensor]:
    '''
    Input:
        x0[nx, ]: init state point
        x[T+1, nx]: state in current iteration
        u[T, nu]: control in current iteration
        Jf[T, nx, ntau]: Jacobian matrix for dynamic
        Jl[T+1, ntau]: Jacobian vector for cost function
        Hl[T+1, ntau, ntau]: Hessian matrix for cost function
    '''
    if params is None: params = {}
    for key in ['cost', 'dynamic']:
        if key not in params:
            params[key] = dict()
    alpha: float = 1.0
    T, nx, ntau = Jf.shape
    nu = ntau - nx
    ks = torch.zeros([T, nu])
    Ks = torch.zeros([T, nu, nx])
    
    # consider terminal cost
    Vx: Tensor   = Jl[-1, :nx].unsqueeze(-1) # (nx, 1)
    Vxx: Tensor  = Hl[-1, :nx, :nx] # (nx, nx)
    # not consider
    # Vx: Tensor = torch.zeros([nx, 1])
    # Vxx: Tensor = torch.zeros([nx, nx])
    for t in range(T-1, -1, -1):
        Jft = Jf[t, :, :] # (nx, ntau)
        Qtau = Jl[t, :].unsqueeze(-1) + Jft.T @ Vx # (ntau, 1)
        Qtautau = Hl[t, :, :] + Jft.T @ Vxx @ Jft # (ntau, ntau)
        Qx, Qu = Qtau[:nx, :], Qtau[nx:, :]
        Qxx, Qxu, Qux, Quu = Qtautau[:nx, :nx], Qtautau[:nx, nx:], Qtautau[nx:, :nx], Qtautau[nx:, nx:]
        k: Tensor = torch.linalg.solve(Quu, -Qu) # (nu, 1)
        K: Tensor = torch.linalg.solve(Quu, -Qux) # (nu, nx)
        Vx = Qx + K.T @ Quu @ k + K.T @ Qu + Qxu @ k # (nx, 1)
        Vxx = Qxx + K.T @ Quu @ K + K.T @ Qux + Qxu @ K # (nx, nx)

        ks[t, :] = k.flatten()
        Ks[t, :, :] = K

    _x = torch.zeros_like(x); _x[0] = x0.flatten()
    _u = torch.zeros_like(u)
    for j in range(line_search_max_num):
        for t in range(T):
            _u[t, :] = u[t, :] + alpha*ks[t, :] + (Ks[t, :, :] @ (_x[t, :] - x[t, :]).reshape(-1, 1)).flatten()
            _x[t+1, :] = dynamic.forward((_x[t, :], _u[t, :]), {**params.get('dynamic'), 't': t}).flatten()

        if metricfunc(_x, _u, params) < metricfunc(x, u, params):
            x = _x.clone(); u = _u.clone()
            break
        alpha *= line_search_decay_rate
    return x, u


def build_ilqr_params(
        x: Tensor,
        u: Tensor,
        dynamic: Dynamic,
        cost: Cost,
        params: Optional[Dict[str, Tensor]],
    ) -> Tuple[Tensor]:
    '''
    Input:
        x[T+1, nx]: state in current iteration
        u[T, nu]: control in current iteration
    '''
    Jf = dynamic.jacobian((x, u))
    Jl = cost.jacobian((x, u), params.get('cost'))
    Hl = cost.hessian((x, u), params.get('cost'))
    return Jf, Jl, Hl


def mpc_solve(
        x0: Tensor,
        nx: int,
        nu: int,
        T: int,
        dynamic: Dynamic,
        cost: Cost,
        params: Optional[Dict[str, Tensor]],
        metricfunc: iLQRMetricFunction,
        iter_nums: int=10,
        line_search_max_num: int=10,
        line_search_decay_rate: float=0.5,
    ) -> Tuple[Tensor]:
    if params is None: params = {}
    for key in ['cost', 'dynamic']:
        if key not in params:
            params[key] = dict()
    with torch.no_grad():
        x: Tensor = torch.zeros([T+1, nx])
        u: Tensor = torch.zeros([T, nu])
        x[0] = x0.flatten()
        for t in range(T):
            x[t+1] = dynamic.forward((x[t], u[t]), {**params.get('dynamic'), 't': t}).flatten()

        for i in range(iter_nums):
            Jf, Jl, Hl = build_ilqr_params(x, u, dynamic, cost, params)
            x, u = lqr_step_solve(x0, x, u, Jf, Jl, Hl, dynamic, params, metricfunc, line_search_max_num, line_search_decay_rate)
    
    x_star, u_star = x.detach(), u.detach()
    Jf, Jl, Hl = build_ilqr_params(x_star, u_star, dynamic, cost, params)
    x_out, u_out = lqr_step_solve(x0, x_star, u_star, Jf, Jl, Hl, dynamic, params, metricfunc, line_search_max_num, line_search_decay_rate)
    return x_out, u_out