from diffmpc_cbf import DiffMPCBF, Tau
from diffmpc_cbf.utils import plot_trajectory
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from system import TvToCost, TvToDynamic, TvToBarrier
from env import Env
from typing import List

def main():
    # basic environment set
    torch.manual_seed(1)
    nx: int = 4
    nu: int = 2
    nb: int = 2
    T: int  = 5
    delta_t: float = 0.05
    N: int  = 80
    ntau = nx + nu
    nl = T * nu
    # build mpc-cbf solver
    cost      = TvToCost(nx, nu, T)
    dynamic   = TvToDynamic(nx, nu, T, delta_t)
    barrier   = TvToBarrier(nx, nu, T, nb)

    umin      = torch.tensor([-1.0])
    umax      = torch.tensor([1.0])
    p0        = torch.tensor([[0.], [0.]], requires_grad=True)
    pref      = torch.tensor([[1.], [1.]], requires_grad=True)
    gamma: float = 0.5

    mpc = DiffMPCBF(
        nx = nx,
        nu = nu,
        nb = nb,
        T = T,
        batch_size = 1,
        cost = cost,
        dynamic = dynamic,
        barrier = barrier,
        umin = umin,
        umax = umax,
    )
    
    env = Env(
        p0.flatten().detach().numpy(),
        pref.flatten().detach().numpy(),
        delta_t=delta_t,
    )

    # start solve
    for t in range(1, N):
        _obs_info = env.get_obs_info(T)
        params = {
            'cost': {'pref': pref, 'obs_info': _obs_info},
            'barrier': {'obs_info': _obs_info, 'gamma': gamma},
        }
        x = torch.from_numpy(env.state).requires_grad_(True)
        tau = mpc.step(x, params)
        x = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
    

if __name__ == '__main__':
    main()