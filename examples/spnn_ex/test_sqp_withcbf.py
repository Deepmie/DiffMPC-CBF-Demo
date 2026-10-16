from diffmpc_cbf import DiffMPCBF, Tau
from diffmpc_cbf.utils import plot_trajectory
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from system import SpnnCost, SpnnDynamic, SpnnBarrier
from env import Env
from typing import List

def main():
    # basic environment set
    torch.manual_seed(1)
    nx: int = 2
    nu: int = 1
    nb: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 80
    ntau = nx + nu
    nl = T * nu
    # build mpc-cbf solver
    cost      = SpnnCost(nx, nu, T)
    dynamic   = SpnnDynamic(nx, nu, T, delta_t)
    barrier   = SpnnBarrier(nx, nu, T)

    umin      = torch.tensor([-10.0])
    umax      = torch.tensor([10.0])
    x0 = torch.tensor([[0.], [0.]], requires_grad=True)
    xref = torch.tensor([[1.], [0.]], requires_grad=True)
    obs: List[ndarray] = [np.array([0.5, 0.5, 0.3, 0.02]), ]
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
    
    env = Env(x0.flatten().detach().numpy(), xref.flatten().detach().numpy(), obs, delta_t=delta_t)
    # start
    cp = list(cost.context_network.named_parameters())
    fp = list(dynamic.residual_network.parameters())
    x = x0.clone()
    for t in range(1, N):
        params = {
            'cost': {'xref': xref},
            'barrier': {'obs': obs, 'gamma': gamma, },
        }
        tau = mpc.step(x, params)
        # u = tau.u.view(-1)
        # dl_dcp = [torch.zeros([nl, *list(p.shape)]) for p in cp]
        # dl_dfp = [torch.zeros([nl, *list(p.shape)]) for p in fp]
        # for i in range(nl):
        #     grad_cp = torch.autograd.grad(u[i], cp, retain_graph=True)
        #     grad_fp = torch.autograd.grad(u[i], fp, retain_graph=True)
        #     for j in range(len(dl_dcp)): dl_dcp[j][i] = grad_cp[j]
        #     for j in range(len(dl_dfp)): dl_dfp[j][i] = grad_fp[j]
        x = env.step(tau.u[0].flatten().detach().numpy())
        x = torch.tensor(x)

    env.show()
    

if __name__ == '__main__':
    main()