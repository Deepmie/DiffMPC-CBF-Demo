from diffmpc_cbf import DiffMPCBF, Tau
from diffmpc_cbf.utils import plot_trajectory
import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from system import SpnnCost, SpnnDynamic
from typing import List

def main():
    # basic environment set
    torch.manual_seed(1)
    nx: int = 2
    nu: int = 1
    nb: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    ntau = nx + nu
    nl = T * nu
    # build mpc-cbf solver
    cost      = SpnnCost(nx, nu, T)
    dynamic   = SpnnDynamic(nx, nu, T, delta_t)
    obs: List[ndarray] = [np.array([0.5, 0.5, 0.3, 0.02]), ]

    umin      = torch.tensor([-10.0])
    umax      = torch.tensor([10.0])
    x0 = torch.tensor([[0.], [0.]], requires_grad=True)
    xref = torch.tensor([[1.], [0.]], requires_grad=True)
    mpc = DiffMPCBF(
        nx,
        nu,

    )

    # start
    x0 = torch.tensor([[0.], [0.]])
    xs = torch.zeros([N, nx]); xs[0] = x0.flatten()

    cp = list(cost.context_network.named_parameters())
    fp = list(dynamic.residual_network.parameters())

    for t in range(1, N):
        tau = mpc.step(xs[t-1], {'cost': {'xref': xref}})
        u = tau.u.view(-1)

        dl_dcp = [torch.zeros([nl, *list(p.shape)]) for p in cp]
        dl_dfp = [torch.zeros([nl, *list(p.shape)]) for p in fp]
        for i in range(nl):
            grad_cp = torch.autograd.grad(u[i], cp, retain_graph=True)
            grad_fp = torch.autograd.grad(u[i], fp, retain_graph=True)
            for j in range(len(dl_dcp)): dl_dcp[j][i] = grad_cp[j]
            for j in range(len(dl_dfp)): dl_dfp[j][i] = grad_fp[j]
        
        input('finished')
        xs[t]  = dynamic.forward((xs[t-1], tau.u[0]), {'t': 0}).flatten()

    # plot_trajectory(xs.detach().numpy())

if __name__ == '__main__':
    main()