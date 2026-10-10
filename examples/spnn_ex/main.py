from diffmpc_cbf import DiffMPC, Tau
from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC
from diffmpc_cbf.utils import plot_trajectory
import torch
from system import SpCost, SpDynamic

def main():
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    cost    = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, delta_t)
    umin    = torch.tensor([-1000.0])
    umax    = torch.tensor([1000.0])
    mpc     = DiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    # solve one step mpc
    x0 = torch.tensor([[0.], [0.]])
    xs = torch.zeros([N, nx]); xs[0] = x0.flatten()
    for t in range(1, N):
        tau = mpc.step(xs[t-1], {'xref': torch.tensor([[1.], [0.]])})
        xs[t]  = dynamic.forward((xs[t-1], tau.u[0])).flatten()
    plot_trajectory(xs, 'imgs/sp_sqp.png')
 

def main2():
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    cost    = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, delta_t)
    umin    = torch.tensor([-1.0])
    umax    = torch.tensor([1.0])
    mpc     = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    x0 = torch.tensor([[0.], [0.]])
    xs = torch.zeros([N, nx]); xs[0] = x0.flatten()
    for t in range(1, N):
        _, u = mpc.step(xs[t-1], {'cost': {'xref': torch.tensor([[1.], [0.]])}})
        xs[t] = dynamic.forward((xs[t-1], u[0])).flatten()
    plot_trajectory(xs, 'imgs/sp_ilqr.png')


if __name__ == '__main__':
    # main()
    main2()