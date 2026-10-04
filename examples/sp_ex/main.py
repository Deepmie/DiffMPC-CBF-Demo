from diffmpc_cbf.utils import plot_trajectory
from diffmpc_cbf import DiffMPC, Tau
from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC
import torch
from system import SpCost, SpDynamic

def main():
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    cost    = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, delta_t)
    umin    = torch.tensor([-1.0])
    umax    = torch.tensor([1.0])
    mpc     = DiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    # solve one step mpc
    x0      = torch.tensor([[0.], [0.]])
    tau = mpc.step(x0, {'xref': torch.tensor([[1.], [0.]])})
    
    print(tau.x, tau.u)

def main2():
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    cost    = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, delta_t)
    umin    = torch.tensor([-1.0])
    umax    = torch.tensor([1.0])
    mpc     = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    x0      = torch.tensor([[0.], [0.]])
    x, u    = mpc.step(x0, {'xref': torch.tensor([[1.], [0.]])})
    print(x, u)


if __name__ == '__main__':
    main()
    main2()