from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC
from diffmpc_cbf.utils import plot_trajectory
import sys
sys.path.append('.')
from thirdparty.mpc_pytorch.mpc.mpc import MPC, QuadCost, LinDx
import torch
from torch import Tensor
from system import LCost, LcDynamic

def main():
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    ntau = nx + nu
    C: Tensor = torch.rand([T+1, ntau, ntau])
    c: Tensor = torch.rand([T+1, ntau])
    F: Tensor = torch.rand([T, nx, ntau])
    cost      = LCost(nx, nu, T, C, c)
    dynamic   = LcDynamic(nx, nu, T, F)
    
    umin      = torch.tensor([-1.0])
    umax      = torch.tensor([1.0])
    mpc       = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    x0 = torch.tensor([[0.], [0.]])
    xs = torch.zeros([N, nx]); xs[0] = x0.flatten()
    for t in range(1, N):
        _, u = mpc.step(xs[t-1], {'cost': {'xref': torch.tensor([[1.], [0.]])}})
        xs[t]  = dynamic.forward((xs[t-1], u[0]), {'t': 0}).flatten()
    
    plot_trajectory(xs, 'imgs/sp_sqp.png')

def main2():
    torch.manual_seed(0)
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    ntau = nx + nu
    C: Tensor = torch.rand([T+1, ntau, ntau], requires_grad=True)
    c: Tensor = torch.rand([T+1, ntau], requires_grad=True)
    F: Tensor = torch.rand([T, nx, ntau], requires_grad=True)
    cost      = LCost(nx, nu, T, C, c)
    dynamic   = LcDynamic(nx, nu, T, F)
    
    umin      = torch.tensor([-10000.0])
    umax      = torch.tensor([10000.0])
    x0 = torch.tensor([[0.], [0.]], requires_grad=True)

    # My MPC.
    mpc = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)
    x0 = torch.tensor([[0.], [0.]])
    xs = torch.zeros([N, nx]); xs[0] = x0.flatten()
    for t in range(1, N):
        _, u = mpc.step(xs[t-1], {'cost': {'xref': torch.tensor([[1.], [0.]])}})
        xs[t]  = dynamic.forward((xs[t-1], u[0]), {'t': 0}).flatten()
    

    # def loss_func(u: Tensor) -> Tensor: # (nu, )
    #     return u[0]

    # loss = loss_func(u)
    # dl_dC, = torch.autograd.grad(loss, (C, ), retain_graph=True)
    # print(dl_dC)
    print('ours')
    print(xs)


if __name__ == '__main__':
    main2()