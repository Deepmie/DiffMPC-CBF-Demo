from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC
from diffmpc_cbf.utils import plot_trajectory
import sys
sys.path.append('.')
from thirdparty.mpc_pytorch.mpc.mpc import MPC, QuadCost, LinDx
import torch
from torch import Tensor
from system import LCost, LcDynamic

def main_ref():
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

    # Ref MPC.
    def loss_func(u: Tensor) -> Tensor: # (nu, )
        return u[0, 0, 0]

    mpc_ref   = MPC(nx, nu, T+1, umin.unsqueeze(0).unsqueeze(1).expand(T+1, 1, nu), umax.unsqueeze(0).unsqueeze(1).expand(T+1, 1, nu), None, None, exit_unconverged=False)
    xs_ref = torch.zeros([N, nx]); xs_ref[0] = x0.flatten()
    for t in range(1, N):
        C = C.unsqueeze(1); c = c.unsqueeze(1); F = F.unsqueeze(1)
        _, u_ref, _ = mpc_ref(xs_ref[t-1].reshape(1, -1), QuadCost(C, c), LinDx(F, None))
        l = loss_func(u_ref)
        dl_dC, = torch.autograd.grad(l, [C, ], retain_graph=True)
        print(dl_dC)
        input('finished...')

        # update
        xs_ref[t] = dynamic.forward((xs_ref[t-1], u_ref[0, 0, :]), {'t': 0}).flatten()


if __name__ == '__main__':
    main_ref()