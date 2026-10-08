from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC
from diffmpc_cbf.utils import plot_trajectory
import sys
sys.path.append('.')
from thirdparty.mpc_pytorch.mpc.mpc import MPC, QuadCost, LinDx
import torch
from torch import Tensor
from system import LCost, LcDynamic

def main_ref():
    torch.manual_seed(1)
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    ntau = nx + nu
    C: Tensor = torch.rand([T+1, ntau, ntau], requires_grad=True)[:T]
    C = C.permute(0, 2, 1) @ C
    c: Tensor = torch.rand([T+1, ntau], requires_grad=True)[:T]
    F: Tensor = torch.rand([T, nx, ntau], requires_grad=True)[:T-1]
    cost      = LCost(nx, nu, T, C, c)
    dynamic   = LcDynamic(nx, nu, T, F)
    
    umin      = torch.tensor([-10000.0])
    umax      = torch.tensor([10000.0])
    x0 = torch.tensor([[0.], [0.]], requires_grad=True)

    # Ref MPC.
    def loss_func(u: Tensor) -> Tensor: # (nu, )
        return u[0, 0, 0]

    xs_ref = torch.zeros([N, nx]); xs_ref[0] = x0.flatten()
    for t in range(1, N):
        C = C.unsqueeze(1); c = c.unsqueeze(1); F = F.unsqueeze(1)
        x_ref, u_ref, obj_ref = MPC(
            nx, nu, T, umin.unsqueeze(0).unsqueeze(1).expand(T, 1, nu), umax.unsqueeze(0).unsqueeze(1).expand(T, 1, nu), None,
            lqr_iter=20, exit_unconverged=False
        )(xs_ref[t-1].reshape(1, -1), QuadCost(C, c), LinDx(F, None))
        
        u_ref = u_ref.view(-1)
        for i in range(len(u_ref)):
            dl_dC, dl_dc, dl_dF= torch.autograd.grad(u_ref[i], [C, c, F], retain_graph=True)
            print('dl_dC:')
            print(dl_dC)
            print('dl_dc')
            print(dl_dc)
            print('dl_dF')
            print(dl_dF)
            input('finished...')

        # update
        xs_ref[t] = dynamic.forward((xs_ref[t-1], u_ref[0, 0, :]), {'t': 0}).flatten()


if __name__ == '__main__':
    main_ref()