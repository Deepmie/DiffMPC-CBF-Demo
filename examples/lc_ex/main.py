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
    torch.manual_seed(1)
    nx: int = 2
    nu: int = 1
    T: int  = 5
    delta_t: float = 0.1
    N: int  = 50
    ntau = nx + nu
    nl = T * nu
    C: Tensor = torch.rand([T+1, ntau, ntau], requires_grad=True)
    C: Tensor = C.permute(0, 2, 1) @ C
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
        u = u.view(-1)
        dl_dC = torch.zeros([nl, T+1, ntau, ntau])
        dl_dc = torch.zeros([nl, T+1, ntau])
        dl_dF = torch.zeros([nl, T, nx, ntau])
        for i in range(nl):
            dl_dCi, dl_dci, dl_dFi = torch.autograd.grad(u[i], [C, c, F], retain_graph=True)
            dl_dC[i] = dl_dCi; dl_dc[i] = dl_dci; dl_dF[i] = dl_dFi
        
        C_ref = C[:T].unsqueeze(1); c_ref = c[:T].unsqueeze(1); F_ref = F[:T-1].unsqueeze(1)
        x_ref, u_ref, obj_ref = MPC(
            nx, nu, T, umin.unsqueeze(0).unsqueeze(1).expand(T, 1, nu), umax.unsqueeze(0).unsqueeze(1).expand(T, 1, nu), None,
            lqr_iter=20, exit_unconverged=False,
        )(xs[t-1].unsqueeze(0), QuadCost(C_ref, c_ref), LinDx(F_ref, None))
        u_ref = u_ref.view(-1)
        dl_dC_ref = torch.zeros([nl, T, ntau, ntau])
        dl_dc_ref = torch.zeros([nl, T, ntau])
        dl_dF_ref = torch.zeros([nl, T-1, nx, ntau])
        for i in range(nl):
            dl_dCi, dl_dci, dl_dFi = torch.autograd.grad(u_ref[i], [C_ref, c_ref, F_ref], retain_graph=True)
            dl_dC_ref[i] = dl_dCi.squeeze(1); dl_dc_ref[i] = dl_dci.squeeze(1); dl_dF_ref[i] = dl_dFi.squeeze(1)


        print(torch.allclose(dl_dC[:, :T], dl_dC_ref))
        print(torch.allclose(dl_dc[:, :T], dl_dc_ref))
        print(torch.allclose(dl_dF[:, :T-1], dl_dF_ref))
        input('finished...')
        xs[t]  = dynamic.forward((xs[t-1], u[0]), {'t': 0}).flatten()
    
    print('ours')
    print(xs)


if __name__ == '__main__':
    main2()