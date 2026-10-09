from diffmpc_cbf import DiffMPC, Tau
from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC
from diffmpc_cbf.utils import plot_trajectory
import torch
from torch import Tensor
from system import LCost, LcDynamic

def main():
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
    umin      = torch.tensor([-10.0])
    umax      = torch.tensor([10.0])
    x0 = torch.tensor([[0.], [0.]], requires_grad=True)

    mpc     = DiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)
    mpc_lqr = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)
    
    # solve one step mpc
    x0 = torch.tensor([[0.], [0.]])
    xs = torch.zeros([N, nx]); xs[0] = x0.flatten()
    for t in range(1, N):
        tau = mpc.step(xs[t-1], {'cost': {'xref': torch.tensor([[1.], [0.]])}})
        u = tau.u.view(-1)
        dl_dC = torch.zeros([nl, T+1, ntau, ntau])
        dl_dc = torch.zeros([nl, T+1, ntau])
        dl_dF = torch.zeros([nl, T, nx, ntau])
        for i in range(nl):
            dl_dCi, dl_dci, dl_dFi = torch.autograd.grad(u[i], [C, c, F], retain_graph=True)
            dl_dC[i] = dl_dCi; dl_dc[i] = dl_dci; dl_dF[i] = dl_dFi
        
        _, u_lqr = mpc_lqr.step(xs[t-1], {'cost': {'xref': torch.tensor([[1.], [0.]])}})
        u_lqr = u_lqr.view(-1)
        dl_dC_lqr = torch.zeros([nl, T+1, ntau, ntau])
        dl_dc_lqr = torch.zeros([nl, T+1, ntau])
        dl_dF_lqr = torch.zeros([nl, T, nx, ntau])
        for i in range(nl):
            dl_dCi, dl_dci, dl_dFi = torch.autograd.grad(u_lqr[i], [C, c, F], retain_graph=True)
            dl_dC_lqr[i] = dl_dCi; dl_dc_lqr[i] = dl_dci; dl_dF_lqr[i] = dl_dFi
        

        def mc_func(t1: Tensor, t2: Tensor) -> float:
            N: int = len(t1.view(-1))
            return 1/N * torch.sum((t1-t2)**2).sqrt().item()

        print(f'dl_dC: {mc_func(dl_dC, dl_dC_lqr)}/{nl*(T+1)*ntau*ntau}, dl_dc: {mc_func(dl_dc, dl_dc_lqr)}/{nl*(T+1)*ntau}, dl_dF: {mc_func(dl_dF, dl_dF_lqr)}/{nl*T*ntau}')
        xs[t]  = dynamic.forward((xs[t-1], tau.u[0]), {'t': 0}).flatten()


if __name__ == '__main__':
    main()