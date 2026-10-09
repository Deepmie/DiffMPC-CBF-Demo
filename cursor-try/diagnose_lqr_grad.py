import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffmpc_cbf.lqr import DiffMPC as OursMPC
from examples.lc_ex.system import LCost, LcDynamic
from thirdparty.mpc_pytorch.mpc.mpc import LinDx, MPC, QuadCost


def jacobians(outputs, inputs):
    rows = [[] for _ in inputs]
    flat_outputs = outputs.reshape(-1)
    for i in range(flat_outputs.numel()):
        grads = torch.autograd.grad(
            flat_outputs[i], inputs, retain_graph=True, allow_unused=False
        )
        for row, grad in zip(rows, grads):
            row.append(grad)
    return [torch.stack(row) for row in rows]


def max_abs(a, b):
    return (a - b).abs().max().item()


def report_close(name, ours, ref):
    error = (ours - ref).abs()
    print(
        f"{name}: max_error={error.max().item():.3e}, "
        f"default={torch.allclose(ours, ref)}, "
        f"atol_1e-6={torch.allclose(ours, ref, atol=1e-6, rtol=1e-5)}"
    )


def run_case(zero_terminal):
    torch.manual_seed(1)
    nx, nu, horizon = 2, 1, 5
    ntau = nx + nu

    raw_c = torch.rand(horizon + 1, ntau, ntau)
    c_full = (raw_c.transpose(-1, -2) @ raw_c).requires_grad_()
    linear_full = torch.rand(horizon + 1, ntau, requires_grad=True)
    dynamics_full = torch.rand(horizon, nx, ntau, requires_grad=True)

    if zero_terminal:
        c_used = torch.cat((c_full[:horizon], torch.zeros_like(c_full[-1:])))
        linear_used = torch.cat(
            (linear_full[:horizon], torch.zeros_like(linear_full[-1:]))
        )
    else:
        c_used = c_full
        linear_used = linear_full

    cost = LCost(nx, nu, horizon, c_used, linear_used)
    dynamics = LcDynamic(nx, nu, horizon, dynamics_full)
    lower = torch.tensor([-10000.0])
    upper = torch.tensor([10000.0])
    x0 = torch.zeros(nx)

    ours = OursMPC(nx, nu, horizon, 1, cost, dynamics, lower, upper)
    _, u_ours = ours.step(x0)

    c_ref = c_used[:horizon].unsqueeze(1)
    linear_ref = linear_used[:horizon].unsqueeze(1)
    dynamics_ref = dynamics_full[: horizon - 1].unsqueeze(1)
    _, u_ref, _ = MPC(
        nx,
        nu,
        horizon,
        lower.view(1, 1, nu).expand(horizon, 1, nu),
        upper.view(1, 1, nu).expand(horizon, 1, nu),
        None,
        lqr_iter=20,
        exit_unconverged=False,
    )(
        x0.unsqueeze(0),
        QuadCost(c_ref, linear_ref),
        LinDx(dynamics_ref, None),
    )

    ours_inputs = [c_used, linear_used, dynamics_full]
    ref_inputs = [c_ref, linear_ref, dynamics_ref]
    ours_jac = jacobians(u_ours, ours_inputs)
    ref_jac = jacobians(u_ref, ref_inputs)

    d_c_ours = ours_jac[0][:, :horizon]
    d_c_ref = ref_jac[0].squeeze(2)
    d_linear_ours = ours_jac[1][:, :horizon]
    d_linear_ref = ref_jac[1].squeeze(2)
    d_dynamics_ours = ours_jac[2][:, : horizon - 1]
    d_dynamics_ref = ref_jac[2].squeeze(2)
    d_c_ours_symmetric = 0.5 * (
        d_c_ours + d_c_ours.transpose(-1, -2)
    )

    print(f"\nzero_terminal={zero_terminal}")
    print("u ours:", u_ours.reshape(-1).detach())
    print("u ref :", u_ref.reshape(-1).detach())
    print("max |u ours-ref|:", max_abs(u_ours.reshape(-1), u_ref.reshape(-1)))
    report_close("dC", d_c_ours, d_c_ref)
    report_close("sym(dC)", d_c_ours_symmetric, d_c_ref)
    report_close("dc", d_linear_ours, d_linear_ref)
    report_close("dF", d_dynamics_ours, d_dynamics_ref)


if __name__ == "__main__":
    run_case(zero_terminal=False)
    run_case(zero_terminal=True)
