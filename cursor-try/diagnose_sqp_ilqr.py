import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffmpc_cbf import DiffMPC as SQPMPC
from diffmpc_cbf.lqr import DiffMPC as ILQRMPC
from examples.lc_ex.system import LCost, LcDynamic


def jacobians(outputs, inputs):
    rows = [[] for _ in inputs]
    for output in outputs.reshape(-1):
        grads = torch.autograd.grad(output, inputs, retain_graph=True)
        for row, grad in zip(rows, grads):
            row.append(grad)
    return [torch.stack(row) for row in rows]


def report(name, sqp, ilqr):
    diff = sqp - ilqr
    sqp_norm = torch.linalg.vector_norm(sqp).item()
    ilqr_norm = torch.linalg.vector_norm(ilqr).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    relative = diff_norm / max(sqp_norm, ilqr_norm, 1e-12)
    rmse = torch.mean(diff.square()).sqrt().item()
    print(
        f"{name:9s} diff_norm={diff_norm:.6e} relative={relative:.6e} "
        f"rmse={rmse:.6e} max={diff.abs().max().item():.6e} "
        f"sqp_norm={sqp_norm:.6e} ilqr_norm={ilqr_norm:.6e}"
    )


def solve_dense_lqr(c_matrix, c_vector, dynamics_matrix, x0, nx, nu, horizon):
    ntau = nx + nu
    nvar = horizon * ntau + nx
    neq = (horizon + 1) * nx
    hessian = torch.zeros(nvar, nvar)
    linear = torch.zeros(nvar)
    equality = torch.zeros(neq, nvar)
    rhs = torch.zeros(neq)

    for t in range(horizon):
        block = slice(t * ntau, (t + 1) * ntau)
        hessian[block, block] = c_matrix[t]
        linear[block] = c_vector[t]
    terminal = slice(horizon * ntau, horizon * ntau + nx)
    hessian[terminal, terminal] = c_matrix[horizon, :nx, :nx]
    linear[terminal] = c_vector[horizon, :nx]

    equality[:nx, :nx] = torch.eye(nx)
    rhs[:nx] = x0
    for t in range(horizon):
        row = slice((t + 1) * nx, (t + 2) * nx)
        tau_block = slice(t * ntau, (t + 1) * ntau)
        next_x = slice((t + 1) * ntau, (t + 1) * ntau + nx)
        equality[row, tau_block] = -dynamics_matrix[t]
        equality[row, next_x] = torch.eye(nx)

    zeros = torch.zeros(neq, neq)
    kkt = torch.cat(
        (
            torch.cat((hessian, equality.transpose(0, 1)), dim=1),
            torch.cat((equality, zeros), dim=1),
        ),
        dim=0,
    )
    solution = torch.linalg.solve(kkt, torch.cat((-linear, rhs)))
    primal = solution[:nvar]
    controls = torch.stack(
        [
            primal[t * ntau + nx : (t + 1) * ntau]
            for t in range(horizon)
        ]
    )
    return controls


def run(bound):
    torch.manual_seed(1)
    nx, nu, horizon = 2, 1, 5
    ntau = nx + nu

    raw_c = torch.rand(horizon + 1, ntau, ntau, requires_grad=True)
    c_matrix = raw_c.transpose(-1, -2) @ raw_c
    c_vector = torch.rand(horizon + 1, ntau, requires_grad=True)
    dynamics_matrix = torch.rand(
        horizon, nx, ntau, requires_grad=True
    )

    cost = LCost(nx, nu, horizon, c_matrix, c_vector)
    dynamics = LcDynamic(nx, nu, horizon, dynamics_matrix)
    lower = torch.tensor([-bound])
    upper = torch.tensor([bound])
    x0 = torch.zeros(nx)

    sqp = SQPMPC(nx, nu, horizon, 1, cost, dynamics, lower, upper)
    ilqr = ILQRMPC(nx, nu, horizon, 1, cost, dynamics, lower, upper)

    tau_sqp = sqp.step(x0, {})
    x_ilqr, u_ilqr = ilqr.step(x0, {})
    u_sqp = tau_sqp.u
    u_dense = solve_dense_lqr(
        c_matrix, c_vector, dynamics_matrix, x0, nx, nu, horizon
    )

    inputs = [c_matrix, c_vector, dynamics_matrix]
    jac_sqp = jacobians(u_sqp, inputs)
    jac_ilqr = jacobians(u_ilqr, inputs)
    jac_dense = jacobians(u_dense, inputs)

    print(f"\nbound={bound:g}")
    report("u", u_sqp, u_ilqr)
    report("u sqp/kkt", u_sqp, u_dense)
    report("u lqr/kkt", u_ilqr, u_dense)
    report("dC raw", jac_sqp[0], jac_ilqr[0])
    report(
        "dC sym",
        0.5 * (jac_sqp[0] + jac_sqp[0].transpose(-1, -2)),
        0.5 * (jac_ilqr[0] + jac_ilqr[0].transpose(-1, -2)),
    )
    report("dc", jac_sqp[1], jac_ilqr[1])
    report("dF", jac_sqp[2], jac_ilqr[2])
    report("dc sqp/k", jac_sqp[1], jac_dense[1])
    report("dc lqr/k", jac_ilqr[1], jac_dense[1])
    report("dF sqp/k", jac_sqp[2], jac_dense[2])
    report("dF lqr/k", jac_ilqr[2], jac_dense[2])

    dc_per_output = torch.linalg.vector_norm(
        jac_sqp[1] - jac_ilqr[1], dim=(-2, -1)
    )
    print("dc diff norm per output:", dc_per_output.tolist())


if __name__ == "__main__":
    for test_bound in (1e2, 1e4, 1e6):
        run(test_bound)
