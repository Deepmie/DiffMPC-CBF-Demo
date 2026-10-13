"""Verify examples/spnn_ex/test_sqp_costgrad.py without modifying project sources."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "spnn_ex"))

from system import SpnnCost, SpnnDynamic  # noqa: E402
from diffmpc_cbf import DiffMPC  # noqa: E402
from diffmpc_cbf.lqr import DiffMPC as iLQRDiffMPC  # noqa: E402


def run_user_style_grad(max_steps: int = 3):
    """Mirror test_sqp_costgrad.py but without interactive input()."""
    torch.manual_seed(1)
    nx, nu, T, N = 2, 1, 5, 50
    delta_t = 0.1
    nl = T * nu

    cost = SpnnCost(nx, nu, T)
    dynamic = SpnnDynamic(nx, nu, T, delta_t)
    umin = torch.tensor([-10.0])
    umax = torch.tensor([10.0])
    xref = torch.tensor([[1.0], [0.0]], requires_grad=True)

    mpc = DiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)
    _ = iLQRDiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    xs = torch.zeros([N, nx])
    xs[0] = torch.tensor([0.0, 0.0])
    netparams = list(cost.context_network.parameters())

    print("=== user-style autograd.grad(u, netparams)[0] ===")
    print("netparam shapes:", [tuple(p.shape) for p in netparams])

    for t in range(1, max_steps + 1):
        tau = mpc.step(xs[t - 1], {"cost": {"xref": xref}})
        u = tau.u.view(-1)
        print(f"\nstep {t}: u={u.detach().tolist()}")
        print("u.requires_grad=", u.requires_grad, "grad_fn=", u.grad_fn)

        grads0 = []
        for i in range(nl):
            g = torch.autograd.grad(
                u[i], netparams, retain_graph=True, allow_unused=True
            )
            grads0.append(g[0])
            if i == 0:
                print(
                    "grad tuple lens/shapes:",
                    len(g),
                    [None if x is None else tuple(x.shape) for x in g],
                )
                print(
                    "grad[0] absmean/absmax:",
                    None
                    if g[0] is None
                    else (g[0].abs().mean().item(), g[0].abs().max().item()),
                )
                for j, gj in enumerate(g):
                    if gj is None:
                        print(f"  param[{j}] grad is None")
                    else:
                        print(
                            f"  param[{j}] absmean={gj.abs().mean().item():.3e} "
                            f"absmax={gj.abs().max().item():.3e}"
                        )

        # Keep trajectory update like the user script.
        xs[t] = dynamic.forward((xs[t - 1], tau.u[0]), {"t": 0}).flatten().detach()


def finite_diff_check(eps: float = 1e-3):
    """Central FD on first Linear weight vs autograd from one MPC step."""
    torch.manual_seed(1)
    nx, nu, T = 2, 1, 5
    cost = SpnnCost(nx, nu, T)
    dynamic = SpnnDynamic(nx, nu, T, 0.1)
    mpc = DiffMPC(
        nx, nu, T, 1, cost, dynamic, torch.tensor([-10.0]), torch.tensor([10.0])
    )
    xref = torch.tensor([[1.0], [0.0]], requires_grad=True)
    x0 = torch.tensor([0.0, 0.0])

    w0 = cost.context_network._net[0].weight  # [8, 3]
    assert w0.shape == (8, 3)

    tau = mpc.step(x0, {"cost": {"xref": xref}})
    u = tau.u.view(-1)
    loss = u.square().sum()
    auto = torch.autograd.grad(loss, w0, allow_unused=True)[0]
    print("\n=== finite-difference check on first Linear.weight ===")
    print("loss=", float(loss.detach()))
    if auto is None:
        print("autograd dloss/dw0 is None  -> graph to net params is broken")
        return

    print(
        "autograd absmean/absmax:",
        auto.abs().mean().item(),
        auto.abs().max().item(),
    )

    # FD with fresh solvers for each perturbation to avoid warm-start bias as much as possible.
    fd = torch.zeros_like(w0)
    base = w0.detach().clone()
    for i in range(w0.shape[0]):
        for j in range(w0.shape[1]):
            wp = base.clone()
            wm = base.clone()
            wp[i, j] += eps
            wm[i, j] -= eps

            def eval_loss(weight_tensor):
                torch.manual_seed(1)
                c = SpnnCost(nx, nu, T)
                # copy all params from original trained-init net, then overwrite w0
                with torch.no_grad():
                    for p_new, p_old in zip(
                        c.context_network.parameters(),
                        cost.context_network.parameters(),
                    ):
                        p_new.copy_(p_old)
                    c.context_network._net[0].weight.copy_(weight_tensor)
                d = SpnnDynamic(nx, nu, T, 0.1)
                m = DiffMPC(
                    nx,
                    nu,
                    T,
                    1,
                    c,
                    d,
                    torch.tensor([-10.0]),
                    torch.tensor([10.0]),
                )
                uu = m.step(x0, {"cost": {"xref": xref.detach()}}).u.view(-1)
                return uu.square().sum().detach()

            fp = eval_loss(wp)
            fm = eval_loss(wm)
            fd[i, j] = (fp - fm) / (2 * eps)

    diff = auto.detach() - fd
    rel = diff.norm() / max(auto.detach().norm().item(), fd.norm().item(), 1e-12)
    print("fd absmean/absmax:", fd.abs().mean().item(), fd.abs().max().item())
    print("diff absmean/absmax:", diff.abs().mean().item(), diff.abs().max().item())
    print("relative fro-error:", rel.item() if torch.is_tensor(rel) else rel)
    print("allclose atol=1e-2?", torch.allclose(auto.detach(), fd, atol=1e-2, rtol=1e-2))


def check_main_param_dict():
    print("\n=== main.py params dict sanity ===")
    print(
        "main() passes {'xref': ...} but DiffMPC.step expects "
        "params['cost']['xref']; SpnnCost reads params.get('xref') "
        "from the cost-params dict only."
    )


def check_cost_hessian_psd_sample():
    torch.manual_seed(1)
    cost = SpnnCost(2, 1, 5)
    xref = torch.tensor([[1.0], [0.0]])
    z = torch.tensor([[0.0], [0.0], [0.0]])
    H = cost.get_stage_cost(z, {"xref": xref, "t": 0}, order=2)
    eigs = torch.linalg.eigvalsh(H.detach())
    print("\n=== stage Hessian at 0 with new weighted cost ===")
    print("H=\n", H.detach())
    print("eigs=", eigs.tolist())


if __name__ == "__main__":
    check_cost_hessian_psd_sample()
    check_main_param_dict()
    run_user_style_grad(max_steps=2)
    finite_diff_check(eps=1e-3)
