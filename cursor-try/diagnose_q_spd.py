import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.sp_ex.system import SpCost, SpDynamic
from diffmpc_cbf import DiffMPC
from qpth.qp import QPFunction


def inspect_first_qp():
    torch.manual_seed(0)
    nx, nu, T = 2, 1, 5
    delta_t = 0.1
    cost = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, delta_t)
    umin = torch.tensor([-1000.0])
    umax = torch.tensor([1000.0])
    mpc = DiffMPC(nx, nu, T, 1, cost, dynamic, umin, umax)

    x0 = torch.tensor([[0.0], [0.0]])
    params = {
        "cost": {"xref": torch.tensor([[1.0], [0.0]])},
        "dynamic": {},
    }

    mpc._tau.set_state_init(x0.flatten())
    dveq = mpc._get_init_dveq()
    for t in range(T):
        nxt = mpc._dynamic.forward(
            mpc._tau[t], {**params["dynamic"], "t": t}
        )
        print("forward shape:", tuple(nxt.shape), "value:", nxt.detach().flatten())
        mpc._tau.set_state(nxt.flatten(), t + 1)

    Q, p, G, h, A, b = mpc._build_sqp_params(x0, mpc._tau, dveq, params)
    print("Q shape:", tuple(Q.shape))
    print("Q has nan:", bool(torch.isnan(Q).any()), "inf:", bool(torch.isinf(Q).any()))
    print("Q symmetric max err:", (Q - Q.T).abs().max().item())
    eigs = torch.linalg.eigvalsh(Q)
    print("eig min/max:", eigs.min().item(), eigs.max().item())
    print("diag of Q:", Q.diag())

    for t in range(T):
        Ht = cost.get_stage_cost(
            mpc._tau[t], {**params["cost"], "t": t}, order=2
        )
        print(f"stage H[{t}] eigmin={torch.linalg.eigvalsh(Ht).min().item():.6e}")
        print(Ht)

    Ht = cost.get_terminal_cost(
        mpc._tau.get_state(T), {**params["cost"], "t": T}, order=2
    )
    print("terminal H eigmin:", torch.linalg.eigvalsh(Ht).min().item())
    print(Ht)

    for t in range(min(2, T)):
        Hf = dynamic.forward(
            mpc._tau[t], {**params["dynamic"], "t": t}, order=2
        )
        print(f"dyn H[{t}] shape:", tuple(Hf.shape))

    _, info = torch.linalg.cholesky_ex(Q)
    print("cholesky info:", info.item())

    try:
        QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, h, A, b)
        print("QPFunction OK")
    except Exception as exc:
        print("QPFunction failed:", type(exc).__name__, exc)


if __name__ == "__main__":
    inspect_first_qp()
