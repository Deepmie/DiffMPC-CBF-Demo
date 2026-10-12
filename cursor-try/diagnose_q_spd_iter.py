import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.sp_ex.system import SpCost, SpDynamic
from diffmpc_cbf import DiffMPC
from qpth.qp import QPFunction


def run(params_mode="nested"):
    torch.manual_seed(0)
    nx, nu, T = 2, 1, 5
    cost = SpCost(nx, nu, T)
    dynamic = SpDynamic(nx, nu, T, 0.1)
    mpc = DiffMPC(nx, nu, T, 1, cost, dynamic,
                  torch.tensor([-1000.0]), torch.tensor([1000.0]),
                  iter_nums=10)
    x0 = torch.tensor([[0.0], [0.0]])
    if params_mode == "nested":
        params = {"cost": {"xref": torch.tensor([[1.0], [0.0]])}, "dynamic": {}}
    else:
        # as in examples/sp_ex/main.py currently
        params = {"xref": torch.tensor([[1.0], [0.0]])}

    print("\n=== params_mode:", params_mode, "===")
    try:
        # manually mirror step to catch which iteration fails
        if "cost" not in params:
            params = {**params, "cost": {}, "dynamic": {}}
            # keep xref only at top-level like broken main
        for key in ["cost", "dynamic"]:
            params.setdefault(key, {})

        mpc._tau.set_state_init(x0.flatten())
        dveq = mpc._get_init_dveq()
        for t in range(T):
            mpc._tau.set_state(
                mpc._dynamic.forward(mpc._tau[t], {**params.get("dynamic"), "t": t}).flatten(),
                t + 1,
            )

        for i in range(mpc._iter_nums):
            Q, p, G, h, A, b = mpc._build_sqp_params(x0, mpc._tau, dveq, params)
            eigs = torch.linalg.eigvalsh(Q.detach())
            _, info = torch.linalg.cholesky_ex(Q.detach())
            print(
                f"iter {i}: eigmin={eigs.min().item():.3e}, "
                f"chol_info={info.item()}, "
                f"dveq_norm={dveq.norm().item():.3e}, "
                f"Q_nan={bool(torch.isnan(Q).any())}"
            )
            try:
                tau_delta, dveq_qp, dvneq_qp = QPFunction(
                    eps=1e-12, verbose=0, maxIter=20
                )(Q, p, G, h, A, b)
            except Exception as exc:
                print("  QP failed:", type(exc).__name__, exc)
                # dump Lagrangian contribution
                for t in range(T):
                    Hc = cost.get_stage_cost(
                        mpc._tau[t], {**params.get("cost"), "t": t}, order=2
                    ).detach()
                    Hf = dynamic.forward(
                        mpc._tau[t], {**params.get("dynamic"), "t": t}, order=2
                    ).detach()
                    lam = dveq[(t + 1) * nx : (t + 2) * nx]
                    Hlag = Hc + (lam.reshape(-1, 1, 1) * Hf).sum(0)
                    print(
                        f"  t={t} Hc_eigmin={torch.linalg.eigvalsh(Hc).min().item():.3e} "
                        f"Hlag_eigmin={torch.linalg.eigvalsh(0.5*(Hlag+Hlag.T)).min().item():.3e} "
                        f"lam={lam.tolist()}"
                    )
                return

            tau_delta, dveq_qp, _ = mpc._qp_var_post_process(
                [tau_delta, dveq_qp, dvneq_qp]
            )
            from diffmpc_cbf.base import Tau
            tau_delta = Tau.build_from_tensor(tau_delta, nx, mpc._ntau)
            dveq_delta = dveq_qp - dveq
            alpha = 1.0
            for _ in range(mpc._line_search_max_num):
                if mpc._metricfunc(mpc._tau + alpha * tau_delta, x0, params, dveq_qp) < \
                   mpc._metricfunc(mpc._tau, x0, params, dveq_qp):
                    break
                alpha *= mpc._line_search_decay_rate
            mpc._tau += alpha * tau_delta
            dveq = dveq + alpha * dveq_delta
            print(f"  alpha={alpha:.3e}, u={mpc._tau.u.detach().flatten().tolist()}")

        # final differentiable QP
        Q, p, G, h, A, b = mpc._build_sqp_params(x0, mpc._tau.detach(), dveq.detach(), params)
        eigs = torch.linalg.eigvalsh(Q.detach())
        print("final Q eigmin", eigs.min().item(), "requires_grad", Q.requires_grad)
        QPFunction(eps=1e-12, verbose=0, maxIter=20)(Q, p, G, h, A, b)
        print("final QP OK")
    except Exception as exc:
        print("outer fail:", type(exc).__name__, exc)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run("nested")
    # also try actual DiffMPC.step
    print("\n=== DiffMPC.step ===")
    torch.manual_seed(0)
    nx, nu, T = 2, 1, 5
    mpc = DiffMPC(nx, nu, T, 1, SpCost(nx, nu, T), SpDynamic(nx, nu, T, 0.1),
                  torch.tensor([-1000.0]), torch.tensor([1000.0]))
    try:
        tau = mpc.step(torch.tensor([[0.0], [0.0]]),
                       {"cost": {"xref": torch.tensor([[1.0], [0.0]])}})
        print("step OK, u=", tau.u.detach().flatten())
    except Exception as exc:
        print("step fail:", type(exc).__name__, exc)
