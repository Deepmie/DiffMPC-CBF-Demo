"""Trace the multiplier storage used by the NLP stationarity check."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.utils import convert_any_to_tensor_batch
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


def inf_norm(value: torch.Tensor) -> float:
    return float(value.detach().norm(torch.inf).cpu())


def nlp_stationarity(
    mpc: DiffMPCBF, dveq: torch.Tensor, dvneq: torch.Tensor
) -> float:
    residual = mpc._pv + mpc._Av.T @ dveq.flatten() + mpc._Gv.T @ dvneq.flatten()
    return inf_norm(residual)


def main() -> None:
    update_mode = os.getenv("MULTIPLIER_UPDATE", "assign")
    if update_mode not in {"assign", "copy"}:
        raise ValueError("MULTIPLIER_UPDATE must be 'assign' or 'copy'")
    print(f"multiplier_update_mode={update_mode}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    dtype = torch.float64
    nx, nu, nb, horizon = 4, 2, 2, 15
    delta_t = 0.05

    cost = TvToCost(nx, nu, horizon).to(device, dtype)
    dynamic = TvToDynamic(nx, nu, horizon, delta_t).to(device, dtype)
    barrier = TvToBarrier(nx, nu, horizon, nb).to(device, dtype)
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=cost,
        dynamic=dynamic,
        barrier=barrier,
        umin=torch.tensor([-1.0, -1.0], device=device, dtype=dtype),
        umax=torch.tensor([1.0, 1.0], device=device, dtype=dtype),
        dtype=dtype,
    ).to(device)

    env = Env(np.zeros(2), np.ones(2), delta_t=delta_t)
    pref = torch.tensor([[1.0], [1.0]], device=device, dtype=dtype)
    obs, radius, gamma = convert_any_to_tensor_batch(
        [env.get_obs_info(horizon + 1), env.radius_agent, 0.2], device, dtype
    )
    params = {
        "cost": {"pref": pref, "obs_info": obs},
        "dynamic": {},
        "barrier": {
            "obs_info": obs,
            "radius_agent": radius,
            "gamma": gamma,
        },
    }
    state = torch.from_numpy(env.state.copy()).to(device, dtype)

    with torch.no_grad():
        mpc.reset()
        mpc._reconstruct(state, params)
        print(
            "before iteration:",
            f"dveq_view_shares_storage={mpc._dveqv.data_ptr() == mpc._dveq.data_ptr()}",
            f"dvneq_view_shares_storage={mpc._dvneqv.data_ptr() == mpc._dvneq.data_ptr()}",
        )

        accepted_step = None
        for iteration in range(mpc._max_iter_nums):
            step, dveq_qp, dvneq_qp = mpc._solve_qp_subproblem(
                mpc._zeta, state, params
            )
            step_wide = F.pad(step, (0, mpc._nb), value=0).view(
                mpc._T + 1, mpc._nv
            )

            source_stat = nlp_stationarity(mpc, mpc._dveqv, mpc._dvneqv)
            rebound_stat = nlp_stationarity(mpc, mpc._dveq, mpc._dvneq)
            qp_dual_stat = nlp_stationarity(mpc, dveq_qp, dvneq_qp)
            qp_stat = inf_norm(
                mpc._Qv @ step
                + mpc._pv
                + mpc._Av.T @ dveq_qp
                + mpc._Gv.T @ dvneq_qp
            )
            pf_eq = inf_norm(mpc._bv)
            pf_ineq = inf_norm((-mpc._qv).clamp_min(0))
            source_dual = inf_norm((-mpc._dvneqv).clamp_min(0))
            source_comp = inf_norm(mpc._dvneqv * mpc._qv)

            print(
                f"iter={iteration:02d} step={inf_norm(step):.3e} "
                f"accepted_previous_step={accepted_step if accepted_step is not None else float('nan'):.3e} "
                f"pf=({pf_eq:.3e},{pf_ineq:.3e})"
            )
            print(
                f"  stationarity: source_views={source_stat:.3e} "
                f"current_attributes={rebound_stat:.3e} "
                f"new_qp_duals={qp_dual_stat:.3e} qp_kkt={qp_stat:.3e}"
            )
            print(
                "  storage:",
                f"dveq_view/current={mpc._dveqv.data_ptr() == mpc._dveq.data_ptr()}",
                f"dvneq_view/current={mpc._dvneqv.data_ptr() == mpc._dvneq.data_ptr()}",
                f"view_dual_norms=({inf_norm(mpc._dveqv):.3e},{inf_norm(mpc._dvneqv):.3e})",
                f"current_dual_norms=({inf_norm(mpc._dveq):.3e},{inf_norm(mpc._dvneq):.3e})",
            )

            converged = (
                pf_eq < 1e-4
                and pf_ineq < 1e-4
                and source_stat < 1e-4
                and source_dual < 1e-4
                and source_comp < 1e-4
                and (accepted_step is None or accepted_step < 1e-3)
            )
            if converged:
                print("stopped: source convergence conditions are all satisfied")
                break

            metric_dir = mpc._metricfunc_direct(step, dveq_qp, dvneq_qp)
            if metric_dir >= 0:
                print(f"stopped: non-descent metric derivative {float(metric_dir):.3e}")
                break

            accepted, alpha = mpc._line_search(
                step_wide, dveq_qp, dvneq_qp, state, params
            )
            if not accepted:
                print("stopped: line search failed")
                break

            accepted_step = alpha * inf_norm(step)
            mpc._zetav += alpha * step

            if update_mode == "assign":
                # Reproduce the assignments in DiffMPCBF.forward.
                mpc._dveq = dveq_qp
                mpc._dvneq = dvneq_qp
            else:
                # Preserve the storage referenced by _dveqv and _dvneqv.
                mpc._dveq.copy_(dveq_qp.view_as(mpc._dveq))
                mpc._dvneq.copy_(dvneq_qp.view_as(mpc._dvneq))


if __name__ == "__main__":
    main()
