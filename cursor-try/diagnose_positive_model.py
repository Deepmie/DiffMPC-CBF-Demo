"""Diagnose why the SQP QP model can be positive without editing source files."""
from __future__ import annotations

import os
import random
import sys
import types

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.utils import convert_any_to_tensor_batch
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def main() -> None:
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
    pref = torch.tensor([[1.0], [1.0]], device=device, dtype=dtype, requires_grad=True)

    original_solve = mpc._solve_qp_subproblem
    original_line_search = mpc._line_search
    solve_number = 0

    def traced_solve(self, zeta, x0, params):
        nonlocal solve_number
        solve_number += 1
        result = original_solve(zeta, x0, params)
        step, eq_dual, ineq_dual = result

        quadratic = 0.5 * step @ self._Qv @ step
        linear = self._pv @ step
        model = quadratic + linear
        zero_eq_violation = self._bv.norm(float("inf"))
        zero_ineq_violation = (-self._qv).clamp_min(0).norm(float("inf"))
        step_eq_residual = (self._Av @ step - self._bv).norm(float("inf"))
        step_ineq_residual = (self._Gv @ step - self._qv).clamp_min(0).norm(float("inf"))
        stationarity = (
            self._Qv @ step
            + self._pv
            + self._Av.T @ eq_dual
            + self._Gv.T @ ineq_dual
        ).norm(float("inf"))
        eig_min = torch.linalg.eigvalsh(0.5 * (self._Qv + self._Qv.T)).min()

        print(
            f"QP {solve_number}: model={scalar(model):+.12g} "
            f"(quadratic={scalar(quadratic):+.12g}, linear={scalar(linear):+.12g})"
        )
        print(
            f"  zero-step violations: equality={scalar(zero_eq_violation):.6g}, "
            f"inequality={scalar(zero_ineq_violation):.6g}"
        )
        print(
            f"  solved-step residuals: equality={scalar(step_eq_residual):.3e}, "
            f"inequality={scalar(step_ineq_residual):.3e}, "
            f"stationarity={scalar(stationarity):.3e}"
        )
        print(
            f"  step_inf={scalar(step.norm(float('inf'))):.6g}, "
            f"Q_min_eigenvalue={scalar(eig_min):.6g}"
        )
        return result

    def traced_line_search(self, *args, **kwargs):
        accepted, alpha = original_line_search(*args, **kwargs)
        print(f"  line search: accepted={accepted}, alpha={alpha:.12g}")
        return accepted, alpha

    mpc._solve_qp_subproblem = types.MethodType(traced_solve, mpc)
    mpc._line_search = types.MethodType(traced_line_search, mpc)

    env.reset()
    mpc.reset()
    obs, radius, gamma = convert_any_to_tensor_batch(
        [env.get_obs_info(horizon + 1), env.radius_agent, 0.2], device, dtype
    )
    params = {
        "cost": {"pref": pref, "obs_info": obs},
        "barrier": {
            "obs_info": obs,
            "radius_agent": radius,
            "gamma": gamma,
        },
    }
    state = torch.from_numpy(env.state.copy()).requires_grad_(True).to(device, dtype)
    solution, status = mpc(state, params)
    print(f"forward status={status.name}, solution_is_none={solution is None}")


if __name__ == "__main__":
    main()
