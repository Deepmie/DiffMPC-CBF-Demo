"""Trace the source of negative curvature in the current tvto SQP Hessian."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.utils import convert_any_to_tensor_batch
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


def spectrum(matrix: torch.Tensor) -> tuple[float, float]:
    matrix = matrix.detach().double()
    eig = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return eig.min().item(), eig.max().item()


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    nx, nu, nb, horizon = 4, 2, 2, 15
    device, dtype = torch.device("cpu"), torch.float32
    cost = TvToCost(nx, nu, horizon)
    dynamic = TvToDynamic(nx, nu, horizon, 0.05)
    barrier = TvToBarrier(nx, nu, horizon, nb)
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=cost,
        dynamic=dynamic,
        barrier=barrier,
        umin=torch.tensor([-1.0, -1.0]),
        umax=torch.tensor([1.0, 1.0]),
        start_dv_eps_detection=True,
    )
    env = Env(np.zeros(2), np.ones(2), 0.05)
    pref = torch.ones(2, 1)

    context = {"step": -1, "sqp_call": 0}
    records: list[dict] = []
    original_get = mpc._get_sqp_matrix
    original_transfer = mpc._transfer_spd_matrix

    def traced_get(tau, slack, dveq, dvneq, x0, params):
        context["sqp_call"] += 1
        result = original_get(tau, slack, dveq, dvneq, x0, params)
        Q = result[0]
        raw_min, raw_max = spectrum(Q)
        context["raw_min"] = raw_min
        context["raw_max"] = raw_max

        if raw_min < -1.0:
            zero_eq = torch.zeros_like(dveq)
            zero_ineq = torch.zeros_like(dvneq)
            h_cost = mpc._get_two_order_term(tau, zero_eq, zero_ineq, params)
            h_cost_dyn = mpc._get_two_order_term(tau, dveq, zero_ineq, params)
            h_cost_bar = mpc._get_two_order_term(tau, zero_eq, dvneq, params)
            h_dynamic = h_cost_dyn - h_cost
            h_barrier = h_cost_bar - h_cost
            context["parts"] = {
                "cost": spectrum(h_cost),
                "dynamic": spectrum(h_dynamic),
                "barrier": spectrum(h_barrier),
                "total": spectrum(Q[: mpc._var_dims, : mpc._var_dims]),
                "dveq_absmax": dveq.abs().max().item(),
                "dvneq_absmax": dvneq.abs().max().item(),
                "slack_absmax": slack.abs().max().item(),
            }
        else:
            context["parts"] = None
        return result

    def traced_transfer(Q, *args, **kwargs):
        Q_spd, beta = original_transfer(Q, *args, **kwargs)
        reg_min, reg_max = spectrum(Q_spd)
        record = {
            "step": context["step"],
            "sqp_call": context["sqp_call"],
            "raw_min": context.get("raw_min"),
            "raw_max": context.get("raw_max"),
            "beta": beta,
            "reg_min": reg_min,
            "reg_max": reg_max,
            "parts": context.get("parts"),
        }
        records.append(record)
        if beta >= 1.0 or record["raw_min"] < -1.0:
            print(record, flush=True)
        return Q_spd, beta

    mpc._get_sqp_matrix = traced_get
    mpc._transfer_spd_matrix = traced_transfer

    statuses = {}
    for step in range(20):
        context["step"] = step
        obs, radius, gamma = convert_any_to_tensor_batch(
            [env.get_obs_info(horizon + 1), env.radius_agent, 0.2],
            device,
            dtype,
        )
        params = {
            "cost": {"pref": pref, "obs_info": obs},
            "barrier": {
                "obs_info": obs,
                "radius_agent": radius,
                "gamma": gamma,
            },
        }
        state = torch.from_numpy(env.state.copy()).float()
        tau, status = mpc(state, params)
        statuses[status.name] = statuses.get(status.name, 0) + 1
        _, done = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
        if done:
            break

    worst = min(records, key=lambda item: item["raw_min"])
    regularized = [record for record in records if record["beta"] > 0]
    print(
        {
            "statuses": statuses,
            "qp_matrices": len(records),
            "regularized_count": len(regularized),
            "max_beta": max((record["beta"] for record in records), default=0.0),
            "worst": worst,
        }
    )


if __name__ == "__main__":
    main()
