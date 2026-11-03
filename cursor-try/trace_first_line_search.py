"""Trace merit values for the first tvto MPC call."""
from __future__ import annotations

import os
import random
import sys
import importlib

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.utils import convert_any_to_tensor_batch
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic


class MetricTrace:
    def __init__(self, metric, alphas):
        self.metric = metric
        self.alphas = alphas
        self.sqp = -1
        self.candidate = 0

    def __call__(self, *args, **kwargs):
        requested_residual = kwargs.get("include_res", False)
        kwargs["include_res"] = True
        value, residual = self.metric(*args, **kwargs)
        if requested_residual:
            self.sqp += 1
            self.candidate = 0
            dveq = args[2]
            print(
                f"SQP {self.sqp} current merit={value.item():.9e} "
                f"res={residual.flatten().tolist()} "
                f"eq_dual_inf={dveq.abs().max().item():.6e} "
                f"eq_dual_l1={dveq.abs().sum().item():.6e}"
            )
            return value, residual

        alpha = self.alphas[self.candidate] if self.candidate < len(self.alphas) else None
        print(
            f"SQP {self.sqp} candidate={self.candidate} alpha={alpha} "
            f"merit={value.item():.9e} delta={value.item() - self.current_value:.9e} "
            f"res={residual.flatten().tolist()}"
        )
        self.candidate += 1
        return value

    @property
    def current_value(self):
        # Filled by the wrapper below after each baseline call.
        return self._current_value

    @current_value.setter
    def current_value(self, value):
        self._current_value = value


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)

    nx, nu, nb, horizon = 4, 2, 2, 15
    mpc = DiffMPCBF(
        nx=nx,
        nu=nu,
        nb=nb,
        T=horizon,
        batch_size=1,
        cost=TvToCost(nx, nu, horizon),
        dynamic=TvToDynamic(nx, nu, horizon, 0.05),
        barrier=TvToBarrier(nx, nu, horizon, nb),
        umin=torch.tensor([-1.0, -1.0]),
        umax=torch.tensor([1.0, 1.0]),
        start_dv_eps_detection=True,
    )
    env = Env(np.zeros(2), np.ones(2), 0.05)
    obs, radius, gamma = convert_any_to_tensor_batch(
        [env.get_obs_info(horizon + 1), env.radius_agent, 0.2],
        torch.device("cpu"),
        torch.float32,
    )
    params = {
        "cost": {"pref": torch.ones(2, 1), "obs_info": obs},
        "barrier": {
            "obs_info": obs,
            "radius_agent": radius,
            "gamma": gamma,
        },
    }

    original_metric = mpc._metricfunc
    trace = MetricTrace(
        original_metric,
        [mpc._line_search_alpha_init * mpc._line_search_decay_rate**i
         for i in range(mpc._line_search_max_num)],
    )

    # Wrap once more to retain each current merit for candidate deltas.
    class Wrapper:
        def __call__(self, *args, **kwargs):
            result = trace(*args, **kwargs)
            if kwargs.get("include_res", False):
                trace.current_value = result[0].item()
            return result

    mpc._metricfunc = Wrapper()
    solver_module = importlib.import_module("diffmpc_cbf.diffmpc_cbf")
    original_qp = solver_module.QPFunction
    qp_index = {"value": 0}

    def traced_qp(*args, **kwargs):
        solve = original_qp(*args, **kwargs)

        def apply(*qp_args):
            result = solve(*qp_args)
            delta = result[0].flatten()
            print(
                f"QP {qp_index['value']} tau_delta_inf="
                f"{delta[:mpc._var_dims].abs().max().item():.6e} "
                f"slack_delta_inf={delta[mpc._var_dims:].abs().max().item():.6e}"
            )
            qp_index["value"] += 1
            return result

        return apply

    solver_module.QPFunction = traced_qp
    state = torch.from_numpy(env.state.copy()).float()
    try:
        tau, status = mpc(state, params)
        print("FINAL", status, "u0", tau.u[0].tolist())
    finally:
        solver_module.QPFunction = original_qp


if __name__ == "__main__":
    main()
