"""Read-only diagnostics for the latest acceleration changes."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples", "tvto"))

from diffmpc_cbf import DiffMPCBF, Tau
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic
from diffmpc_cbf.utils import convert_any_to_tensor_batch


def test_tau_shift():
    tau = Tau(4, 2, 5)
    for t in range(5):
        tau.set_control(torch.tensor([float(t + 1), -float(t + 1)]), t)
    try:
        tau.set_state_init(torch.zeros(4), need_shift=True)
        print("tau shift: OK", tau.u[:, 0].tolist())
    except Exception as exc:
        print("tau shift: FAILED", type(exc).__name__, str(exc))


def build_mpc(max_iter=2):
    nx, nu, nb, horizon = 4, 2, 2, 5
    cost = TvToCost(nx, nu, horizon)
    dynamic = TvToDynamic(nx, nu, horizon, 0.05)
    barrier = TvToBarrier(nx, nu, horizon, nb)
    mpc = DiffMPCBF(
        nx=nx, nu=nu, nb=nb, T=horizon, batch_size=1,
        cost=cost, dynamic=dynamic, barrier=barrier,
        umin=torch.tensor([-1.0, -1.0]),
        umax=torch.tensor([1.0, 1.0]),
        min_iter_nums=0, max_iter_nums=max_iter,
    )
    env = Env(np.zeros(2), np.ones(2), delta_t=0.05)
    obs, radius, gamma = convert_any_to_tensor_batch(
        [env.get_obs_info(horizon + 1), env.radius_agent, 0.5],
        torch.device("cpu"), torch.float32,
    )
    params = {
        "cost": {"pref": torch.ones(2, 1), "obs_info": obs},
        "barrier": {"obs_info": obs, "radius_agent": radius, "gamma": gamma},
    }
    x = torch.from_numpy(env.state.copy()).float()
    return mpc, env, x, params


def test_first_flag_and_metric():
    mpc, env, x, params = build_mpc(max_iter=1)
    print("is_first before forward:", mpc._is_first)
    tau = mpc(x, params)
    print("is_first after forward:", mpc._is_first)
    residual = mpc._metricfunc(
        tau, x, params, mpc._get_init_dveq(), mpc._get_init_dvneq(),
        only_res=True,
    )
    merit = mpc._metricfunc(
        tau, x, params, mpc._get_init_dveq(), mpc._get_init_dvneq(),
    )
    print("metric only_res:", residual.shape, residual.tolist())
    print("metric default:", type(merit).__name__,
          merit.item() if torch.is_tensor(merit) else merit)


def test_barrier_units():
    _, env, x, params = build_mpc(max_iter=1)
    barrier = TvToBarrier(4, 2, 5, 2)
    h = barrier(x, {**params["barrier"], "t": 0, "t_ts": torch.tensor(0.0)})
    pos, radii = params["barrier"]["obs_info"]
    dp2 = ((x[:2].reshape(1, 2) - pos[:, 0, :]) ** 2).sum(dim=1)
    print("barrier h implementation:", h.flatten().tolist())
    print("distance squared:", dp2.tolist())
    print("base radii:", radii[:, 0].tolist(),
          "agent radius:", params["barrier"]["radius_agent"].item())


if __name__ == "__main__":
    test_tau_shift()
    test_first_flag_and_metric()
    test_barrier_units()
