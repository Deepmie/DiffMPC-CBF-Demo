"""Read-only reproduction of examples/tvto/main.py, without animation or log writes."""
import random
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "tvto"))
from diffmpc_cbf import DiffMPCBF
from diffmpc_cbf.base import SQPStatus
from diffmpc_cbf.utils import convert_any_to_tensor_batch, convert_tensor_iteration, tcon
from env import Env
from system import TvToBarrier, TvToCost, TvToDynamic
import importlib


class MetricProbe:
    def __init__(self, original):
        self.original = original
        self.rows = []

    def __call__(self, *args, **kwargs):
        requested_residual = kwargs.get("include_res", False)
        ret = self.original(*args, **{**kwargs, "include_res": True})
        value, residual = ret
        self.rows.append(("base" if kwargs.get("include_res") else "trial",
                          float(value), residual.flatten().tolist(),
                          float(args[1].min()), float(args[1].max())))
        return ret if requested_residual else value


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    nx, nu, nb, T, dt, N = 4, 2, 2, 15, .05, 80
    device = torch.device("cpu")
    dtype = torch.float64 if os.getenv("TVTO_FLOAT64") else torch.float32
    cost, dynamic, barrier = TvToCost(nx, nu, T), TvToDynamic(nx, nu, T, dt), TvToBarrier(nx, nu, T, nb)
    pref = torch.tensor([[1.], [1.]], dtype=dtype, requires_grad=True)
    mpc = DiffMPCBF(nx, nu, nb, T, 1, cost, dynamic, barrier,
                    torch.tensor([-1., -1.], dtype=dtype), torch.tensor([1., 1.], dtype=dtype),
                    dtype=dtype, start_dv_eps_detection=True)
    mpc.to(dtype=dtype)
    env = Env(np.zeros(2), np.ones(2), dt)
    optimizer = torch.optim.Adam(mpc.parameters(), lr=5e-4)
    probe = MetricProbe(mpc._metricfunc)
    mpc._metricfunc = probe
    solver_module = importlib.import_module("diffmpc_cbf.diffmpc_cbf")
    original_qp = solver_module.QPFunction
    qp_rows = []
    def traced_qp(*qa, **qkw):
        if os.getenv("TVTO_QP_MORE_ITERS"):
            qkw = {**qkw, "eps": 1e-8, "maxIter": 50, "notImprovedLim": 10}
        solve = original_qp(*qa, **qkw)
        def apply(Q, p, G, q, A, b):
            ret = solve(Q, p, G, q, A, b)
            d = ret[0].flatten()
            qp_rows.append((float(torch.linalg.eigvalsh(Q).min()), float(p @ d),
                            float((d @ Q @ d)/2), float(d.abs().max()),
                            float(d[:mpc._var_dims].abs().max()),
                            float(d[mpc._var_dims:].abs().max()),
                            float((A@d-b).abs().max()),
                            float((G@d-q).max())))
            return ret
        return apply
    solver_module.QPFunction = traced_qp
    for epoch in range(int(os.getenv("TVTO_EPOCHS", "100"))):
        env.reset()
        mpc.reset()
        xs, us, bs, xr = [], [], [], []
        for k in range(N-1):
            obs, radius, gamma = convert_any_to_tensor_batch(
                [env.get_obs_info(T+1), env.radius_agent, .2], device, dtype)
            params = {"cost": {"pref": pref, "obs_info": obs},
                      "barrier": {"obs_info": obs, "radius_agent": radius, "gamma": gamma}}
            x = torch.from_numpy(env.state.copy()).requires_grad_(True).to(dtype=dtype)
            probe.rows.clear()
            tau, status = mpc(x, params)
            if status == SQPStatus.LINE_SEARCH_FAILED:
                print(f"FAIL epoch={epoch} step={k} state={x.flatten().tolist()}", flush=True)
                print("last QP rows: (eigmin, p_dot_d, quad, dmax, tau_dmax, slack_dmax, eq_res, ineq_res)", qp_rows[-5:], flush=True)
                print("last metric calls:", flush=True)
                for row in probe.rows[-25:]:
                    print(row, flush=True)
                print("slack max", float(mpc._slack.max()), "eq dual max", float(mpc._dveq.abs().max()),
                      "ineq dual max", float(mpc._dvneq.abs().max()), flush=True)
                return
            x_env, done = env.step(tau.u[0].detach().numpy().reshape(-1, 1))
            xs.append(tau.x[1]); us.append(tau.u[0])
            bs.append(barrier.forward(tau.x[1].reshape(-1, 1), tcon(params["barrier"], 1, device, dtype)).flatten())
            xr.append(convert_tensor_iteration(x_env.copy(), device, dtype).flatten())
            if done:
                break
        xs, us, bs, xr = map(torch.stack, (xs, us, bs, xr))
        loss_obj = (xs[:, :2]-pref.reshape(1, -1)).square().mean()
        loss_control = us.square().mean()
        loss_safe = torch.nn.functional.relu(-bs).square().mean()
        loss_dynamic = (xs-xr).square().mean()
        loss = loss_obj + loss_control + 1e4*loss_safe + 1e3*loss_dynamic
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"epoch={epoch} steps={k+1} loss={loss.item():.6g} parts={[float(v) for v in (loss_obj,loss_control,loss_safe,loss_dynamic)]}", flush=True)


if __name__ == "__main__":
    main()
