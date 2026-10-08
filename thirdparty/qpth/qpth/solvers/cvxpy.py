"""Optional CVXPY reference backend."""

import numpy as np


def forward_single_np(Q, p, G, h, A, b):
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ImportError(
            "The CVXPY backend is optional. Install it with `pip install 'qpth[cvxpy]'`."
        ) from exc

    nz = p.shape[0]
    neq = A.shape[0] if A is not None else 0
    nineq = G.shape[0] if G is not None else 0

    z_ = cp.Variable(nz)
    obj = cp.Minimize(0.5 * cp.quad_form(z_, Q) + p.T @ z_)
    eq_con = A @ z_ == b if neq > 0 else None

    if nineq > 0:
        slack_var = cp.Variable(nineq)
        ineq_con = G @ z_ + slack_var == h
        slack_con = slack_var >= 0
    else:
        ineq_con = slack_var = slack_con = None

    constraints = [x for x in (eq_con, ineq_con, slack_con) if x is not None]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if prob.status is None or "optimal" not in prob.status:
        raise RuntimeError(f"CVXPY failed to solve the QP (status={prob.status!r}).")

    zhat = np.asarray(z_.value).ravel()
    nu = np.asarray(eq_con.dual_value).ravel() if eq_con is not None else None
    if ineq_con is not None:
        lam = np.asarray(ineq_con.dual_value).ravel()
        slacks = np.asarray(slack_var.value).ravel()
    else:
        lam = slacks = None

    return prob.value, zhat, nu, lam, slacks
