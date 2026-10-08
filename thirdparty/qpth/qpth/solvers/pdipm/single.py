"""Single-problem compatibility wrappers around the batched PDIPM solver.

The historical implementation used removed PyTorch APIs such as ``potrf``,
``potrs`` and ``gesv``.  Keeping one linear-algebra implementation reduces
maintenance and makes this module work on current PyTorch releases.
"""

import torch

from . import batch as _batch


def _as_batched_matrix(x):
    return x.unsqueeze(0) if x is not None and x.ndim == 2 else x


def _as_batched_vector(x):
    return x.unsqueeze(0) if x is not None and x.ndim == 1 else x


def _empty_eq_like(Q, nz):
    return Q.new_empty((0, nz))


def pre_factor_kkt(Q, G, A):
    Qb = _as_batched_matrix(Q)
    Gb = _as_batched_matrix(G)
    Ab = _as_batched_matrix(A) if A is not None and A.numel() else _empty_eq_like(Q, Q.size(-1))
    return _batch.pre_factor_kkt(Qb, Gb, Ab)


def factor_kkt(S_LU, R, d):
    db = _as_batched_vector(d)
    return _batch.factor_kkt(S_LU, R, db)


def solve_kkt(U_Q, d, G, A, U_S, rx, rs, rz, ry, dbg=False):
    del dbg
    Gb = _as_batched_matrix(G)
    Ab = _as_batched_matrix(A) if A is not None and A.numel() else G.new_empty((0, G.size(-1)))
    out = _batch.solve_kkt(
        U_Q,
        _as_batched_vector(d),
        Gb,
        Ab,
        U_S,
        _as_batched_vector(rx),
        _as_batched_vector(rs),
        _as_batched_vector(rz),
        _as_batched_vector(ry) if ry is not None else None,
    )
    return tuple(x.squeeze(0) if x is not None else None for x in out)


def factor_solve_kkt(Q, D, G, A, rx, rs, rz, ry):
    Qb = _as_batched_matrix(Q)
    Db = _as_batched_matrix(D)
    Gb = _as_batched_matrix(G)
    Ab = _as_batched_matrix(A) if A is not None and A.numel() else G.new_empty((0, G.size(-1)))
    out = _batch.factor_solve_kkt(
        Qb,
        Db,
        Gb,
        Ab,
        _as_batched_vector(rx),
        _as_batched_vector(rs),
        _as_batched_vector(rz),
        _as_batched_vector(ry) if ry is not None else None,
    )
    return tuple(x.squeeze(0) if x is not None else None for x in out)


def get_step(v, dv):
    return _batch.get_step(_as_batched_vector(v), _as_batched_vector(dv)).squeeze(0)


def forward(inputs_i, Q, G, A, b, h, U_Q, U_S, R, verbose=False):
    """Solve one QP using the same implementation as the batched backend.

    The return signature matches the historical single solver: ``x, y, z``.
    """
    Qb = _as_batched_matrix(Q)
    pb = _as_batched_vector(inputs_i)
    Gb = _as_batched_matrix(G)
    hb = _as_batched_vector(h)
    if A is not None and A.numel():
        Ab = _as_batched_matrix(A)
        bb = _as_batched_vector(b)
    else:
        Ab = Q.new_empty((0, Q.size(-1)))
        bb = Q.new_empty((0,))

    x, y, z, _ = _batch.forward(
        Qb,
        pb,
        Gb,
        hb,
        Ab,
        bb,
        U_Q,
        U_S,
        R,
        verbose=1 if verbose else 0,
    )
    return x.squeeze(0), (y.squeeze(0) if y is not None else None), z.squeeze(0)
