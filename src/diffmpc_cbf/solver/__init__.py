from .ilqr import iLQR
from .mpc import MPC
from .sqp import SQP
from enum import Enum

class SolverType(Enum):
    MPC  = 1
    ILQR = 2
    SQP  = 3


def get_solver(solver_type: SolverType):
    if solver_type == SolverType.MPC:
        return MPC
    elif solver_type == SolverType.ILQR:
        return iLQR
    elif solver_type == SolverType.SQP:
        return SQP
    else:
        raise ValueError(f'type named {solver_type} not support in current.')