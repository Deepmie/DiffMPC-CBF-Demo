from .sqp import SQP
from enum import Enum

class SolverType(Enum):
    SQP  = 31


def get_solver(solver_type: SolverType):
    if solver_type == SolverType.SQP:
        return SQP
    else:
        raise ValueError(f'type named {solver_type} not support in current.')