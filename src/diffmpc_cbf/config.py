from dataclasses import dataclass
from .solver import SolverType
from . import Cost, Dynamic

@dataclass
class MPConfig:
    nx: int
    nu: int
    T: int
    solver: SolverType
    
    @property
    def ntau(self) -> int:
        return self.nx + self.nu


@dataclass
class SQPConfig:
    nx: int
    nu: int
    T: int
    batch_size: int
    line_search_max_num: int
    line_search_decay: float
    u_min: float
    u_max: float
    cost: Cost
    dynamic: Dynamic
    verbose: int

    @property
    def ntau(self) -> int:
        return self.nx + self.nu