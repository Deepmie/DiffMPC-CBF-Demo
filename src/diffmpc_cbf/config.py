from dataclasses import dataclass
from .solver import SolverType

@dataclass
class MPConfig:
    nx: int
    nu: int
    T: int
    solver: SolverType
    
    @property
    def ntau(self) -> int:
        return self.nx + self.nu