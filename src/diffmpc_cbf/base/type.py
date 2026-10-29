from enum import Enum, unique

@unique
class SQPStatus(Enum):
    CONVERGED: int = 0
    LINE_SEARCH_FAILED: int = 1
    MAX_ITERATIONS: int = 2