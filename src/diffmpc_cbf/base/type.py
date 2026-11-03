from enum import Enum, unique

@unique
class SQPStatus(Enum):
    INIT: int = 0
    CONVERGED: int = 1
    LINE_SEARCH_FAILED: int = 2
    MAX_ITERATIONS: int = 3
    NEED_RESTORE: int = 4