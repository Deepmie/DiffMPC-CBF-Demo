from enum import Enum, unique
from typing import Dict, List
import torch
from torch import Tensor

@unique
class SQPStatus(Enum):
    INIT: int = 0
    CONVERGED: int = 1
    LINE_SEARCH_FAILED: int = 2
    MAX_ITERATIONS: int = 3
    NEED_RESTORE: int = 4
    NON_DESCENT: int = 5
    FAILED: int = 6


