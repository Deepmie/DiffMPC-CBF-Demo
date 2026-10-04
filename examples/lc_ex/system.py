from diffmpc_cbf import QuadCost, LinDynamic
from typing import Dict, Tuple, Union, Optional
import torch
from torch import Tensor

class LCost(QuadCost):
    ...


class LcDynamic(LinDynamic):
    ...