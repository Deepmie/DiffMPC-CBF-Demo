import torch
import torch.nn as nn
from enum import Enum

class LSP(Enum):
    CONVERGENCE: int = 0
    FAILURE: int = 1
    

def test():
    a2 = torch.rand([3, 2])
    print(a2)

    amin = 0.5 * torch.rand(2)
    amax = 4 * amin
    a2 = a2.clamp(amin, amax)

    print(amin, amax)
    print(a2)


if __name__ == '__main__':
    print(LSP.CONVERGENCE)
    test()