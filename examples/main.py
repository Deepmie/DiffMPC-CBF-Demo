import torch
import torch.nn as nn
from enum import Enum

class LSP(Enum):
    CONVERGENCE: int = 0
    FAILURE: int = 1
    

def test():
    a2 = torch.rand([3, 2])
    print(a2)
    print(a2.reshape(-1))



if __name__ == '__main__':
    print(LSP.CONVERGENCE)