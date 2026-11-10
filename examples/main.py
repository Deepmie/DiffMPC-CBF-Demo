import torch
import torch.nn as nn
from enum import Enum
from torch import Tensor
from torch.func import jacrev, vmap

def func(x: Tensor) -> Tensor:
    # x: (3, 2)
    return torch.concat([
        x, # (3, 2)
        x**2, # (3, 2)
    ], dim=-1) # (3, 4)

def test():
    x = torch.rand([3, 2])
    v = func(x)
    J = jacrev(func)(x)

    print(v.shape)
    print(J.shape)

if __name__ == '__main__':
    test()