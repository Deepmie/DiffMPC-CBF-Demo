import torch
from torch import Tensor
from diffmpc_cbf.base import DerivativesFunctionWrapper

def print_tensor(t: Tensor, n: str):
    print(f'{n} [{t.shape}]:')
    print(t)

def f(x: Tensor, params=None) -> Tensor:
    x1, x2 = x
    return torch.stack([x1+x2**2, x1, x2]).reshape(-1, 1)

def test():
    nx: int = 2
    nf: int = 3
    x = torch.rand([nx, ], requires_grad=True)
    f_ = DerivativesFunctionWrapper(f)
    v = f_(x)
    g = f_.derivatives(x, ord=1)
    h = f_.derivatives(x, ord=2)
    print_tensor(g, 'g')
    print_tensor(h, 'h')


if __name__ == '__main__':
    test()