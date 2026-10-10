import torch
from torch import Tensor

def f(x: Tensor) -> Tensor:
    x1, x2 = x
    return torch.stack([x1+x2**2, x1, x2])

def test():
    nx: int = 2
    nf: int = 3
    x = torch.rand([nx, ], requires_grad=True)
    v = f(x)
    g, = torch.autograd.grad(v[0], [x, ], create_graph=True)

    print(f'x [{x.shape}]: \n{x}')
    print(f'v [{v.shape}]: \n{v}')
    print(f'g [{g.shape}]: \n{g}')


if __name__ == '__main__':
    test()