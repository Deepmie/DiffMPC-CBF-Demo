import torch
import torch.nn as nn

def test():
    a2 = torch.rand([3, 2])
    print(a2)
    print(a2.reshape(-1))



if __name__ == '__main__':
    test()