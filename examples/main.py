import torch


def main():
    a1 = torch.rand([10, 5])
    a2 = torch.rand([10, 5, 6, 6])

    r = a1.unsqueeze(-1).unsqueeze(-1) * a2
    print(r.shape)

if __name__ == '__main__':
    main()
    