import torch

tau = torch.zeros(6, 3).requires_grad_()
weight = torch.ones(3, requires_grad=True)

with torch.no_grad():
    row = tau[0]
print("slice under no_grad requires_grad:", row.requires_grad)

detached_row = tau.detach()[0]
print("slice after detach requires_grad:", detached_row.requires_grad)

for name, ipt in [("no_grad slice", row), ("detached slice", detached_row)]:
    with torch.enable_grad():
        value = (weight * ipt).sum() + weight.sum()
        try:
            torch.autograd.grad(value, [ipt], create_graph=True)
        except RuntimeError as error:
            print(f"{name}: {error}")

with torch.enable_grad():
    ipt = detached_row.detach().requires_grad_()
    value = (weight * ipt.square()).sum() + weight.sum()
    grad = torch.autograd.grad(value, [ipt], create_graph=True)[0]
print("fixed grad:", grad, "depends on weight:", grad.requires_grad)
