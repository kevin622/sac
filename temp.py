import torch

a = torch.tensor([[1., 0., 1.],[1., 0., 1.],[1., 0., 1.]])
print(a * -1 + 1)