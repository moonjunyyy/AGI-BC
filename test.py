import torch

x = torch.randn(4, 4, 3)
y = torch.randn(4, 4, 3)

masked_x = torch.randn(4, 4).argsort(dim=-1)
visiable_x = masked_x[:, 2:]
masked_x = masked_x[:, :2]

x = x[torch.arange(4).unsqueeze(1), masked_x]

print(x)

x = torch.zeros(4, 4, 3).scatter(1, visiable_x.unsqueeze(-1).expand(*x.shape), x)
print(x)