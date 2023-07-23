import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(10, 2, requires_grad=True)
print(x.softmax(dim=1))

for i in range(10000):
    
    y = F.softmax(x, dim=1)
    loss = (y + 1e-6).log()
    loss = (- 0.5 * loss[...,0].clone() - 0.5 * loss[...,1].clone()).mean()
    loss.backward()
    
    # loss = F.cross_entropy(x, torch.ones(10, dtype=torch.long))
    # loss.backward()

    x = x - x.grad
    x.grad = torch.zeros_like(x)
print(x.softmax(dim=1))