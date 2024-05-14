import torch
import torch.nn as nn
import torch.nn.functional as F

def LoRA_implant(model, rank:int=4, alpha:float=1.0):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            setattr(model, name, LoRA(layer, rank=rank, alpha=alpha))
    return model

class LoRA(nn.Module):
    def __init__(self, linear_layer:nn.Linear, rank:int=4, alpha:float=1.0):
        super(LoRA, self).__init__()
        self.alpha = alpha
        self.linear_layer = linear_layer
        self.A = nn.Parameter(torch.randn(self.linear_layer.in_features, rank))
        self.B = nn.Parameter(torch.randn(self.linear_layer.out_features, rank))

    def forward(self, x):
        Ax  = torch.mm(x, self.A)
        BAx = torch.mm(Ax, self.B.t())
        BAx = BAx * self.alpha
        return self.linear_layer(x) + BAx