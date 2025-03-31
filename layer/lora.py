from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA(nn.Module):
    def __init__(self, layer : nn.Linear, rank : int, alpha : int=None) -> None:
        super().__init__()
        self.register_buffer('W', torch.zeros(layer.weight.shape))
        self.W = layer.weight.detach().clone().requires_grad_(False)
        if layer.bias is not None:
            self.register_buffer('b', torch.zeros(layer.bias.shape))
            self.b = layer.bias.detach().clone().requires_grad_(False)
        else:
            self.b = None
        self.rank = rank
        self.dim_in  = layer.weight.shape[1]
        self.dim_out = layer.weight.shape[0]
        self.alpha   = alpha if alpha is not None else rank
        self.lora_a  = nn.Parameter(torch.randn(self.dim_in,  self.rank, requires_grad=True))
        self.lora_b  = nn.Parameter(torch.randn(self.rank, self.dim_out, requires_grad=True))
        self.lora_scale = self.alpha / self.rank
        nn.init.normal_(self.lora_a, 0, 1)
        nn.init.zeros_(self.lora_b)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.lora_a, 0, 1)
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        Wx = x @ self.W.t()
        Ax  = x @ self.lora_a
        BAx = Ax @ self.lora_b
        BAx = BAx * self.lora_scale
        if self.b is not None:
            return Wx + BAx + self.b
        else:
            return Wx + BAx
        
    def __repr__(self):
        return f"LoRA({self.dim_in}, {self.dim_out}, rank={self.rank}, alpha={self.alpha})"
    
    def extra_repr(self):
        return f"rank={self.rank}, alpha={self.alpha}"

class SelectionLoRA(nn.Module):
    def __init__(self, linear_layer:nn.Linear, num_loras, rank:int=4, alpha:float=1.0):
        super(SelectionLoRA, self).__init__()
        self.num_loras = num_loras
        self.alpha = alpha
        self.linear_layer = linear_layer
        self.A = nn.Parameter(torch.randn(self.num_loras, self.linear_layer.in_features, rank))
        self.B = nn.Parameter(torch.randn(self.num_loras, self.linear_layer.out_features, rank))
        self.selection = None
    
    def set_selection(self, selection):
        self.selection = selection
    
    def forward(self, x):
        B, N, D = x.size()
        assert D == self.linear_layer.in_features
        ret = self.linear_layer(x)
        if self.selection is not None:
            assert self.selection.size(0) == B
            Ax  = torch.bmm(x, self.A[self.selection].clone())
            BAx = torch.bmm(Ax, self.B[self.selection].clone().transpose(1, 2))
            BAx = BAx * self.alpha
            ret = ret + BAx
        return ret