from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA(nn.Module):
    def __init__(self, layer : nn.Linear, rank : int, alpha : int=None) -> None:
        super().__init__()

        self.register_buffer('W', torch.zeros(layer.weight.shape))
        self.register_buffer('b', torch.zeros(layer.bias.shape))
        self.W = layer.weight.detach().clone()
        self.b = layer.bias.detach().clone()

        self.rank = rank
        self.dim_in = layer.weight.shape[1]
        self.dim_out = layer.weight.shape[0]
        self.alpha = alpha if alpha is not None else rank

        self.lora_a = nn.Parameter(torch.randn(self.dim_in, self.rank, requires_grad=True))
        self.lora_b = nn.Parameter(torch.randn(self.rank, self.dim_out, requires_grad=True))
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
        BAx = x * self.lora_scale

        return Wx + BAx + self.b