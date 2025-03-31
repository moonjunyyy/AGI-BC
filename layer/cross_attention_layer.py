import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_query, d_kv, nhead, dropout=0.5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_query, nhead, batch_first=True, kdim=d_kv, vdim=d_kv)
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_query)
        self.ffn_1 = nn.Linear(d_query, d_query * 4)
        self.ffn_2 = nn.Linear(d_query * 4, d_query)
        self.norm_2 = nn.LayerNorm(d_query)

    def forward(self, x, y):
        x = self.norm_1(x)
        # y = self.norm_1(y)

        x = self.dropout(x)
        # y = self.dropout(y)
        x2, _ = self.self_attn(x, y, y)
        x = x + x2

        x = self.norm_2(x)
        x2 = self.ffn_1(self.dropout(x))
        x2 = F.gelu(x2)
        x2 = self.ffn_2(self.dropout(x2))
        x = x + x2
        return x