import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProduct(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, scale, attn_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.scale, k.transpose(3, 4))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Headed Attention """
    def __init__(self, n_heads: int, d_feature: int, dropout=0.1):
        super().__init__()
        assert d_feature % n_heads == 0
        dim = d_feature // n_heads
        self.dim = dim
        self.n_heads = n_heads

        self.w_qs = nn.Linear(d_feature, n_heads * dim, bias=True)
        self.w_ks = nn.Linear(d_feature, n_heads * dim, bias=True)
        self.w_vs = nn.Linear(d_feature, n_heads * dim, bias=True)
        self.fc = nn.Linear(n_heads * dim, d_feature, bias=True) 

        self.attention = ScaledDotProduct(scale = dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_feature, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k = self.dim
        d_v = self.dim
        n_heads = self.n_heads

        n_batches = q.size(0)
        n_sublines = q.size(1)
        n_words_q = q.size(2)
        n_words_k = k.size(2)
        n_words_v = v.size(2)

        residual = q

        q = self.w_qs(q).view(n_batches, n_sublines, n_words_q, n_heads, d_k)
        k = self.w_ks(k).view(n_batches, n_sublines, n_words_k, n_heads, d_k)
        v = self.w_vs(v).view(n_batches, n_sublines, n_words_v, n_heads, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)

        if mask is not None:
            mask = mask.unsqueeze(2)   # For head axis broadcasting.
            
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(2,3).contiguous().view(n_batches, n_sublines, n_words_q, -1)
        q = self.dropout(self.fc(q))
        
        q += residual
        q = self.layer_norm(q)

        return q, attn

class FeedForward(nn.Module):
    """ Feed Forward layer """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)   # d_in: 256, d_in: 1024
        self.w_2 = nn.Linear(d_hid, d_in)   # d_in: 256, d_in: 1024
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x