import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, dimension, attn_dropout=0.1):
        super().__init__()
        self.dimension = dimension
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.dimension, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, seq_len=7, bias=False, drop=0.5) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.drop = drop
        self.n_heads = n_heads
        self.q_liner = nn.Linear(d_model, d_k * n_heads, bias=bias)
        self.k_liner = nn.Linear(d_model, d_k * n_heads, bias=bias)
        self.v_liner = nn.Linear(d_model, d_v * n_heads, bias=bias)
        self.attention = ScaledDotProductAttention(dimension=d_k ** 0.5, attn_dropout=drop)
        self.liner = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.add_norm1 = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model, bias=bias),
            nn.Dropout(p=drop),
            nn.LayerNorm(d_model, eps=1e-6)
        )
        self.add_norm2 = nn.Sequential(
            nn.Linear(d_model, 2 * d_k, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(2 * d_k, d_model),
            nn.LayerNorm(d_model, eps=1e-6)
        )
        self.auto_encoder = nn.Sequential(
            nn.Linear(d_model * seq_len, d_k),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(d_k, d_model * seq_len)
        )

    def forward(self, x, mask=None):
        residual = x
        # 处理掩码
        if mask is not None:
            x = self.auto_encoder(x.flatten(1))
            scores = F.softmax(x, dim=-1)
            values = scores.masked_fill(mask.flatten(0) == 0, -1e9).reshape(residual.shape[0], residual.shape[1], residual.shape[2])
            x = (residual * values)
        batch_size, seq_len, d_model = x.shape
        q = self.q_liner(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.k_liner(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.v_liner(x).view(batch_size, seq_len, self.n_heads, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # TODO 修改mask
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v)
        # q, attn = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        q = self.add_norm1(q) + residual
        q = self.add_norm2(q) + q
        return q, attn
