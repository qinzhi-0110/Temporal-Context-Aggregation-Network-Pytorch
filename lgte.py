from mha import MultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace

class LocalGlobalTemporalEncoder(nn.Module):
    def __init__(self, input_dim, dropout, temporal_scale=100, window_size=9):
        super(LocalGlobalTemporalEncoder, self).__init__()
        dim_feedforward = 256
        self.self_atten = GlobalLocalAttention(input_dim, num_heads=8, dropout=0.1, temporal_scale=temporal_scale, window_size=window_size)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features):
        src = features.permute(2, 0, 1)
        q = k = src
        src2 = self.self_atten(q, k, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.permute(1, 2, 0)
        return src


class GlobalLocalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, temporal_scale, window_size=9):
        super(GlobalLocalAttention, self).__init__()
        self.num_heads = num_heads
        self.temporal_scale = temporal_scale
        self.scale_attention = MultiheadAttention(input_dim,
                                                  num_heads=self.num_heads,
                                                  dropout=dropout)
        self.mask_matrix = nn.Parameter(self._mask_matrix(window_size).float(), requires_grad=False)

    def _mask_matrix(self, window_size):
        m = torch.ones((1, self.num_heads,
                        self.temporal_scale,
                        self.temporal_scale), dtype=torch.bool)
        w_len = window_size
        local_len = self.num_heads // 2
        for i in range(local_len):
            for j in range(self.temporal_scale):
                for k in range(w_len):
                    m[0, i, j, min(max(j - w_len // 2 + k, 0), self.temporal_scale - 1)] = False

        for i in range(local_len, self.num_heads):
            m[0, i] = False
        return m

    def forward(self, query, key, value):
        b = query.size(1)
        mask = self.mask_matrix.bool().repeat(b, 1, 1, 1)
        r, w = self.scale_attention(query, key, value, key_padding_mask=mask)
        return r, w
