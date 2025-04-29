import torch
import torch.nn.functional as F
from torch import nn


class CrossAttention3D(nn.Module):
    def __init__(self, context_dim, voxel_channels, dropout=0.1):
        super().__init__()

        self.query_fc = nn.Linear(context_dim, voxel_channels)
        self.key_fc = nn.Conv3d(voxel_channels, voxel_channels, kernel_size=1)
        self.value_fc = nn.Conv3d(voxel_channels, voxel_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # ✅ Apply LayerNorm to Query, Key, and Value (not the final output)
        self.layer_norm_q = nn.LayerNorm(voxel_channels)
        self.layer_norm_k = nn.LayerNorm(voxel_channels)
        self.layer_norm_v = nn.LayerNorm(voxel_channels)

    def forward(self, voxel_input, context, mask=None):
        bs, lat_dim, res_1, res_2, res_3 = voxel_input.shape

        # Generate query, key, and value, applying LayerNorm before attention
        query = self.query_fc(context).view(bs, lat_dim, 1, 1, 1)
        query = self.layer_norm_q(query.transpose(1, -1)).transpose(
            1, -1
        )  # Normalize query

        key = self.layer_norm_k(self.key_fc(voxel_input).transpose(1, -1)).transpose(
            1, -1
        )  # Normalize query
        value = self.layer_norm_v(
            self.value_fc(voxel_input).transpose(1, -1)
        ).transpose(
            1, -1
        )  # Normalize query

        # Scaled dot-product attention
        attn_scores = (query * key) / (lat_dim**0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_scores = self.softmax(attn_scores)
        attn_scores = self.dropout(attn_scores)

        # Compute attention-weighted output
        attended_voxel = attn_scores * value

        return attended_voxel  # Shape: (batch_size, channels, res, res, res)
