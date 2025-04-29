# This code has been adapted from https://github.com/y9miao/VLSG/blob/main/src/models/sgaligner/src/aligner/networks/gat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class MultiGCN(nn.Module):
    def __init__(self, n_units=[17, 128, 100], dropout=0.0):
        super(MultiGCN, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            layer_stack.append(
                GCNConv(
                    in_channels=n_units[i], out_channels=n_units[i + 1], cached=False
                )
            )
        self.layer_stack = nn.ModuleList(layer_stack)

    def forward(self, x, edges):
        edges = edges.long()
        for idx, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x=x, edge_index=edges)
            if idx + 1 < self.num_layers:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class MultiGAT(nn.Module):
    def __init__(self, n_units=[17, 128, 100], n_heads=[2, 2], dropout=0.0):
        super(MultiGAT, self).__init__()
        self.num_layers = len(n_units) - 1
        self.dropout = dropout
        layer_stack = []

        # in_channels, out_channels, heads
        for i in range(self.num_layers):
            in_channels = n_units[i] * n_heads[i - 1] if i else n_units[i]
            layer_stack.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=n_units[i + 1],
                    cached=False,
                    heads=n_heads[i],
                )
            )

        self.layer_stack = nn.ModuleList(layer_stack)

    def forward(self, x, edges):

        for idx, gat_layer in enumerate(self.layer_stack):
            x = F.dropout(x, self.dropout, training=self.training)
            x = gat_layer(x, edges)
            if idx + 1 < self.num_layers:
                x = F.elu(x)

        return x
