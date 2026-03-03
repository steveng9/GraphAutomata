"""
Graph Convolutional Network (Kipf & Welling 2017).

Key details:
  - Symmetric degree normalisation: D^{-1/2} A D^{-1/2}
  - Self-loops added before propagation
  - Isolated nodes (degree 0) handled by masking inf → 0
  - Readout: global_add_pool for graph tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, degree

from models.base import GNNBase


class GCNConv(nn.Module):
    """
    Single GCN layer implementing the propagation rule:
        H' = D̃^{-1/2} Ã D̃^{-1/2} H W
    where Ã = A + I (self-loops already added by caller).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # Compute degree
        row, col = edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0  # isolated nodes

        # Normalise: d_src^{-1/2} * d_dst^{-1/2}
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Message passing: aggregate normalised neighbour features
        x = self.lin(x)  # (N, out)
        out = torch.zeros_like(x)
        msg = x[row] * norm.unsqueeze(-1)  # normalised messages from source
        out.scatter_add_(0, col.unsqueeze(-1).expand_as(msg), msg)

        return out + self.bias


class GCN(GNNBase):
    """
    Multi-layer GCN.

    Architecture:
      GCNConv → BN → ReLU → Dropout  (× num_layers-1)
      GCNConv → [global_add_pool] → Linear
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        task: str = "graph",
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, task)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [hidden_channels]
        for i in range(num_layers):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            x = global_add_pool(x, batch)
            return self.lin(x)
        elif self.task == "node":
            return self.lin(x)
        else:  # link
            # Expects data.edge_label_index for scoring
            src, dst = data.edge_label_index
            return (x[src] * x[dst]).sum(dim=-1, keepdim=True)
