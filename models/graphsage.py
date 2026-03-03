"""
GraphSAGE (Hamilton et al. 2017) — mean aggregation variant.

Key details:
  - Mean aggregation over neighbours
  - Separate linear for self (lin_self) and neighbours (lin_neigh)
  - NO self-loops added — lin_self handles the self-connection explicitly
  - L2-normalise embeddings after each layer (standard SAGE practice)
  - Readout: global_add_pool for graph tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree

from models.base import GNNBase


class SAGEConv(nn.Module):
    """
    Single GraphSAGE layer (mean aggregation).

    h_v = σ( lin_self(h_v) + lin_neigh( mean_{u∈N(v)} h_u ) )
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)
        row, col = edge_index  # row=src, col=dst (source_to_target convention)

        # Mean aggregation: accumulate and divide by degree
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(-1)), x[row])

        deg = degree(col, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1)
        agg = agg / deg.unsqueeze(-1)

        out = self.lin_self(x) + self.lin_neigh(agg) + self.bias
        return out


class GraphSAGE(GNNBase):
    """
    Multi-layer GraphSAGE.

    Architecture:
      SAGEConv → BN → ReLU → L2-norm → Dropout  (× num_layers-1)
      SAGEConv → [global_add_pool] → Linear
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

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.normalize(x, p=2, dim=-1)  # L2 normalise (SAGE practice)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            x = global_add_pool(x, batch)
            return self.lin(x)
        elif self.task == "node":
            return self.lin(x)
        else:  # link
            src, dst = data.edge_label_index
            return (x[src] * x[dst]).sum(dim=-1, keepdim=True)
