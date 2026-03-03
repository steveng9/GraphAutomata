"""
Graph Isomorphism Network (Xu et al. 2019).

Key details:
  - aggr='add' (non-negotiable — required for WL-test equivalence)
  - MLP with BatchNorm after each GIN layer
  - Multi-layer readout: concatenate global_add_pool(h^k) for k=1..K
    (required by the paper for full expressivity)
  - epsilon learned (eps=True) — initialized to 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool

from models.base import GNNBase


class GINConv(nn.Module):
    """
    Single GIN layer.

    h_v^{k} = MLP^{k}( (1 + ε^{k}) · h_v^{k-1}  +  Σ_{u∈N(v)} h_u^{k-1} )
    """

    def __init__(self, in_channels: int, out_channels: int, train_eps: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
        )
        self.eps = nn.Parameter(torch.zeros(1), requires_grad=train_eps)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        nn.init.zeros_(self.eps)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        row, col = edge_index  # source_to_target: row=src aggregated into col=dst
        agg = torch.zeros_like(x)
        agg.scatter_add_(0, col.unsqueeze(-1).expand(-1, x.size(-1)), x[row])
        out = self.mlp((1.0 + self.eps) * x + agg)
        return out


class GIN(GNNBase):
    """
    Multi-layer GIN with multi-layer readout.

    Architecture:
      GINConv (× num_layers)
      Readout: concat( global_add_pool(h^k) for k=1..K ) → Linear
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
        # First layer maps in_channels → hidden; remaining are hidden → hidden
        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            self.convs.append(GINConv(dims[i], dims[i + 1]))

        # Graph task readout: concatenates num_layers pooled vectors
        self.graph_lin = nn.Linear(hidden_channels * num_layers, out_channels)
        # Node / link task readout: operates on the final node embedding
        self.node_lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        layer_pools = []

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.task == "graph":
                layer_pools.append(global_add_pool(x, batch))

        if self.task == "graph":
            # Concatenate all per-layer pooled representations  (B, hidden*K)
            out = torch.cat(layer_pools, dim=-1)
            return self.graph_lin(out)
        elif self.task == "node":
            return self.node_lin(x)
        else:  # link
            src, dst = data.edge_label_index
            return (x[src] * x[dst]).sum(dim=-1, keepdim=True)
