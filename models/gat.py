"""
Graph Attention Network (Veličković et al. 2018).

Key details:
  - Multi-head edge-softmax attention
  - Intermediate layers: concat=True  →  out_channels_per_head * heads features
  - Final layer:         concat=False →  mean over heads  →  hidden_channels
  - LeakyReLU with negative_slope=0.2 (paper default)
  - Readout: global_add_pool for graph tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, softmax

from models.base import GNNBase


class GATConv(nn.Module):
    """
    Single GAT layer.

    For each head h:
      e_{ij}^h = LeakyReLU( a^h · [ W^h h_i || W^h h_j ] )
      α_{ij}^h = softmax_j( e_{ij}^h )
      h_i^{h'} = σ( Σ_j α_{ij}^h · W^h h_j )

    If concat=True  → output dim = heads * out_channels_per_head
    If concat=False → output dim = out_channels_per_head  (mean of heads)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels_per_head: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.out_channels_per_head = out_channels_per_head
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Linear projection W^h (shared across all nodes, separate per head)
        self.lin = nn.Linear(in_channels, heads * out_channels_per_head, bias=False)
        # Attention parameters a = [a_src || a_dst], shape (1, heads, 2*out)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels_per_head))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels_per_head))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels_per_head if concat else out_channels_per_head))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight.view(1, -1, self.lin.in_features))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.size(0)
        H, C = self.heads, self.out_channels_per_head

        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index  # row=src, col=dst

        # Linear projection → (N, H, C)
        x_proj = self.lin(x).view(-1, H, C)

        # Attention scores: e_ij = LeakyReLU(a_src·h_i + a_dst·h_j)
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)  # (N, H)

        # Per-edge score (source contrib + dest contrib)
        e = alpha_src[row] + alpha_dst[col]              # (E, H)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # Softmax normalisation per destination node
        alpha = softmax(e, col, num_nodes=num_nodes)     # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate: weighted sum of source node projections
        msg = x_proj[row] * alpha.unsqueeze(-1)          # (E, H, C)
        out = torch.zeros(num_nodes, H, C, device=x.device)
        out.scatter_add_(0, col.view(-1, 1, 1).expand_as(msg), msg)  # (N, H, C)

        if self.concat:
            out = out.view(num_nodes, H * C)
        else:
            out = out.mean(dim=1)                        # (N, C)

        return out + self.bias


class GAT(GNNBase):
    """
    Multi-layer GAT.

    Architecture:
      GATConv(concat=True)  → BN → ELU → Dropout  (× num_layers-1)
      GATConv(concat=False) → BN → [global_add_pool] → Linear
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        heads: int = 4,
        task: str = "graph",
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, task)

        self.heads = heads
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # out_channels_per_head for intermediate layers
        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )
        head_dim = hidden_channels // heads

        for i in range(num_layers):
            is_last = i == num_layers - 1
            in_ch = in_channels if i == 0 else hidden_channels  # intermediate input=H*head_dim
            conv = GATConv(
                in_channels=in_ch,
                out_channels_per_head=head_dim,
                heads=heads,
                concat=not is_last,
                dropout=dropout,
            )
            bn_dim = hidden_channels if not is_last else head_dim
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(bn_dim))

        self.lin = nn.Linear(head_dim, out_channels)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.task == "graph":
            x = global_add_pool(x, batch)
            return self.lin(x)
        elif self.task == "node":
            return self.lin(x)
        else:  # link
            src, dst = data.edge_label_index
            return (x[src] * x[dst]).sum(dim=-1, keepdim=True)
