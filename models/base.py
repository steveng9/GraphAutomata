"""
Abstract base class for all GNN models in this repo.

Subclasses must implement `forward(data)` and return logits shaped for the task:
  - 'graph' : (batch_size, out_channels)
  - 'node'  : (num_nodes,  out_channels)
  - 'link'  : (num_edges,  1)           — dot-product of endpoint embeddings
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch_geometric.data import Data


class GNNBase(nn.Module, ABC):
    """
    Parameters
    ----------
    in_channels     : input node feature dimension
    hidden_channels : width of hidden layers
    out_channels    : number of output classes / 1 for binary
    num_layers      : number of message-passing layers
    dropout         : dropout probability applied after each hidden layer
    task            : 'graph' | 'node' | 'link'
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
        super().__init__()
        assert task in ("graph", "node", "link"), f"Unknown task: {task}"
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        """Return logits for the configured task."""
        ...

    def reset_parameters(self):
        """Re-initialise all learnable parameters (called by subclasses)."""
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()
