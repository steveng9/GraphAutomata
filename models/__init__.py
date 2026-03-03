"""
Model registry — maps string names to model classes.

Usage:
    from models import MODEL_REGISTRY
    ModelCls = MODEL_REGISTRY["gcn"]
    model = ModelCls(in_channels=3, hidden_channels=64, out_channels=4)
"""

from models.gcn import GCN
from models.gat import GAT
from models.graphsage import GraphSAGE
from models.gin import GIN

MODEL_REGISTRY = {
    "gcn": GCN,
    "gat": GAT,
    "graphsage": GraphSAGE,
    "gin": GIN,
}

__all__ = ["GCN", "GAT", "GraphSAGE", "GIN", "MODEL_REGISTRY"]
