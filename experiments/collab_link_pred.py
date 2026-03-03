"""
Train a GNN on ogbl-collab (link prediction, Hits@50).

Strategy:
  1. Encode all nodes with the GNN to get embeddings
  2. Score edges as dot-product of endpoint embeddings
  3. Evaluate with OGB Hits@K evaluator

Usage:
    python experiments/collab_link_pred.py --model gcn --epochs 50
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from data.ogb.link_prediction import load_collab
from models import MODEL_REGISTRY
from training.losses import get_loss_fn
from utils.seed import set_seed


def encode(model, data, device):
    """Full-graph forward pass — returns node embeddings without the readout head."""
    # For link prediction we need node embeddings, not graph-level logits.
    # We monkey-patch task to 'node' temporarily.
    orig_task = model.task
    model.task = "node"
    model.eval()
    with torch.no_grad():
        emb_data = Data(x=data.x.to(device), edge_index=data.edge_index.to(device))
        emb_data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        h = model(emb_data)
    model.task = orig_task
    return h


def train_epoch(model, data, split_edge, optimizer, loss_fn, device, batch_size):
    model.train()

    pos_edge = split_edge["train"]["edge"].t().to(device)   # (2, E_pos)
    neg_edge = negative_sampling(
        edge_index=data.edge_index.to(device),
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge.size(1),
        method="sparse",
    )

    optimizer.zero_grad()
    # Get node embeddings (task=node)
    train_data = Data(x=data.x.to(device), edge_index=data.edge_index.to(device))
    train_data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    model.task = "node"
    h = model(train_data)
    model.task = "link"

    pos_scores = (h[pos_edge[0]] * h[pos_edge[1]]).sum(dim=-1)
    neg_scores = (h[neg_edge[0]] * h[neg_edge[1]]).sum(dim=-1)

    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_hits(model, data, split_edge, evaluator, device, K=50):
    model.eval()
    emb_data = Data(x=data.x.to(device), edge_index=data.edge_index.to(device))
    emb_data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    model.task = "node"
    h = model(emb_data)
    model.task = "link"

    results = {}
    for split in ("valid", "test"):
        pos_edge = split_edge[split]["edge"].t().to(device)
        neg_edge = split_edge[split]["edge_neg"].t().to(device)
        pos_scores = (h[pos_edge[0]] * h[pos_edge[1]]).sum(dim=-1)
        neg_scores = (h[neg_edge[0]] * h[neg_edge[1]]).sum(dim=-1)
        result = evaluator.eval({
            "y_pred_pos": pos_scores,
            "y_pred_neg": neg_scores,
        })
        results[split] = result[f"hits@{K}"]
    return results


def main():
    parser = argparse.ArgumentParser(description="ogbl-collab link prediction")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "graphsage", "gin"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    data, split_edge, evaluator, num_features = load_collab()

    ModelCls = MODEL_REGISTRY[args.model]
    model_kwargs = dict(
        in_channels=num_features,
        hidden_channels=args.hidden,
        out_channels=args.hidden,  # embed dim = hidden for dot-product scoring
        num_layers=args.layers,
        dropout=args.dropout,
        task="link",
    )
    if args.model == "gat":
        model_kwargs["heads"] = args.heads
    model = ModelCls(**model_kwargs).to(device)
    print(f"collab | {args.model.upper()} | params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_hits = -float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, data, split_edge, optimizer, None, device, batch_size=64 * 1024)
        hits = eval_hits(model, data, split_edge, evaluator, device, K=50)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | val hits@50={hits['valid']:.4f}")

        if hits["valid"] > best_val_hits:
            best_val_hits = hits["valid"]
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    final_hits = eval_hits(model, data, split_edge, evaluator, device, K=50)
    print(f"\nTest Hits@50={final_hits['test']:.4f}")


if __name__ == "__main__":
    main()
