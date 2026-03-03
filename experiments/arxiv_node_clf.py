"""
Train a GNN on ogbn-arxiv (node classification, 40 classes).

Usage:
    python experiments/arxiv_node_clf.py --model gcn --epochs 50
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator

from data.ogb.node_classification import load_arxiv
from models import MODEL_REGISTRY
from training.losses import get_loss_fn
from training.evaluator import ogb_metric_name
from utils.seed import set_seed


class ArxivEvaluator:
    """Wraps OGB node evaluator for Trainer compatibility."""

    def __init__(self):
        self._eval = Evaluator(name="ogbn-arxiv")

    def eval(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        if isinstance(y_pred, torch.Tensor):
            pred_cls = y_pred.argmax(dim=-1, keepdim=True)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1)
        return self._eval.eval({"y_true": y_true, "y_pred": pred_cls})


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = total_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        # NeighborLoader seeds batch with input nodes first
        logits = logits[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1)
        loss = loss_fn(logits, y.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.batch_size
        total_nodes += batch.batch_size
    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def eval_epoch(model, loader, evaluator, device):
    model.eval()
    y_true_list, y_pred_list = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1)
        y_true_list.append(y.cpu())
        y_pred_list.append(logits.cpu())
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})


def main():
    parser = argparse.ArgumentParser(description="ogbn-arxiv node classification")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "graphsage", "gin"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, num_features, num_classes = load_arxiv(
        batch_size=args.batch_size
    )

    ModelCls = MODEL_REGISTRY[args.model]
    model_kwargs = dict(
        in_channels=num_features,
        hidden_channels=args.hidden,
        out_channels=num_classes,
        num_layers=args.layers,
        dropout=args.dropout,
        task="node",
    )
    if args.model == "gat":
        model_kwargs["heads"] = args.heads
    model = ModelCls(**model_kwargs).to(device)
    print(f"arxiv | {args.model.upper()} | params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = get_loss_fn(task="node", dataset_name="ogbn-arxiv")
    evaluator = ArxivEvaluator()
    metric = ogb_metric_name("ogbn-arxiv")

    best_val = -float("inf")
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = eval_epoch(model, val_loader, evaluator, device)
        val_score = val_metrics.get(metric, 0.0)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | val {metric}={val_score:.4f}")

        if val_score > best_val:
            best_val = val_score
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_metrics = eval_epoch(model, test_loader, evaluator, device)
    print(f"\nTest: " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    main()
