"""
Train a GNN on ogbg-molhiv (binary graph classification).

Usage:
    python experiments/molhiv_classification.py --model gcn --epochs 50
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ogb.graphproppred import Evaluator

from data.ogb.graph_classification import load_molhiv
from models import MODEL_REGISTRY
from training.losses import get_loss_fn
from training.evaluator import ogb_metric_name
from training.trainer import Trainer
from utils.seed import set_seed


class MolhivEvaluator:
    """Wraps OGB Evaluator to match the Trainer's evaluator.eval() interface."""

    def __init__(self):
        self._eval = Evaluator(name="ogbg-molhiv")

    def eval(self, input_dict):
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]
        # OGB expects (N,1) for both
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(-1)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(-1).long()
        # OGB evaluator expects raw scores (logits fine for rocauc)
        return self._eval.eval({"y_true": y_true, "y_pred": y_pred})


def main():
    parser = argparse.ArgumentParser(description="ogbg-molhiv binary classification")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "graphsage", "gin"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, num_features, _ = load_molhiv(
        batch_size=args.batch_size
    )

    ModelCls = MODEL_REGISTRY[args.model]
    model_kwargs = dict(
        in_channels=num_features,
        hidden_channels=args.hidden,
        out_channels=1,
        num_layers=args.layers,
        dropout=args.dropout,
        task="graph",
    )
    if args.model == "gat":
        model_kwargs["heads"] = args.heads
    model = ModelCls(**model_kwargs).to(device)

    print(f"molhiv | {args.model.upper()} | params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = get_loss_fn(task="graph", dataset_name="ogbg-molhiv")
    evaluator = MolhivEvaluator()
    metric = ogb_metric_name("ogbg-molhiv")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        task="graph",
        device=device,
        evaluator=evaluator,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        metric_key=metric,
        verbose=True,
    )

    test_metrics = trainer.eval_epoch(test_loader)
    print(f"\nTest: " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    main()
