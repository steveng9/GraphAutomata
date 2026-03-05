"""
Train a GNN on the Cellular Automata graph classification task.

Usage:
    python experiments/automata_classification.py \\
        --model gcn --rules 30 90 110 184 --samples 500 --epochs 100

All four models (gcn, gat, graphsage, gin) are supported once Phase 3 is done.
For Phase 2 only gcn is available.
"""

import argparse
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import DataLoader

from data.automata.dataset import CellularAutomataDataset
from utils.config import AutomataConfig, ModelConfig, TrainingConfig
from utils.seed import set_seed
from training.losses import get_loss_fn
from training.evaluator import AutomataEvaluator
from training.trainer import Trainer


def build_model(model_type: str, in_ch: int, hidden: int, out_ch: int, n_layers: int, dropout: float, heads: int):
    """Lazy import so Phase 2 works without Phase 3 models."""
    model_type = model_type.lower()
    if model_type == "gcn":
        from models.gcn import GCN
        return GCN(in_ch, hidden, out_ch, n_layers, dropout, task="graph")
    elif model_type == "graphsage":
        from models.graphsage import GraphSAGE
        return GraphSAGE(in_ch, hidden, out_ch, n_layers, dropout, task="graph")
    elif model_type == "gin":
        from models.gin import GIN
        return GIN(in_ch, hidden, out_ch, n_layers, dropout, task="graph")
    elif model_type == "gat":
        from models.gat import GAT
        return GAT(in_ch, hidden, out_ch, n_layers, dropout, heads=heads, task="graph")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train GNN on Cellular Automata classification")
    parser.add_argument("--model", default="gcn", choices=["gcn", "gat", "graphsage", "gin"])
    parser.add_argument("--rules", nargs="+", type=int, default=[30, 90, 110, 184])
    parser.add_argument("--samples", type=int, default=500, help="Samples per rule")
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4, help="GAT attention heads")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--graph_mode", default="standard", choices=["standard", "topological"],
                        help="standard: node features=[state,w,t]; topological: uniform nodes, edges only to 'on' cells")
    parser.add_argument("--cache_dir", default="data/automata/cache")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    ca_config = AutomataConfig(
        rules=args.rules,
        num_samples_per_rule=args.samples,
        width=args.width,
        timesteps=args.timesteps,
        graph_mode=args.graph_mode,
    )
    print(f"Loading/generating dataset (config hash: {ca_config.config_hash()}) ...")
    dataset = CellularAutomataDataset(root=args.cache_dir, config=ca_config, seed=args.seed)

    # Shuffle and split 70/15/15
    dataset = dataset.shuffle()
    n = len(dataset)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_ds = dataset[:n_train]
    val_ds = dataset[n_train : n_train + n_val]
    test_ds = dataset[n_train + n_val :]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    print(
        f"Dataset: {n} graphs | {len(args.rules)} classes | "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        model_type=args.model,
        in_ch=dataset.num_node_features,
        hidden=args.hidden,
        out_ch=dataset.num_classes,
        n_layers=args.layers,
        dropout=args.dropout,
        heads=args.heads,
    ).to(device)
    print(f"Model: {args.model.upper()}  |  params={sum(p.numel() for p in model.parameters()):,}")

    # ── Training ──────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = get_loss_fn(task="graph", dataset_name="automata")
    evaluator = AutomataEvaluator()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        task="graph",
        device=device,
        evaluator=evaluator,
    )

    print(f"\nTraining {args.model.upper()} for up to {args.epochs} epochs "
          f"(patience={args.patience}) ...\n")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        metric_key="acc",
        verbose=True,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    test_metrics = trainer.eval_epoch(test_loader)
    print(f"\nTest results: " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    main()
