"""
Universal Trainer for graph / node / link tasks.

Supports early stopping via patience parameter.
Evaluator must expose: evaluator.eval({"y_true": ..., "y_pred": ...}) → dict
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.loader import DataLoader


class Trainer:
    """
    Parameters
    ----------
    model      : GNNBase subclass
    optimizer  : torch Optimizer
    loss_fn    : nn.Module (from losses.get_loss_fn)
    task       : 'graph' | 'node' | 'link'
    device     : torch.device or str
    evaluator  : object with .eval() matching OGB interface
    scheduler  : optional LR scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        task: str,
        device,
        evaluator,
        scheduler: Optional[_LRScheduler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.task = task
        self.device = device
        self.evaluator = evaluator
        self.scheduler = scheduler

    # ──────────────────────────────────────────────────────────────────────────

    def _get_labels(self, batch) -> torch.Tensor:
        """Extract and shape labels for the current task."""
        y = batch.y
        if self.task == "graph":
            # OGB graph datasets: y shape (N, 1); automata: (N,) — normalise
            if y.dim() > 1 and y.size(1) == 1:
                y = y.view(-1)
            return y
        elif self.task == "node":
            if y.dim() > 1 and y.size(1) == 1:
                y = y.view(-1)
            return y
        else:  # link
            return batch.edge_label.float().view(-1, 1)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_graphs = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch)
            labels = self._get_labels(batch)

            # BCE expects float; CE expects long
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
                loss = self.loss_fn(logits.view(-1, 1), labels.float().view(-1, 1))
            else:
                loss = self.loss_fn(logits, labels.long())

            loss.backward()
            self.optimizer.step()

            n = batch.num_graphs if hasattr(batch, "num_graphs") else batch.y.size(0)
            total_loss += loss.item() * n
            total_graphs += n

        return total_loss / max(total_graphs, 1)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        y_true_list = []
        y_pred_list = []

        for batch in loader:
            batch = batch.to(self.device)
            logits = self.model(batch)
            labels = self._get_labels(batch)
            y_true_list.append(labels.cpu())
            y_pred_list.append(logits.cpu())

        y_true = torch.cat(y_true_list, dim=0)
        y_pred = torch.cat(y_pred_list, dim=0)

        return self.evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 20,
        metric_key: str = "acc",
        verbose: bool = True,
    ) -> dict:
        """
        Train for up to `epochs` epochs with early stopping.

        Returns a dict with training history: {'train_loss', 'val_metrics'}.
        """
        best_val = -float("inf")
        epochs_no_improve = 0
        best_state = None

        history = {"train_loss": [], "val_metrics": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.eval_epoch(val_loader)
            val_score = val_metrics.get(metric_key, 0.0)

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_metrics"].append(val_metrics)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val: {metrics_str}")

            if val_score > best_val:
                best_val = val_score
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val {metric_key}={best_val:.4f})")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history
