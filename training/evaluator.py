"""
Evaluators that mirror the OGB evaluator interface:
    evaluator.eval({"y_true": ..., "y_pred": ...}) → dict of metrics

AutomataEvaluator  : accuracy + macro-F1 for multi-class classification
ogb_metric_name()  : returns the primary metric name for a given OGB dataset
"""

from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score


class AutomataEvaluator:
    """
    Multi-class classification evaluator (graph-level).

    Input dict keys:
        y_true : (N,) long tensor
        y_pred : (N, C) float tensor (logits or probabilities)

    Returns {"acc": float, "f1": float}
    """

    @staticmethod
    def eval(input_dict: Dict) -> Dict[str, float]:
        y_true = input_dict["y_true"]
        y_pred = input_dict["y_pred"]

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            pred_cls = y_pred.argmax(dim=-1).cpu().numpy()
        else:
            import numpy as np
            pred_cls = y_pred.argmax(axis=-1)

        acc = float(accuracy_score(y_true, pred_cls))
        f1 = float(f1_score(y_true, pred_cls, average="macro", zero_division=0))
        return {"acc": acc, "f1": f1}


_OGB_METRIC_MAP = {
    "ogbg-molhiv": "rocauc",
    "ogbn-arxiv": "acc",
    "ogbl-collab": "hits@50",
}


def ogb_metric_name(dataset_name: str) -> str:
    """Return the primary metric name used by OGB for the given dataset."""
    return _OGB_METRIC_MAP.get(dataset_name, "acc")
