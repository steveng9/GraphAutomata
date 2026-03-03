"""
Loss function factory.

  Binary tasks  (molhiv, link prediction) → BCEWithLogitsLoss
  Multi-class   (automata, arxiv)          → CrossEntropyLoss
"""

import torch.nn as nn


BINARY_DATASETS = {"ogbg-molhiv", "ogbl-collab", "link"}


def get_loss_fn(task: str, dataset_name: str = "") -> nn.Module:
    """
    Return an appropriate loss function.

    Parameters
    ----------
    task         : 'graph' | 'node' | 'link'
    dataset_name : OGB dataset name or empty string for automata
    """
    if task == "link" or dataset_name in BINARY_DATASETS:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()
