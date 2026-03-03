"""
Convert a CA simulation grid into a PyTorch Geometric Data object.

Node indexing (row-major): node_id = t * W + w
Node features (3-D float32): [state, w/(W-1), t/(T-1)]
Edges (directed): cell (t,w) → (t+1, w-1), (t+1, w), (t+1, w+1)  [boundary-clipped]
Optionally symmetrised to undirected via to_undirected().
"""

from typing import Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def grid_to_pyg(
    grid: np.ndarray,
    rule_number: int,
    label_map: Dict[int, int],
    undirected: bool = True,
) -> Data:
    """
    Parameters
    ----------
    grid        : (T, W) uint8 array from run_automaton (T = timesteps+1)
    rule_number : Wolfram rule used to generate grid
    label_map   : dict mapping rule_number → class index
    undirected  : if True apply to_undirected (needed for GCN/SAGE)

    Returns
    -------
    torch_geometric.data.Data with fields x, edge_index, y, num_nodes
    """
    T, W = grid.shape

    # ── Node features ─────────────────────────────────────────────────────────
    states = grid.reshape(-1).astype(np.float32)          # (T*W,)
    t_idx = np.repeat(np.arange(T), W).astype(np.float32)
    w_idx = np.tile(np.arange(W), T).astype(np.float32)
    w_norm = w_idx / max(W - 1, 1)
    t_norm = t_idx / max(T - 1, 1)
    x = np.stack([states, w_norm, t_norm], axis=1)        # (T*W, 3)
    x = torch.from_numpy(x)

    # ── Edges ─────────────────────────────────────────────────────────────────
    # For each cell (t, w) add edges to (t+1, w'), w' in {w-1, w, w+1}
    src_list: List[int] = []
    dst_list: List[int] = []

    for t in range(T - 1):
        for w in range(W):
            src = t * W + w
            for dw in (-1, 0, 1):
                w2 = w + dw
                if 0 <= w2 < W:
                    dst = (t + 1) * W + w2
                    src_list.append(src)
                    dst_list.append(dst)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=T * W)

    # ── Label ─────────────────────────────────────────────────────────────────
    y = torch.tensor([label_map[rule_number]], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=T * W)
