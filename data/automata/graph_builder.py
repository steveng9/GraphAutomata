"""
Convert a CA simulation grid into a PyTorch Geometric Data object.

Two modes
---------
"standard"    (default)
    Node features (3-D float32): [state, w/(W-1), t/(T-1)]
    Edges (directed): cell (t,w) → (t+1, w'), w' ∈ {w-1,w,w+1} — always present

"topological"
    Node features: uniform ones (1-D) — no state information on nodes
    Edges: same directed pattern, BUT only drawn to child cells that are "on"
           (grid[t+1, w'] == 1).  The automata rule is expressed purely in
           the edge topology rather than node attributes.

Node indexing (row-major): node_id = t * W + w
Optionally symmetrised to undirected via to_undirected().
"""

from typing import Dict, List
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def grid_to_pyg(
    grid: np.ndarray,
    rule_number: int,
    label_map: Dict[int, int],
    undirected: bool = True,
    mode: str = "standard",
) -> Data:
    """
    Parameters
    ----------
    grid        : (T, W) uint8 array from run_automaton (T = timesteps+1)
    rule_number : Wolfram rule used to generate grid
    label_map   : dict mapping rule_number → class index
    undirected  : if True apply to_undirected (needed for GCN/SAGE)
    mode        : "standard" or "topological" (see module docstring)

    Returns
    -------
    torch_geometric.data.Data with fields x, edge_index, y, num_nodes
    """
    T, W = grid.shape
    N = T * W

    src_list: List[int] = []
    dst_list: List[int] = []

    if mode == "topological":
        # ── Uniform node features ──────────────────────────────────────────────
        x = torch.ones(N, 1)

        # ── Edges only to "on" children ───────────────────────────────────────
        # Iterate by child: for each (t+1, w2) that is "on", add edges from its
        # three parents at (t, w2-1), (t, w2), (t, w2+1).
        for t in range(T - 1):
            for w2 in range(W):
                if grid[t + 1, w2] == 1:
                    dst = (t + 1) * W + w2
                    for dw in (-1, 0, 1):
                        w = w2 + dw
                        if 0 <= w < W:
                            src_list.append(t * W + w)
                            dst_list.append(dst)

    else:  # "standard"
        # ── Node features: [state, w_norm, t_norm] ────────────────────────────
        states = grid.reshape(-1).astype(np.float32)
        t_idx = np.repeat(np.arange(T), W).astype(np.float32)
        w_idx = np.tile(np.arange(W), T).astype(np.float32)
        w_norm = w_idx / max(W - 1, 1)
        t_norm = t_idx / max(T - 1, 1)
        x = torch.from_numpy(np.stack([states, w_norm, t_norm], axis=1))

        # ── All edges ─────────────────────────────────────────────────────────
        for t in range(T - 1):
            for w in range(W):
                src = t * W + w
                for dw in (-1, 0, 1):
                    w2 = w + dw
                    if 0 <= w2 < W:
                        src_list.append(src)
                        dst_list.append((t + 1) * W + w2)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    if undirected:
        edge_index = to_undirected(edge_index, num_nodes=N)

    # ── Label ─────────────────────────────────────────────────────────────────
    y = torch.tensor([label_map[rule_number]], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, num_nodes=N)
