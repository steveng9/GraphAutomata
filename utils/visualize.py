"""
Visualization utilities for CA grids and their PyG graphs.

Requirements: matplotlib, networkx  (pip install matplotlib networkx)

Quick usage:
    from utils.visualize import show_ca_grid, show_ca_graph, compare_rules
    compare_rules([30, 90, 110, 184], width=30, timesteps=30)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data.automata.generator import apply_rule, run_automaton
from data.automata.graph_builder import grid_to_pyg


# ── 1. Raw CA grid (simplest, most readable) ──────────────────────────────────

def show_ca_grid(rule_number: int, width: int = 60, timesteps: int = 60,
                 seed: int = 42, ax=None):
    """
    Display a CA simulation as a black-and-white pixel grid.
    Time flows downward; columns are cells.
    """
    rule = apply_rule(rule_number)
    grid = run_automaton(rule, width=width, timesteps=timesteps,
                         init="random", rng=np.random.default_rng(seed))
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    ax.set_title(f"Rule {rule_number}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cell (w)")
    ax.set_ylabel("Timestep (t)")
    return ax


def compare_rules(rules: list, width: int = 60, timesteps: int = 60, seed: int = 42):
    """Side-by-side grid view of multiple rules."""
    fig, axes = plt.subplots(1, len(rules), figsize=(5 * len(rules), 5))
    if len(rules) == 1:
        axes = [axes]
    for ax, r in zip(axes, rules):
        show_ca_grid(r, width=width, timesteps=timesteps, seed=seed, ax=ax)
    fig.suptitle("Wolfram Elementary CA Rules", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ── 2. Graph view (nodes positioned on the CA grid) ───────────────────────────

def show_ca_graph(rule_number: int, width: int = 20, timesteps: int = 10,
                  seed: int = 42, show_edges: bool = True,
                  mode: str = "standard", ax=None):
    """
    Draw the PyG graph with nodes placed at their (w, -t) grid coordinates.

    mode="standard"    — node colour encodes cell state (white=0, black=1);
                         all causal edges are drawn.
    mode="topological" — all nodes drawn uniformly (grey); edges are only
                         drawn to child cells that are "on", reflecting the
                         representation where topology carries the rule signal.

    Keep width/timesteps small (≤20×15) for a readable plot.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("pip install networkx")

    rule = apply_rule(rule_number)
    grid = run_automaton(rule, width=width, timesteps=timesteps,
                         init="random", rng=np.random.default_rng(seed))
    T, W = grid.shape

    label_map = {rule_number: 0}
    data = grid_to_pyg(grid, rule_number, label_map, undirected=False, mode=mode)

    G = nx.DiGraph()
    for node_id in range(T * W):
        t = node_id // W
        w = node_id % W
        G.add_node(node_id, t=t, w=w, state=int(grid[t, w]))

    ei = data.edge_index.numpy()
    for s, d in zip(ei[0], ei[1]):
        G.add_edge(int(s), int(d))

    pos = {n: (G.nodes[n]["w"], -G.nodes[n]["t"]) for n in G.nodes}

    if mode == "topological":
        colors = ["#888888"] * (T * W)   # uniform grey — no state on nodes
        border = ["#444444"] * (T * W)
        legend_handles = [mpatches.Patch(facecolor="#888888", edgecolor="#444444",
                                         label="node (uniform)")]
        title_suffix = "topological — edges only to 'on' cells"
    else:
        colors = ["black" if G.nodes[n]["state"] == 1 else "white" for n in G.nodes]
        border = ["white" if G.nodes[n]["state"] == 1 else "black" for n in G.nodes]
        legend_handles = [
            mpatches.Patch(facecolor="white", edgecolor="black", label="state=0"),
            mpatches.Patch(facecolor="black", edgecolor="white", label="state=1"),
        ]
        title_suffix = "standard — all causal edges"

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, W * 0.5), max(4, T * 0.5)))

    if show_edges:
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                               arrowsize=8, edge_color="#aaaaaa",
                               connectionstyle="arc3,rad=0.0", width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           edgecolors=border, node_size=200, linewidths=0.8)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(f"Rule {rule_number}  [{title_suffix}]\n"
                 f"{T*W} nodes, {G.number_of_edges()} directed edges", fontsize=10)
    ax.set_xlabel("Cell (w) →")
    ax.set_ylabel("← Timestep (t)")
    ax.axis("off")
    return ax


def compare_graphs(rules: list, width: int = 15, timesteps: int = 8,
                   seed: int = 42, mode: str = "standard"):
    """Side-by-side graph view of multiple rules."""
    fig, axes = plt.subplots(1, len(rules),
                             figsize=(6 * len(rules), max(4, timesteps * 0.6)))
    if len(rules) == 1:
        axes = [axes]
    for ax, r in zip(axes, rules):
        show_ca_graph(r, width=width, timesteps=timesteps, seed=seed, mode=mode, ax=ax)
    mode_label = "topological (edges only to 'on' cells)" if mode == "topological" \
        else "standard (all causal edges)"
    fig.suptitle(f"CA Rules as Graphs  [{mode_label}]", fontsize=13)
    plt.tight_layout()
    plt.show()


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CA grid view (large):")
    compare_rules([30, 90, 110, 184], width=60, timesteps=40)

    print("\nGraph view — standard (all edges, state-coloured nodes):")
    compare_graphs([110, 184], width=30, timesteps=15, mode="standard")

    print("\nGraph view — topological (edges only to 'on' cells, uniform nodes):")
    compare_graphs([110, 184], width=30, timesteps=15, mode="topological")
