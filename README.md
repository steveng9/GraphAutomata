# GraphAutomata

A structured GNN practice repository built with **PyTorch** and **PyTorch Geometric**.

It implements four canonical GNNs from scratch, generates original training data from **Wolfram 1D elementary cellular automata**, and validates against **Stanford OGB benchmarks**.

---

## What it does

**Core task:** classify which Wolfram rule generated a given graph (graph-level classification).

A cellular automaton simulation is run for 30 timesteps on a 30-wide grid. The resulting 2D grid of 0s and 1s is converted into a graph where each cell is a node and edges encode causal influence between timesteps. A GNN reads the graph and predicts which of the 8 possible rules produced it.

The same model architecture is also benchmarked on three OGB datasets covering graph classification, node classification, and link prediction.

---

## Results (2-rule baseline)

Running `gcn` on rules 30 vs 90, 100 samples per rule, 100 epochs:

```
Test results: acc=0.9800  f1=0.9800
```

---

## Project structure

```
GraphAutomata/
├── main.py                         # CLI dispatcher — routes to experiments
├── requirements.txt
│
├── data/
│   ├── automata/
│   │   ├── generator.py            # Pure NumPy CA simulation engine
│   │   ├── graph_builder.py        # CA grid → PyG Data object
│   │   └── dataset.py              # CellularAutomataDataset (InMemoryDataset)
│   └── ogb/
│       ├── graph_classification.py # ogbg-molhiv loader
│       ├── node_classification.py  # ogbn-arxiv loader
│       └── link_prediction.py      # ogbl-collab loader
│
├── models/
│   ├── __init__.py                 # MODEL_REGISTRY dict + re-exports
│   ├── base.py                     # Abstract GNNBase(nn.Module)
│   ├── gcn.py                      # GCN  — symmetric degree normalisation
│   ├── gat.py                      # GAT  — multi-head edge-softmax attention
│   ├── graphsage.py                # GraphSAGE — mean aggregation, no self-loops
│   └── gin.py                      # GIN  — add aggregation, multi-layer readout
│
├── training/
│   ├── trainer.py                  # Universal Trainer (graph / node / link)
│   ├── evaluator.py                # AutomataEvaluator + OGB metric names
│   └── losses.py                   # Loss factory (BCE vs CE per task)
│
├── utils/
│   ├── config.py                   # Dataclasses: ModelConfig, TrainingConfig, AutomataConfig
│   ├── logger.py                   # Structured logging helpers
│   ├── seed.py                     # Global seed setter
│   └── visualize.py                # CA grid and graph visualisation helpers
│
└── experiments/
    ├── automata_classification.py  # Train on CA graphs
    ├── molhiv_classification.py    # ogbg-molhiv binary classification
    ├── arxiv_node_clf.py           # ogbn-arxiv node classification
    └── collab_link_pred.py         # ogbl-collab link prediction
```

---

## Installation

```bash
# 1. Create and activate a conda environment
conda create -n GraphAutomata python=3.9
conda activate GraphAutomata

# 2. Install PyTorch (CPU example; see pytorch.org for GPU builds)
pip install torch

# 3. Install PyTorch Geometric and OGB
pip install torch-geometric ogb

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quickstart

### Automata classification

```bash
# 2-class (Rule 30 vs Rule 90) — fast sanity check
python main.py automata --model gcn --rules 30 90 --samples 100 --epochs 20

# 4-class — full default run
python main.py automata --model gcn

# Compare all four models
python main.py automata --model gcn       --rules 30 90 110 184 --epochs 100
python main.py automata --model graphsage --rules 30 90 110 184 --epochs 100
python main.py automata --model gin       --rules 30 90 110 184 --epochs 100
python main.py automata --model gat       --rules 30 90 110 184 --epochs 100
```

### OGB benchmarks

```bash
python main.py molhiv  --model gcn --epochs 100   # graph classification, ROC-AUC
python main.py arxiv   --model gcn --epochs 100   # node classification, accuracy
python main.py collab  --model gcn --epochs 100   # link prediction, Hits@50
```

### Visualisation

```bash
pip install matplotlib networkx
python utils/visualize.py
```

This opens two windows:

1. **Grid view** — the raw CA output for rules 30, 90, 110, 184. You can see immediately why they are distinguishable: Rule 90 produces a clean Sierpiński fractal, Rule 30 is chaotic, Rule 110 shows structured stripes, Rule 184 behaves like traffic flow.
2. **Graph view** — a small crop of the actual nodes and edges the GNN receives. Nodes are coloured by cell state (black=1, white=0); arrows show causal influence flowing downward through time.

---

## How the CA graph is built

```
CA grid (31 × 30):           Resulting graph:

t=0  1 0 1 1 0 ...           node(0,0)  node(0,1)  node(0,2) ...
t=1  0 0 1 0 1 ...              ↓↘         ↓↘↙         ↓↘↙
t=2  0 1 1 0 1 ...           node(1,0)  node(1,1)  node(1,2) ...
...
```

- **Node** `n = t*W + w` — one node per cell, features `[state, w/(W-1), t/(T-1)]`
- **Edges** — cell `(t, w)` → `(t+1, w-1)`, `(t+1, w)`, `(t+1, w+1)` (boundary-clipped), then symmetrised
- **Label** — the index of the rule (0-based, sorted): e.g. rules [30, 90] → {30:0, 90:1}

The edge structure is identical for all rules. The GCN must learn to distinguish rules purely from the *pattern of node states* propagated through the fixed causal graph.

---

## GNN models

All models inherit from `GNNBase(in_channels, hidden_channels, out_channels, num_layers, dropout, task)` and support `task ∈ {graph, node, link}`.

| Model | Aggregation | Key detail |
|-------|-------------|------------|
| **GCN** | add | Symmetric `D⁻½ÃD⁻½` normalisation; isolated-node masking |
| **GAT** | add | Multi-head attention; intermediate layers `concat=True`, final `concat=False` |
| **GraphSAGE** | mean | Separate `lin_self + lin_neigh`; no self-loops added |
| **GIN** | add | Learned ε; readout concatenates `global_add_pool` from **all** layers |

All are implemented from scratch (no `torch_geometric.nn.GCNConv` etc.) so the internals are fully readable.

---

## Key CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gcn` | `gcn \| gat \| graphsage \| gin` |
| `--rules` | `30 90 110 184` | Wolfram rules to classify |
| `--samples` | `500` | Graphs per rule |
| `--width` | `30` | CA grid width |
| `--timesteps` | `30` | CA evolution steps |
| `--hidden` | `64` | Hidden layer width |
| `--layers` | `3` | Number of GNN layers |
| `--epochs` | `100` | Max training epochs |
| `--patience` | `20` | Early-stopping patience |
| `--seed` | `42` | Global random seed |
| `--device` | `cpu` | `cpu \| cuda` |

---

## Interesting rules to try

| Rule | Behaviour |
|------|-----------|
| 30  | Chaotic — pseudo-random output, used in Mathematica's RNG |
| 90  | XOR rule — produces Sierpiński triangle fractal |
| 110 | Complex — proven Turing-complete |
| 184 | Traffic flow — moving blocks, conserves particle count |
| 45  | Chaotic (asymmetric variant of 30) |
| 18  | Sparse fractal |
| 126 | Dense symmetric fractal |
| 22  | Sparse chaotic |
