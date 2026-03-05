"""
GRAPH CLASSIFICATION — batching, pooling, MLP head
====================================================
Dataset : 8 tiny graphs (4 path graphs, 4 star graphs), all with 4 nodes.
          Node features are all-ones  →  the model must learn from STRUCTURE.

  Path graph (label 0)       Star graph (label 1)
     0 ─ 1 ─ 2 ─ 3              1
                                 |
                             0 ─ 2 ─ 3

  Degrees in path: [1, 2, 2, 1]      mean = 1.5
  Degrees in star: [3, 1, 1, 1]      mean = 1.5  ← same mean, different shape!

  A 1-layer GNN cannot distinguish them (same degree-mean).
  A 2-layer GNN can  — after 2 hops the centre of the star looks very
  different from the middle of the path.

Model   : GIN (2 layers) + global_mean_pool + 2-layer MLP
Focus   : how PyG batches multiple graphs and what pooling produces

Run:   python tutorials/02_graph_classification.py
Debug: set IDE breakpoints on the lines marked  ◀ BREAKPOINT
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — BUILD THE MINI DATASET
# ══════════════════════════════════════════════════════════════════════════════

def make_path():
    """0 ─ 1 ─ 2 ─ 3  (chain)"""
    x = torch.ones(4, 1)          # uniform node features
    edge_index = torch.tensor([
        [0, 1,  1, 2,  2, 3],
        [1, 0,  2, 1,  3, 2],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([0]))

def make_star():
    """Node 0 is the hub connected to leaves 1, 2, 3."""
    x = torch.ones(4, 1)
    edge_index = torch.tensor([
        [0, 1,  0, 2,  0, 3],
        [1, 0,  2, 0,  3, 0],
    ], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([1]))

paths = [make_path() for _ in range(4)]
stars = [make_star() for _ in range(4)]

# Balanced split: 3 paths + 3 stars for training, 1+1 for test
train_data = paths[:3] + stars[:3]
test_data  = [paths[3], stars[3]]
dataset    = paths + stars

train_loader = DataLoader(train_data, batch_size=6, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=2, shuffle=False)

print("─── Dataset ─────────────────────────────────────────────────────────")
print(f"  total graphs : {len(dataset)}  (4 paths + 4 stars)")
print(f"  node features: 1  (all ones — structure only)")
print(f"  train / test : {len(train_data)} / {len(test_data)}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BATCHING EXPLAINED
# When DataLoader collects multiple graphs into one batch it:
#   1. Stacks all node feature matrices → single large x
#   2. Offsets each graph's edge_index by the cumulative node count
#   3. Creates a `batch` tensor: batch[i] = which graph node i belongs to
# ══════════════════════════════════════════════════════════════════════════════

# Peek at one batch to understand the structure
sample_batch = next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))

print("\n─── Batch structure (3 graphs merged) ───────────────────────────────")
print(f"  batch.x.shape          : {sample_batch.x.shape}")
print(f"  batch.edge_index.shape : {sample_batch.edge_index.shape}")
print(f"  batch.batch            : {sample_batch.batch.tolist()}")
print(f"  batch.y                : {sample_batch.y.tolist()}")
# ◀ BREAKPOINT 1 — sample_batch
# batch.batch maps each node → its graph index (0, 1, or 2 here)
# global_mean_pool uses this tensor to average nodes per graph

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL
# Architecture:
#   node features (1-D)
#   → GIN layer 1  (hidden)          per-node embeddings
#   → GIN layer 2  (hidden)          per-node embeddings (2-hop context)
#   → global_mean_pool               one vector per graph
#   → MLP(hidden → hidden → 2)       class logits
# ══════════════════════════════════════════════════════════════════════════════

class GraphClassifier(nn.Module):
    def __init__(self, in_ch=1, hidden=16, num_classes=2):
        super().__init__()

        mlp1 = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.conv1 = GINConv(mlp1, train_eps=True)
        self.conv2 = GINConv(mlp2, train_eps=True)

        # Classifier head applied to the pooled graph embedding
        # No dropout — the dataset is tiny so it only adds noise here.
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, batch):
        # ── GNN layers (per node) ─────────────────────────────────────────────
        h = F.relu(self.conv1(x, edge_index))
        # ◀ BREAKPOINT 2a — h after layer 1, shape (total_nodes_in_batch, hidden)
        # Nodes in the same graph position have different values for path vs star.

        h = F.relu(self.conv2(h, edge_index))
        # ◀ BREAKPOINT 2b — h after layer 2
        # 2-hop context: the star's hub now looks very different from path middles.

        # ── Pooling (per graph) ───────────────────────────────────────────────
        g = global_mean_pool(h, batch)
        # ◀ BREAKPOINT 2c — g, shape (num_graphs_in_batch, hidden)
        # One row per graph.  Path and star graphs should now produce
        # clearly different vectors — that's what the classifier reads.

        # ── Classification head ───────────────────────────────────────────────
        logits = self.head(g)
        # ◀ BREAKPOINT 2d — logits, shape (num_graphs_in_batch, 2)
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
model = GraphClassifier()
opt   = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n─── Training ────────────────────────────────────────────────────────")
for epoch in range(1, 201):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        opt.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        # ◀ BREAKPOINT 3a — logits for the current mini-batch
        loss = F.cross_entropy(logits, batch.y)
        # ◀ BREAKPOINT 3b — loss for this mini-batch
        loss.backward()
        opt.step()
        total_loss += loss.item()

    if epoch % 50 == 0 or epoch == 1:
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch.x, batch.edge_index, batch.batch).argmax(1)
                correct += (pred == batch.y).sum().item()
        acc = correct / len(test_data)
        print(f"  epoch {epoch:3d}  loss={total_loss:.4f}  test_acc={acc:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INSPECT FINAL GRAPH EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Final graph embeddings (pooled, before classifier head) ─────────")
model.eval()

all_graphs  = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
full_batch  = next(iter(all_graphs))

with torch.no_grad():
    h = F.relu(model.conv1(full_batch.x, full_batch.edge_index))
    h = F.relu(model.conv2(h, full_batch.edge_index))
    g = global_mean_pool(h, full_batch.batch)   # (8, hidden)
    preds = model.head(g).argmax(1)

# ◀ BREAKPOINT 4 — g: the 8 graph-level embeddings
# Embeddings for paths (label 0) should cluster together;
# embeddings for stars (label 1) should form a separate cluster.
print(f"  pooled embeddings shape : {g.shape}   (8 graphs × 16 hidden)")
print(f"  predicted labels        : {preds.tolist()}")
print(f"  true labels             : {full_batch.y.tolist()}")

print("""
Key take-aways
──────────────
Batching    PyG merges graphs by stacking x and shifting edge indices.
            The `batch` tensor tells global_mean_pool which nodes belong
            to which graph.

Pooling     global_mean_pool collapses all node embeddings for one graph
            into a single fixed-size vector.  This is how per-node GNN
            outputs become per-graph representations.

2 layers    A 1-layer GNN cannot distinguish path from star (same degree
            mean).  After 2 hops the star's hub accumulates 3 neighbours'
            aggregations — a very different signature from any path node.
""")
