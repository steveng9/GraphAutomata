"""
NODE CLASSIFICATION — BasicGNN vs GCN vs GIN
============================================
Graph : 6 nodes, 2 features, 2 classes.
        Two triangles joined by a single bridge edge.

        Cluster A (label 0)      Cluster B (label 1)
            0 ─ 1                    3 ─ 4
             \ /          bridge      \ /
              2  ──────────────────── 5

Features: [a, b] where cluster A has a≈1,b≈0 and cluster B has a≈0,b≈1.

Models compared
───────────────
1. BasicGNN  vanilla sum aggregation, no normalisation
2. GCN       symmetric degree-normalised aggregation
3. GIN       MLP on sum — the most expressive standard GNN

Run:   python tutorials/01_node_classification.py
Debug: set IDE breakpoints on the lines marked  ◀ BREAKPOINT
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, MessagePassing
from torch_geometric.utils import add_self_loops

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA
# ══════════════════════════════════════════════════════════════════════════════

x = torch.tensor([
    [1.0, 0.0],   # node 0  ┐
    [0.9, 0.1],   # node 1  ├─ cluster A → label 0
    [0.8, 0.2],   # node 2  ┘  (bridge end)
    [0.2, 0.8],   # node 3  ┐  (bridge end)
    [0.1, 0.9],   # node 4  ├─ cluster B → label 1
    [0.0, 1.0],   # node 5  ┘
], dtype=torch.float)

y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

edge_index = torch.tensor([
    # cluster A triangle   cluster B triangle    bridge
    [0, 1,  1, 2,  0, 2,   3, 4,  4, 5,  3, 5,   2, 3],
    [1, 0,  2, 1,  2, 0,   4, 3,  5, 4,  5, 3,   3, 2],
], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print("─── Data ────────────────────────────────────────────────────────────")
print(f"  nodes        : {data.num_nodes}")
print(f"  edges        : {data.num_edges}  ({data.num_edges//2} undirected)")
print(f"  node features: {data.num_node_features}")
print(f"  classes      : {y.unique().tolist()}")

# ◀ BREAKPOINT 1 — raw graph
# Useful expressions to evaluate in your debugger:
#   data.x            → (6, 2) feature matrix
#   data.edge_index   → (2, 14) both directions of each edge
#   data.y            → [0, 0, 0, 1, 1, 1]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MANUAL MESSAGE-PASSING ILLUSTRATION
# One forward step written out explicitly so you can see what's happening
# before any learned weights are involved.
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Manual message-passing (1 step, no weights) ─────────────────────")

src, dst = edge_index           # src[i] sends a message to dst[i]

# Step 1 — MESSAGE: each edge collects the source node's feature vector
messages = x[src]               # shape (E, F)  one row per directed edge
# ◀ BREAKPOINT 2a — messages
# messages[i] is the feature of node src[i] travelling along edge i.
# e.g. edge (0→1): messages[0] = x[0] = [1.0, 0.0]

# Step 2 — AGGREGATE: sum all incoming messages at each destination node
agg = torch.zeros_like(x)
agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
# ◀ BREAKPOINT 2b — agg
# agg[v] = sum of features of every neighbour of v (no self yet)

# Step 3 — UPDATE: add self-feature, apply (identity) transform
h = F.relu(agg + x)            # BasicGNN-style: neighbour sum + self
# ◀ BREAKPOINT 2c — h
# h[v] = ReLU(x[v] + sum_{u in N(v)} x[u])
print("  one-step manual output (no weights):")
print("  " + str(h.detach()))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. BasicGNN ─────────────────────────────────────────────────────────────
# Vanilla GNN: sum aggregation, linear transform, no degree normalisation.
# h_v' = ReLU(W · (h_v + Σ_{u∈N(v)} h_u))
# Risk: a high-degree node's aggregated sum grows with its degree.

class BasicConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='sum')
        self.lin = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # propagate() calls: message() → aggregate(sum) → update()
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j: source-node feature for each edge, shape (E, F)
        # ◀ BREAKPOINT 3a — x_j (one row per directed edge)
        return x_j

    def update(self, aggr_out):
        # aggr_out: summed messages arriving at each node, shape (N, F)
        # ◀ BREAKPOINT 3b — aggr_out before linear transform
        return F.relu(self.lin(aggr_out))


class BasicGNN(nn.Module):
    def __init__(self, in_ch=2, hidden=8, out_ch=2):
        super().__init__()
        self.conv1 = BasicConv(in_ch, hidden)
        self.conv2 = BasicConv(hidden, out_ch)

    def forward(self, x, edge_index):
        h1 = self.conv1(x, edge_index)       # (N, hidden)
        # ◀ BREAKPOINT 3c — h1: embeddings after layer 1
        h2 = self.conv2(h1, edge_index)      # (N, out_ch) = logits
        # ◀ BREAKPOINT 3d — h2: final logits, shape (6, 2)
        return h2


# ── 3b. GCN ──────────────────────────────────────────────────────────────────
# Adds degree normalisation: scale each message by 1/√(deg_u · deg_v).
# h_v' = ReLU(Σ_{u∈N(v)∪{v}} (1/√(d_u·d_v)) W h_u)
# Effect: keeps embedding magnitudes stable regardless of degree.

class GCNModel(nn.Module):
    def __init__(self, in_ch=2, hidden=8, out_ch=2):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, out_ch)

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        # ◀ BREAKPOINT 3e — h1: GCN layer 1 embeddings
        # Compare with BasicGNN's h1 — values should be smaller (normalised)
        h2 = self.conv2(h1, edge_index)
        # ◀ BREAKPOINT 3f — h2: GCN final logits
        return h2


# ── 3c. GIN ──────────────────────────────────────────────────────────────────
# MLP applied after sum aggregation; ε is a learned scalar.
# h_v' = MLP((1+ε)·h_v + Σ_{u∈N(v)} h_u)
# Theoretically equivalent in power to the Weisfeiler–Lehman graph test —
# the most expressive message-passing GNN possible.

class GINModel(nn.Module):
    def __init__(self, in_ch=2, hidden=8, out_ch=2):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        mlp2 = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_ch))
        self.conv1 = GINConv(mlp1, train_eps=True)   # ε is learnable
        self.conv2 = GINConv(mlp2, train_eps=True)

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        # ◀ BREAKPOINT 3g — h1: GIN layer 1 embeddings
        # GIN's MLP gives it more flexibility than the single linear in GCN
        h2 = self.conv2(h1, edge_index)
        # ◀ BREAKPOINT 3h — h2: GIN final logits
        return h2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(model, data, epochs=300, lr=0.01):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        logits = model(data.x, data.edge_index)  # (N, num_classes)
        # ◀ BREAKPOINT 4a — logits: one row per node, two class scores each
        # logits.softmax(dim=1) gives class probabilities

        loss = F.cross_entropy(logits, data.y)
        # ◀ BREAKPOINT 4b — loss scalar; watch it decrease across epochs
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
            acc = (pred == data.y).float().mean().item()
            print(f"    epoch {epoch:3d}  loss={loss.item():.4f}  acc={acc:.2f}")

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    # ◀ BREAKPOINT 4c — pred: final class prediction per node
    return (pred == data.y).float().mean().item()


torch.manual_seed(42)
models = {"BasicGNN": BasicGNN(), "GCN": GCNModel(), "GIN": GINModel()}

print("\n─── Training ────────────────────────────────────────────────────────")
results = {}
for name, model in models.items():
    print(f"\n  {name}")
    results[name] = train(model, data)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Final test accuracy ─────────────────────────────────────────────")
for name, acc in results.items():
    bar = "█" * int(acc * 20)
    print(f"  {name:<10}  {acc:.2f}  {bar}")

print("""
Key take-aways
──────────────
BasicGNN  Sum aggregation.  Works on this easy task but magnitudes grow
          with node degree — bad for heterogeneous graphs.

GCN       Degree normalisation keeps embeddings scaled consistently.
          Misses some structural patterns but is very stable.

GIN       The MLP after aggregation can distinguish graph structures that
          GCN cannot — e.g. two nodes with the same neighbour-sum but
          different neighbour distributions.  Most powerful of the three.
""")
