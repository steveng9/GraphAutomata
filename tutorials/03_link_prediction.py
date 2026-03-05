"""
LINK PREDICTION — encode nodes, score pairs
============================================
Graph : 6 nodes arranged in a partial grid.  Some edges are held out
        as the test set; the model must learn to predict them.

  Full graph (edges not all shown to the model at training time):
      0 ─ 1 ─ 2
      │   │   │
      3 ─ 4 ─ 5

  Edge split (RandomLinkSplit):
    train edges  → used for message passing AND as positive training labels
    val edges    → used only as positive val labels (not for message passing)
    test edges   → held out completely

  Negative examples are pairs of nodes with NO edge between them.

Model   : 2-layer GCN encoder → dot-product decoder → sigmoid score
          score(u, v) = sigmoid(h_u · h_v)

Metric  : AUC-ROC  (better than accuracy for imbalanced pos/neg sets)

Run:   python tutorials/03_link_prediction.py
Debug: set IDE breakpoints on the lines marked  ◀ BREAKPOINT
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA AND EDGE SPLIT
# ══════════════════════════════════════════════════════════════════════════════

# 6 nodes, simple 2×3 grid
x = torch.eye(6)            # one-hot identity features — each node unique

# All 7 undirected edges of the grid (stored as directed both ways)
edge_index = torch.tensor([
    [0, 1,  1, 2,  0, 3,  1, 4,  2, 5,  3, 4,  4, 5],
    [1, 0,  2, 1,  3, 0,  4, 1,  5, 2,  4, 3,  5, 4],
], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, num_nodes=6)

print("─── Data ────────────────────────────────────────────────────────────")
print(f"  nodes : {data.num_nodes}")
print(f"  edges : {data.num_edges // 2} undirected  ({data.num_edges} directed)")

# RandomLinkSplit splits edges into train/val/test.
# It adds four new attributes to each split:
#   edge_index          → edges used for message passing
#   edge_label_index    → edges to score (pos + neg pairs)
#   edge_label          → 1 for real edges, 0 for negative samples
splitter = RandomLinkSplit(
    num_val=0.2,
    num_test=0.2,
    is_undirected=True,
    add_negative_train_samples=False,   # we sample negatives ourselves each epoch
)
train_data, val_data, test_data = splitter(data)

print(f"\n─── Edge split ──────────────────────────────────────────────────────")
print(f"  train message-passing edges : {train_data.edge_index.shape[1] // 2}")
print(f"  train positive labels       : {(train_data.edge_label == 1).sum()}")
print(f"  val   positive labels       : {(val_data.edge_label == 1).sum()}")
print(f"  test  positive labels       : {(test_data.edge_label == 1).sum()}")

# ◀ BREAKPOINT 1 — train_data, val_data, test_data
# train_data.edge_index   → subset of edges for message passing
# train_data.edge_label_index → (2, pos) pairs to score during training
# val_data.edge_label_index   → (2, pos+neg) pairs for validation

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL
# Encoder: 2-layer GCN that produces a dense embedding per node.
# Decoder: dot product between the two endpoint embeddings, then sigmoid.
# ══════════════════════════════════════════════════════════════════════════════

class GCNEncoder(nn.Module):
    """Turns node features into node embeddings using graph structure."""
    def __init__(self, in_ch, hidden, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, out_ch)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        # ◀ BREAKPOINT 2a — h after layer 1, shape (N, hidden)
        h = self.conv2(h, edge_index)
        # ◀ BREAKPOINT 2b — h after layer 2, shape (N, out_ch)
        # These are the final node embeddings.  Two nodes that should be
        # linked will ideally end up with similar embedding vectors.
        return h


def decode(z, edge_label_index):
    """
    Dot-product decoder.
    score(u,v) = sigmoid(z_u · z_v)
    High score → model predicts an edge between u and v.
    """
    src = edge_label_index[0]
    dst = edge_label_index[1]
    dot = (z[src] * z[dst]).sum(dim=1)   # element-wise product then sum
    # ◀ BREAKPOINT 2c — dot, one scalar per candidate edge
    # Positive pairs (real edges) should converge to high dot products.
    # Negative pairs (non-edges) should converge to low dot products.
    return torch.sigmoid(dot)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION HELPER
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, split_data, z):
    """Score all edge_label_index pairs and return AUC."""
    model.eval()
    with torch.no_grad():
        scores = decode(z, split_data.edge_label_index)
    labels = split_data.edge_label.float()
    # ◀ BREAKPOINT 3 — scores, labels
    # scores: predicted probability of edge existing for each pair
    # labels: 1 = real edge, 0 = negative sample
    return roc_auc_score(labels.numpy(), scores.numpy())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

torch.manual_seed(42)
model = GCNEncoder(in_ch=6, hidden=16, out_ch=8)
opt   = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n─── Training ────────────────────────────────────────────────────────")
for epoch in range(1, 201):
    model.train()
    opt.zero_grad()

    # Encode: run GCN over the training message-passing graph
    z = model(train_data.x, train_data.edge_index)
    # ◀ BREAKPOINT 4a — z: node embeddings, shape (6, 8)
    # These change each epoch as the model learns.

    # Positive pairs: edges that actually exist (from the training split)
    pos_edge = train_data.edge_label_index

    # Negative pairs: randomly sample non-edges of the same count
    neg_edge = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=pos_edge.shape[1],
    )
    # ◀ BREAKPOINT 4b — pos_edge, neg_edge
    # pos_edge: (2, num_pos) pairs we want score → 1
    # neg_edge: (2, num_neg) pairs we want score → 0

    # Score both and compute binary cross-entropy
    all_edges  = torch.cat([pos_edge, neg_edge], dim=1)
    all_labels = torch.cat([
        torch.ones(pos_edge.shape[1]),
        torch.zeros(neg_edge.shape[1]),
    ])
    scores = decode(z, all_edges)
    # ◀ BREAKPOINT 4c — scores, all_labels
    # After many epochs: pos scores approach 1, neg scores approach 0.
    loss = F.binary_cross_entropy(scores, all_labels)
    loss.backward()
    opt.step()

    if epoch % 50 == 0 or epoch == 1:
        z_eval = model(train_data.x, train_data.edge_index)
        val_auc  = evaluate(model, val_data,  z_eval)
        test_auc = evaluate(model, test_data, z_eval)
        print(f"  epoch {epoch:3d}  loss={loss.item():.4f}"
              f"  val_auc={val_auc:.3f}  test_auc={test_auc:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INSPECT FINAL EMBEDDINGS AND SCORES
# ══════════════════════════════════════════════════════════════════════════════

print("\n─── Final node embeddings (6 × 8) ──────────────────────────────────")
model.eval()
with torch.no_grad():
    z_final = model(train_data.x, train_data.edge_index)

# ◀ BREAKPOINT 5a — z_final
# Nodes that are connected tend to have higher dot products.
# Try: (z_final[0] * z_final[1]).sum()  vs  (z_final[0] * z_final[5]).sum()
print(z_final.round(decimals=3))

print("\n─── Edge scores (all candidate pairs) ───────────────────────────────")
# Identify which edges were used for message passing vs held out
train_ei = set(zip(train_data.edge_index[0].tolist(),
                   train_data.edge_index[1].tolist()))

with torch.no_grad():
    for u in range(6):
        for v in range(u + 1, 6):
            pair  = torch.tensor([[u], [v]])
            score = decode(z_final, pair).item()
            is_edge = any(
                (edge_index[0] == u) & (edge_index[1] == v)
            )
            in_train = (u, v) in train_ei or (v, u) in train_ei
            if is_edge:
                tag = "← train edge" if in_train else "← HELD-OUT (val/test)"
            else:
                tag = ""
            print(f"  ({u},{v})  score={score:.3f}  {tag}")

print("""
Note on held-out edges:
  RandomLinkSplit removes val/test edges from the message-passing graph.
  The encoder never "sees" those connections, so their dot-product score
  can be low even though they are real edges — that is the whole point of
  the task.  Val/test AUC above measures how well the model ranks those
  held-out pairs relative to sampled non-edges.""")

# ◀ BREAKPOINT 5b — final scores
# Real edges should score higher than non-edges.

print("""
Key take-aways
──────────────
Encoder     The GCN encoder maps each node to an embedding that reflects
            its neighbourhood.  Connected nodes end up with similar vectors.

Decoder     A simple dot product measures "compatibility" of two embeddings.
            No edge-specific parameters needed.

Negatives   During training we randomly sample non-edges each epoch.
            This teaches the model to push non-neighbours apart.

AUC-ROC     Measures ranking quality: does the model rank real edges above
            non-edges?  Better than accuracy for imbalanced pos/neg sets.
""")
