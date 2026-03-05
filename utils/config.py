from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    model_type: str = "gcn"          # gcn | gat | graphsage | gin
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    # GAT-specific
    heads: int = 4


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 64
    patience: int = 20              # early-stopping patience
    scheduler: str = "none"        # none | cosine | step
    device: str = "cpu"


@dataclass
class AutomataConfig:
    rules: List[int] = field(default_factory=lambda: [30, 90, 110, 184])
    num_samples_per_rule: int = 500
    width: int = 30
    timesteps: int = 30
    init: str = "random"            # random | center
    undirected: bool = True
    graph_mode: str = "standard"    # standard | topological

    def config_hash(self) -> str:
        """Stable string representation used to name the processed cache file."""
        rule_str = "_".join(str(r) for r in sorted(self.rules))
        return (
            f"rules{rule_str}"
            f"_n{self.num_samples_per_rule}"
            f"_w{self.width}"
            f"_t{self.timesteps}"
            f"_init{self.init}"
            f"_{'undir' if self.undirected else 'dir'}"
            f"_{self.graph_mode}"
        )
