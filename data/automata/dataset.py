"""
CellularAutomataDataset — PyG InMemoryDataset.

Generates graphs procedurally from Wolfram 1D elementary CA rules.
Config params are encoded in the processed filename so changing any
parameter triggers automatic re-generation.
"""

import os
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from data.automata.generator import apply_rule, run_automaton
from data.automata.graph_builder import grid_to_pyg
from utils.config import AutomataConfig


class CellularAutomataDataset(InMemoryDataset):
    """
    Graph-level classification dataset.

    Each graph is a CA simulation; the label is the rule index (0-based).

    Parameters
    ----------
    root   : directory where processed data is cached
    config : AutomataConfig (rules, samples, width, timesteps, init, undirected)
    seed   : random seed for reproducibility
    """

    def __init__(
        self,
        root: str = "data/automata/cache",
        config: Optional[AutomataConfig] = None,
        seed: int = 42,
        transform=None,
        pre_transform=None,
    ):
        self.config = config or AutomataConfig()
        self.seed = seed
        # Build label map: rule_number → class index (sorted for stability)
        self.label_map = {r: i for i, r in enumerate(sorted(self.config.rules))}
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # No raw files — data is generated procedurally
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return [f"ca_{self.config.config_hash()}_seed{self.seed}.pt"]

    def download(self):
        # No download needed — data is generated procedurally
        pass

    def process(self):
        rng = np.random.default_rng(self.seed)
        data_list: List[Data] = []

        for rule_number in self.config.rules:
            rule = apply_rule(rule_number)
            for _ in range(self.config.num_samples_per_rule):
                grid = run_automaton(
                    rule=rule,
                    width=self.config.width,
                    timesteps=self.config.timesteps,
                    init=self.config.init,
                    rng=rng,
                )
                data = grid_to_pyg(
                    grid=grid,
                    rule_number=rule_number,
                    label_map=self.label_map,
                    undirected=self.config.undirected,
                )
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_classes(self) -> int:
        return len(self.config.rules)

    @property
    def num_node_features(self) -> int:
        return 3  # [state, w_norm, t_norm]
