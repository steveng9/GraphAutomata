"""
Pure NumPy 1D elementary cellular automata simulation engine.

A Wolfram elementary CA is defined by a rule number 0–255.
Each rule maps every (left, center, right) neighbourhood triple to 0 or 1.
"""

import numpy as np
from typing import Dict, Optional

INTERESTING_RULES = [30, 90, 110, 184, 45, 18, 126, 22]


def apply_rule(rule_number: int) -> Dict[int, int]:
    """
    Decode an 8-bit rule number into a lookup table.

    Returns a dict mapping neighbourhood int codes 0-7 → output state.
    Neighbourhood code: (left << 2) | (center << 1) | right
    """
    assert 0 <= rule_number <= 255, f"Rule must be 0-255, got {rule_number}"
    rule_bits = format(rule_number, "08b")  # MSB = neighbourhood 7
    return {n: int(rule_bits[7 - n]) for n in range(8)}


def run_automaton(
    rule: Dict[int, int],
    width: int = 30,
    timesteps: int = 30,
    init: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate a 1D elementary CA for `timesteps` steps.

    Parameters
    ----------
    rule      : lookup table from apply_rule()
    width     : number of cells per row
    timesteps : number of evolution steps (output has timesteps+1 rows)
    init      : 'random' — uniform random 0/1 row (recommended for training)
                'center' — single live cell in the middle
    rng       : numpy Generator for reproducibility; created fresh if None

    Returns
    -------
    grid : np.ndarray of shape (timesteps+1, width), dtype uint8
    """
    if rng is None:
        rng = np.random.default_rng()

    grid = np.zeros((timesteps + 1, width), dtype=np.uint8)

    # Initialise row 0
    if init == "random":
        grid[0] = rng.integers(0, 2, size=width, dtype=np.uint8)
    elif init == "center":
        grid[0, width // 2] = 1
    else:
        raise ValueError(f"Unknown init mode: {init!r}")

    # Evolve
    for t in range(timesteps):
        row = grid[t]
        # Zero-pad boundaries (cells outside grid treated as 0)
        left = np.roll(row, 1)
        left[0] = 0
        right = np.roll(row, -1)
        right[-1] = 0
        codes = (left.astype(np.int32) << 2) | (row.astype(np.int32) << 1) | right.astype(np.int32)
        grid[t + 1] = np.vectorize(rule.__getitem__)(codes)

    return grid
