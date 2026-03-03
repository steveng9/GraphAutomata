"""
GraphAutomata — CLI dispatcher.

Routes to the appropriate experiment script.

Usage:
    python main.py automata  [--model gcn] [--rules 30 90] [--samples 500] [--epochs 100]
    python main.py molhiv    [--model gcn] [--epochs 100]
    python main.py arxiv     [--model gcn] [--epochs 100]
    python main.py collab    [--model gcn] [--epochs 100]

All model-specific and training flags are forwarded to the sub-script unchanged.
Run `python main.py <experiment> --help` for full argument lists.
"""

import sys
import subprocess
import os

EXPERIMENT_MAP = {
    "automata": "experiments/automata_classification.py",
    "molhiv": "experiments/molhiv_classification.py",
    "arxiv": "experiments/arxiv_node_clf.py",
    "collab": "experiments/collab_link_pred.py",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("Available experiments:")
        for name, path in EXPERIMENT_MAP.items():
            print(f"  {name:10s}  →  {path}")
        sys.exit(0)

    experiment = sys.argv[1]
    if experiment not in EXPERIMENT_MAP:
        print(f"Unknown experiment: {experiment!r}")
        print(f"Choose from: {', '.join(EXPERIMENT_MAP)}")
        sys.exit(1)

    script = os.path.join(os.path.dirname(__file__), EXPERIMENT_MAP[experiment])
    # Forward all remaining args to the sub-script
    cmd = [sys.executable, script] + sys.argv[2:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
