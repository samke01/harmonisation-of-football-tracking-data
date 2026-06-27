"""Run direct harmonization metrics for the merged DFL/SkillCorner dataset.

Usage:
    python -m scripts.run_harmonization_evaluation

Outputs:
    dataset/reports/harmonization_evaluation.json
    plots/harmonization_evaluation.png
"""

import argparse
import sys

from src.evaluation import run_harmonization_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-source", type=int, default=20_000)
    parser.add_argument("--mmd-subsample", type=int, default=1_500)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    run_harmonization_evaluation(
        max_per_source=args.max_per_source,
        mmd_subsample=args.mmd_subsample,
        n_permutations=args.permutations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
