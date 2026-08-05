"""
Aggregate per-seed evaluation results into mean ± standard deviation.

Run after evaluating several seeds with distinct --tag values:

    for %s in (1 2 3 4 5) do python src/evaluate.py --split test ^
        --checkpoint checkpoints\\best_seed%s.pt --tag seed%s
    python src/aggregate_seeds.py

Reporting spread across seeds matters here because a single accuracy figure on
~1,100 images carries a standard error near 1.5 points. Two runs differing by a
point are indistinguishable, and a headline number without that context invites
reading precision into noise that isn't there.
"""

import argparse
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path("results")

# Fields to summarize: json key -> (label, unit, decimals)
FIELDS = {
    "accuracy": ("Accuracy", "", 4),
    "mean_decade_error_years": ("Mean decade error", " years", 2),
}


def load_runs(results_dir: Path, split: str) -> list[dict]:
    """Load metrics_{split}_*.json, skipping the untagged single-run file."""
    paths = sorted(results_dir.glob(f"metrics_{split}_*.json"))
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = path
        runs.append(data)
    return runs


def summarize(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation; std is 0.0 for a single run."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    runs = load_runs(args.results_dir, args.split)
    if not runs:
        raise SystemExit(
            f"No metrics_{args.split}_*.json in {args.results_dir}. "
            f"Evaluate each seed with a --tag first."
        )

    print(f"\n{len(runs)} run(s) on the {args.split} split\n")
    header = f"{'run':<12}" + "".join(f"{label:>22}" for label, _, _ in FIELDS.values())
    print(header)
    print("-" * len(header))

    for run in runs:
        cells = ""
        for key, (_, unit, dp) in FIELDS.items():
            cells += f"{run[key]:>22.{dp}f}" if key in run else f"{'—':>22}"
        print(f"{run.get('tag') or run['_path'].stem:<12}" + cells)

    print("-" * len(header))

    summary = {"split": args.split, "n_runs": len(runs)}
    cells = ""
    for key, (_, unit, dp) in FIELDS.items():
        values = [r[key] for r in runs if key in r]
        if not values:
            cells += f"{'—':>22}"
            continue
        mean, std = summarize(values)
        summary[key] = {"mean": round(mean, dp + 1), "std": round(std, dp + 1)}
        cells += f"{mean:>12.{dp}f} +/- {std:.{dp}f}"
    print(f"{'mean +/- std':<12}" + cells)

    n_samples = {r.get("n_samples") for r in runs}
    if len(n_samples) == 1:
        summary["n_samples"] = n_samples.pop()

    out_path = args.results_dir / f"summary_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    if len(runs) == 1:
        print("Only one run found — std is 0.0 by definition, not a real spread.")


if __name__ == "__main__":
    main()
