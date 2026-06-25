"""Merge per-model metric JSON files into training_results.csv."""

import argparse
import glob
import json
import os

import pandas as pd


def aggregate(metrics_dir, output_path):
    pattern = os.path.join(metrics_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model metrics found in {metrics_dir}. Run model training stages first.")

    rows = []
    for path in files:
        with open(path, "r") as f:
            rows.append(json.load(f))

    df = pd.DataFrame(rows).set_index("model")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Aggregated {len(rows)} models into {output_path}")


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    parser = argparse.ArgumentParser(description="Aggregate per-model metrics into one CSV.")
    parser.add_argument(
        "--metrics-dir",
        default=os.path.join(root, "reports", "metrics", "models"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(root, "reports", "metrics", "training_results.csv"),
    )
    args = parser.parse_args()
    aggregate(args.metrics_dir, args.output)
