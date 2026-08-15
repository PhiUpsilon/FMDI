"""Aggregate FMDI result files into a portable CSV table."""

import argparse
import csv
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing result_nsample*.pk files")
    parser.add_argument("--output", type=Path, default=Path("results/aggregated_results.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.rglob("result_nsample*.pk")):
        with path.open("rb") as handle:
            values = pickle.load(handle)
        if isinstance(values, (list, tuple)) and len(values) >= 3:
            rows.append({"result_file": str(path), "RMSE": values[0], "MAE": values[1], "CRPS": values[2]})
    if not rows:
        raise FileNotFoundError("No valid result_nsample*.pk files under {}".format(args.root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote {} rows to {}".format(len(rows), args.output))


if __name__ == "__main__":
    main()
