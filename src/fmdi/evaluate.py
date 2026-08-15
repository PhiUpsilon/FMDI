"""Inspect RMSE, MAE, and CRPS files produced by FMDI evaluation."""

import argparse
import json
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, help="Path to result_nsample*.pk")
    args = parser.parse_args()
    with args.result.open("rb") as handle:
        values = pickle.load(handle)
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise ValueError("Expected [RMSE, MAE, CRPS] in {}".format(args.result))
    print(json.dumps({"RMSE": values[0], "MAE": values[1], "CRPS": values[2]}, indent=2))


if __name__ == "__main__":
    main()
