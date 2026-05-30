"""Reshape a ``stylo`` distance table into the comma-separated wide layout.

``stylo`` writes ``distance_table_<k>mfw_0c.csv`` as a space-separated matrix with
quoted row/column labels. This reproduces ``Wide_Format_Distance_Matrix_for_Scenario_N.csv``
(a plain comma-separated matrix with an ``Author`` index column).
"""
from __future__ import annotations

import argparse

import pandas as pd

import config


def read_stylo_distance_table(path) -> pd.DataFrame:
    """Read a whitespace-delimited, quote-labelled stylo distance table."""
    return pd.read_csv(path, sep=r"\s+", engine="python", index_col=0, quotechar='"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to stylo distance_table_*.csv/.txt")
    parser.add_argument("--output", required=True, help="Path to wide CSV to write")
    args = parser.parse_args()

    matrix = read_stylo_distance_table(args.input)
    matrix.index.name = "Author"
    matrix.to_csv(args.output)
    print(f"Wrote {args.output} ({matrix.shape[0]}x{matrix.shape[1]})")


if __name__ == "__main__":
    main()
