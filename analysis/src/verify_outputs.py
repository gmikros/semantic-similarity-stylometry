"""Sanity-check reconstructed artifacts against the original ``../Data`` files.

Checks performed (each is skipped if the relevant files are absent):

1. Reshape fidelity: stylo ``distance_table_*mfw_0c`` reshaped to wide format must
   match ``Wide_Format_Distance_Matrix_for_Scenario_N.csv`` exactly.
2. Embedding fidelity: reconstructed ``output/Sc{N}_*_embeddings.csv`` compared
   row-wise (cosine similarity) against the originals in ``../Data``.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

import config
from reshape_distances import read_stylo_distance_table


def _find(directory, pattern):
    hits = sorted(glob.glob(str(directory / pattern)))
    return hits[0] if hits else None


def check_reshape(scenario: str) -> None:
    dist = _find(config.DATA_DIR, f"Scenario {scenario}_distance_table_*mfw_0c.csv") \
        or _find(config.DATA_DIR, f"Scenario {scenario}_distance_table_*mfw_0c.txt")
    wide = config.DATA_DIR / f"Wide_Format_Distance_Matrix_for_Scenario_{scenario}.csv"
    if not dist or not wide.exists():
        print(f"[reshape ] Sc{scenario}: skipped (missing source files)")
        return
    reshaped = read_stylo_distance_table(dist).to_numpy(dtype=np.float64)
    original = pd.read_csv(wide, index_col=0).to_numpy(dtype=np.float64)
    if reshaped.shape != original.shape:
        print(f"[reshape ] Sc{scenario}: SHAPE MISMATCH {reshaped.shape} vs {original.shape}")
        return
    max_abs = np.nanmax(np.abs(reshaped - original))
    status = "OK" if max_abs < 1e-6 else f"DIFF max|Δ|={max_abs:.2e}"
    print(f"[reshape ] Sc{scenario}: {status}")


def _load_we(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    we = [c for c in frame.columns if c.upper().startswith("WE")]
    return frame[we].reset_index(drop=True)


def check_embeddings(scenario: str, backend: str) -> None:
    name = {
        "openai": f"Sc{scenario}_OpenAI_average_embeddings.csv",
        "spacy": f"Sc{scenario}_Spacy_embeddings.csv",
    }[backend]
    orig_path = config.DATA_DIR / name
    new_path = config.OUTPUT_DIR / name
    if not orig_path.exists() or not new_path.exists():
        print(f"[{backend:7}] Sc{scenario}: skipped (need both ../Data and output files)")
        return
    a = _load_we(orig_path).to_numpy(dtype=np.float64)
    b = _load_we(new_path).to_numpy(dtype=np.float64)
    if a.shape != b.shape:
        print(f"[{backend:7}] Sc{scenario}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = num / np.where(den == 0, np.nan, den)
    print(f"[{backend:7}] Sc{scenario}: mean row cosine={np.nanmean(cos):.4f} "
          f"(min={np.nanmin(cos):.4f})  — note: row order may differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()
    check_reshape(args.scenario)
    check_embeddings(args.scenario, "spacy")
    check_embeddings(args.scenario, "openai")


if __name__ == "__main__":
    main()
