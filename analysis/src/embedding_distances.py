"""Cosine distance matrices from embedding tables.

Reads an embeddings CSV (``WE*`` columns), computes the pairwise cosine-distance matrix
across documents and writes it in both wide and long layouts.

By default it reads the original ``../Data`` embedding files so the distances can be
produced without re-running the (paid) OpenAI step; pass ``--source output`` to use the
reconstructed files instead.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

import config


def _embedding_path(scenario: str, backend: str, source: str) -> "tuple[str, object]":
    name = {
        "openai": f"Sc{scenario}_OpenAI_average_embeddings.csv",
        "spacy": f"Sc{scenario}_Spacy_embeddings.csv",
    }[backend]
    directory = config.DATA_DIR if source == "data" else config.OUTPUT_DIR
    return name, directory / name


def load_embeddings(scenario: str, backend: str, source: str) -> pd.DataFrame:
    _, path = _embedding_path(scenario, backend, source)
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    frame = pd.read_csv(path)
    we_cols = [c for c in frame.columns if c.upper().startswith("WE")]
    if not we_cols:
        raise RuntimeError(f"No WE* embedding columns in {path}")
    # Build stable document labels. Prefer an explicit doc_id; otherwise number within group.
    if "doc_id" in frame.columns:
        labels = frame["doc_id"].astype(str).tolist()
    else:
        labels = _labels_from_group(frame.iloc[:, 0].astype(str).tolist())
    frame = frame[we_cols].copy()
    frame.index = labels
    return frame


def _labels_from_group(group_labels: list[str]) -> list[str]:
    """Turn repeated group labels (Sc1_A, Sc1_A, ...) into Sc1_A_1, Sc1_A_2, ..."""
    counts: dict[str, int] = {}
    out = []
    for g in group_labels:
        counts[g] = counts.get(g, 0) + 1
        out.append(f"{g}_{counts[g]}")
    return out


def cosine_distance_matrix(embeddings: pd.DataFrame) -> pd.DataFrame:
    dist = cosine_distances(embeddings.to_numpy(dtype=np.float64))
    return pd.DataFrame(dist, index=embeddings.index, columns=embeddings.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--backend", choices=["openai", "spacy"], required=True)
    parser.add_argument("--source", choices=["data", "output"], default="data")
    args = parser.parse_args()

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    emb = load_embeddings(args.scenario, args.backend, args.source)
    wide = cosine_distance_matrix(emb)

    wide_path = config.OUTPUT_DIR / f"Sc{args.scenario}_{args.backend}_embedding_distance_wide.csv"
    wide.to_csv(wide_path)

    long_df = (
        wide.rename_axis("source")
        .reset_index()
        .melt(id_vars="source", var_name="target", value_name="cosine_distance")
    )
    long_path = config.OUTPUT_DIR / f"Sc{args.scenario}_{args.backend}_embedding_distance_long.csv"
    long_df.to_csv(long_path, index=False)

    print(f"Wrote {wide_path}")
    print(f"Wrote {long_path}")


if __name__ == "__main__":
    main()
