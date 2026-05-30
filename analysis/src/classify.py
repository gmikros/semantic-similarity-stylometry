"""Authorship-attribution comparison: classic stylometry vs. dense embeddings.

For every scenario and every document representation we evaluate how well the three
author groups (A/B/C) are recovered:

Representations
  * delta  : stylo Cosine-Delta distance matrix (output/Scenario_N_distance_table_*.csv)
  * openai : OpenAI text-embedding-3-small vectors (output/Sc{N}_OpenAI_average_embeddings.csv)
  * spacy  : spaCy en_core_web_lg vectors        (output/Sc{N}_Spacy_embeddings.csv)

Metrics
  * 1NN-LOO accuracy : leave-one-out 1-nearest-neighbour attribution accuracy
                       (cosine distance for embeddings; the Delta matrix as-is)
  * ARI              : Adjusted Rand Index of a 3-cluster agglomerative clustering
                       (average linkage, precomputed distances) vs. the true groups

Writes ``output/classification_summary.csv`` and prints a table.
"""
from __future__ import annotations

import argparse
import glob
import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances

import config
from embedding_distances import load_embeddings
from reshape_distances import read_stylo_distance_table


def _norm(label: str) -> str:
    return re.sub(r"^Sc[0-9a-z]+_", "", str(label).strip().strip('"'))


def _group_of(label: str) -> str:
    return _norm(label).split("_")[0]


def loo_1nn_accuracy(dist: np.ndarray, groups: list[str]) -> float:
    """Leave-one-out 1-nearest-neighbour accuracy from a full distance matrix."""
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    correct = sum(groups[i] == groups[nn[i]] for i in range(len(groups)))
    return correct / len(groups)


def cluster_ari(dist: np.ndarray, groups: list[str]) -> float:
    n_clusters = len(set(groups))
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    # guard against tiny negative values from floating point
    d = np.clip(dist, 0, None)
    labels = model.fit_predict(d)
    codes = pd.factorize(np.asarray(groups))[0]
    return adjusted_rand_score(codes, labels)


def _delta_matrix(scenario: str):
    mfw = config.MFW_PER_SCENARIO.get(scenario, 500)
    path = config.OUTPUT_DIR / f"Scenario_{scenario}_distance_table_{mfw}mfw_0c.csv"
    if not path.exists():
        hits = glob.glob(str(config.OUTPUT_DIR / f"Scenario_{scenario}_distance_table_*mfw_0c.csv"))
        if not hits:
            return None, None
        path = hits[0]
    m = read_stylo_distance_table(path)
    groups = [_group_of(x) for x in m.index]
    # Keep only genuine A/B/C documents; stylo may have ingested stray non-corpus
    # files (e.g. leftover .csv/.png) present in the corpus folder.
    keep = [i for i, g in enumerate(groups) if g in config.GROUPS]
    sub = m.iloc[keep, keep]
    return sub.to_numpy(dtype=np.float64), [groups[i] for i in keep]


def _embedding_matrix(scenario: str, backend: str):
    try:
        emb = load_embeddings(scenario, backend, source="output")
    except FileNotFoundError:
        return None, None
    groups = [_group_of(x) for x in emb.index]
    dist = cosine_distances(emb.to_numpy(dtype=np.float64))
    return dist, groups


def evaluate_scenario(scenario: str) -> list[dict]:
    rows = []
    reps = {
        "delta": lambda: _delta_matrix(scenario),
        "openai": lambda: _embedding_matrix(scenario, "openai"),
        "spacy": lambda: _embedding_matrix(scenario, "spacy"),
    }
    for rep, getter in reps.items():
        dist, groups = getter()
        if dist is None:
            continue
        rows.append({
            "scenario": scenario,
            "representation": rep,
            "n_docs": len(groups),
            "n_groups": len(set(groups)),
            "acc_1nn_loo": round(loo_1nn_accuracy(dist, groups), 4),
            "ari_3cluster": round(cluster_ari(dist, groups), 4),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="*", default=list(config.ALL_SCENARIOS))
    args = parser.parse_args()

    all_rows = []
    for scn in args.scenarios:
        all_rows.extend(evaluate_scenario(scn))

    if not all_rows:
        print("No representations found. Generate embeddings / distance tables first.")
        return

    df = pd.DataFrame(all_rows)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "classification_summary.csv"
    df.to_csv(out, index=False)

    pivot = df.pivot(index="scenario", columns="representation", values="acc_1nn_loo")
    print("\n1NN leave-one-out attribution accuracy (rows=scenario, cols=representation):")
    print(pivot.to_string())
    print("\nFull metrics:")
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
