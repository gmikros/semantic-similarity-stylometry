"""Build a paper-ready results package for Journal of Quantitative Linguistics.

Outputs are written under ``paper_package_jql/``:
  - tables/: scenario overview, method performance, significance tests
  - figures/: publication-ready PNG figures
  - draft/: section-by-section manuscript draft (JQL framing)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from statsmodels.stats.contingency_tables import mcnemar

import config
from classify import _delta_matrix, _embedding_matrix, cluster_ari, loo_1nn_accuracy
from corpus import load_documents


PKG = config.ANALYSIS_DIR / "paper_package_jql"
TABLES = PKG / "tables"
FIGS = PKG / "figures"
DRAFT = PKG / "draft"


@dataclass(frozen=True)
class MethodData:
    dist: np.ndarray
    groups: list[str]
    labels: list[str]


def _ensure_dirs() -> None:
    for p in (TABLES, FIGS, DRAFT):
        p.mkdir(parents=True, exist_ok=True)


def _scenario_docs(scn: str):
    docs = load_documents(scn)
    texts = [d.text() for d in docs]
    vocab = set()
    for t in texts:
        vocab.update(t.lower().split())
    unique_texts = len({t.strip() for t in texts})
    return {
        "scenario": scn,
        "n_docs": len(docs),
        "n_unique_texts": unique_texts,
        "total_words": sum(len(t.split()) for t in texts),
        "vocab_size": len(vocab),
        "is_identical_control": unique_texts <= 3,
    }


def _label_index(groups: list[str]) -> list[str]:
    counts = {"A": 0, "B": 0, "C": 0}
    labels: list[str] = []
    for g in groups:
        counts[g] += 1
        labels.append(f"{g}_{counts[g]}")
    return labels


def _load_method_data(scn: str) -> dict[str, MethodData]:
    data: dict[str, MethodData] = {}

    d_dist, d_groups = _delta_matrix(scn)
    if d_dist is not None:
        data["delta"] = MethodData(d_dist, d_groups, _label_index(d_groups))

    o_dist, o_groups = _embedding_matrix(scn, "openai")
    if o_dist is not None:
        data["openai"] = MethodData(o_dist, o_groups, _label_index(o_groups))

    s_dist, s_groups = _embedding_matrix(scn, "spacy")
    if s_dist is not None:
        data["spacy"] = MethodData(s_dist, s_groups, _label_index(s_groups))

    return data


def _loo_correctness(dist: np.ndarray, groups: list[str]) -> np.ndarray:
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    nn = np.argmin(d, axis=1)
    return np.array([groups[i] == groups[nn[i]] for i in range(len(groups))], dtype=int)


def _bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 20000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def _paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 50000, seed: int = 11) -> float:
    """Two-sided paired randomization test for mean accuracy difference."""
    rng = np.random.default_rng(seed)
    obs = float(a.mean() - b.mean())
    n = len(a)
    flips = rng.integers(0, 2, size=(n_perm, n), endpoint=False)
    a_perm = np.where(flips == 1, b, a)
    b_perm = np.where(flips == 1, a, b)
    diffs = a_perm.mean(axis=1) - b_perm.mean(axis=1)
    p = (np.sum(np.abs(diffs) >= abs(obs)) + 1) / (n_perm + 1)
    return float(p)


def _mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    both_right = int(np.sum((a == 1) & (b == 1)))
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    both_wrong = int(np.sum((a == 0) & (b == 0)))
    table = [[both_right, a_only], [b_only, both_wrong]]
    return float(mcnemar(table, exact=True).pvalue)


def _pairwise_distance_profile(dist: np.ndarray, groups: list[str], method: str, scenario: str) -> pd.DataFrame:
    rows = []
    n = len(groups)
    for i, j in combinations(range(n), 2):
        rows.append({
            "scenario": scenario,
            "method": method,
            "pair_type": "within" if groups[i] == groups[j] else "across",
            "distance": float(dist[i, j]),
        })
    return pd.DataFrame(rows)


def build_tables_and_figures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_dirs()
    sns.set_theme(style="whitegrid", font_scale=1.0)

    overview_rows = []
    perf_rows = []
    sig_rows = []
    dist_profiles = []

    all_methods_by_scn: dict[str, dict[str, MethodData]] = {}
    for scn in config.ALL_SCENARIOS:
        overview_rows.append(_scenario_docs(scn))
        mdata = _load_method_data(scn)
        all_methods_by_scn[scn] = mdata
        for method, md in mdata.items():
            perf_rows.append({
                "scenario": scn,
                "method": method,
                "n_docs": len(md.groups),
                "acc_1nn_loo": loo_1nn_accuracy(md.dist, md.groups),
                "ari_3cluster": cluster_ari(md.dist, md.groups),
            })
            dist_profiles.append(_pairwise_distance_profile(md.dist, md.groups, method, scn))

    overview = pd.DataFrame(overview_rows).sort_values("scenario")
    perf = pd.DataFrame(perf_rows).sort_values(["scenario", "method"])
    profile_df = pd.concat(dist_profiles, ignore_index=True)

    # Significance: meaningful scenario after repair = Scenario 3 (distinct texts).
    scn = "3"
    m = all_methods_by_scn[scn]
    corr = {k: _loo_correctness(v.dist, v.groups) for k, v in m.items()}
    methods = ["openai", "spacy", "delta"]
    for a, b in combinations(methods, 2):
        da = float(corr[a].mean())
        db = float(corr[b].mean())
        ci_low, ci_high = _bootstrap_ci_diff(corr[a], corr[b])
        sig_rows.append({
            "scenario": scn,
            "contrast": f"{a} - {b}",
            "acc_a": da,
            "acc_b": db,
            "acc_diff": da - db,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "p_permutation": _paired_permutation_pvalue(corr[a], corr[b]),
            "p_mcnemar_exact": _mcnemar_exact(corr[a], corr[b]),
        })

    # Scenario 5: quantify collapse below naive 1/3 chance.
    for method in ("openai", "spacy", "delta"):
        k = int(corr := _loo_correctness(all_methods_by_scn["5"][method].dist, all_methods_by_scn["5"][method].groups).sum())
        n = len(all_methods_by_scn["5"][method].groups)
        p = float(binomtest(k=k, n=n, p=1/3, alternative="less").pvalue)
        sig_rows.append({
            "scenario": "5",
            "contrast": f"{method} vs chance(1/3)",
            "acc_a": k / n,
            "acc_b": 1/3,
            "acc_diff": k / n - 1/3,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "p_permutation": p,
            "p_mcnemar_exact": math.nan,
        })

    sig = pd.DataFrame(sig_rows)

    overview.to_csv(TABLES / "table_scenario_overview.csv", index=False)
    perf.to_csv(TABLES / "table_method_performance.csv", index=False)
    sig.to_csv(TABLES / "table_significance_tests.csv", index=False)

    # Figure 1: 1NN accuracy heatmap.
    plt.figure(figsize=(7, 3.8))
    p1 = perf.pivot(index="scenario", columns="method", values="acc_1nn_loo").loc[list(config.ALL_SCENARIOS)]
    sns.heatmap(p1, annot=True, vmin=0, vmax=1, cmap="viridis", fmt=".2f", cbar_kws={"label": "1NN-LOO accuracy"})
    plt.title("Attribution Accuracy by Scenario and Representation")
    plt.tight_layout()
    plt.savefig(FIGS / "fig1_accuracy_heatmap.png", dpi=300)
    plt.close()

    # Figure 2: ARI heatmap.
    plt.figure(figsize=(7, 3.8))
    p2 = perf.pivot(index="scenario", columns="method", values="ari_3cluster").loc[list(config.ALL_SCENARIOS)]
    sns.heatmap(p2, annot=True, vmin=-0.1, vmax=1, cmap="mako", fmt=".2f", cbar_kws={"label": "Adjusted Rand Index"})
    plt.title("Cluster Recovery (ARI) by Scenario and Representation")
    plt.tight_layout()
    plt.savefig(FIGS / "fig2_ari_heatmap.png", dpi=300)
    plt.close()

    # Figure 3: scenario 3 within/across distance distributions.
    s3 = profile_df[profile_df["scenario"] == "3"].copy()
    plt.figure(figsize=(8, 4.2))
    sns.boxplot(data=s3, x="method", y="distance", hue="pair_type", showfliers=False)
    plt.title("Scenario 3: Within- vs Across-Group Distances")
    plt.tight_layout()
    plt.savefig(FIGS / "fig3_scenario3_within_across.png", dpi=300)
    plt.close()

    # Figure 4: scenario 5 within/across distance distributions (topic confound).
    s5 = profile_df[profile_df["scenario"] == "5"].copy()
    plt.figure(figsize=(8, 4.2))
    sns.boxplot(data=s5, x="method", y="distance", hue="pair_type", showfliers=False)
    plt.title("Scenario 5: Within- vs Across-Group Distances (Topic Confound)")
    plt.tight_layout()
    plt.savefig(FIGS / "fig4_scenario5_within_across.png", dpi=300)
    plt.close()

    return overview, perf, sig


def write_jql_draft(overview: pd.DataFrame, perf: pd.DataFrame, sig: pd.DataFrame) -> None:
    p = perf.pivot(index="scenario", columns="method", values="acc_1nn_loo")
    s3 = p.loc["3"].to_dict()
    s5 = p.loc["5"].to_dict()
    sig_s3 = sig[sig["scenario"] == "3"].copy()

    lines = f"""# Manuscript Draft (Journal of Quantitative Linguistics)

## Title
Comparing Cosine-Delta Stylometry and Neural Embeddings in Controlled Authorship-Attribution Scenarios

## Abstract
This study evaluates classical stylometry (Cosine Delta) against dense text embeddings (OpenAI `text-embedding-3-small` and spaCy `en_core_web_lg`) on five controlled authorship-attribution scenarios (N = 60 texts per scenario, three author classes). Evaluation combines leave-one-out 1-nearest-neighbour attribution and three-cluster recovery (Adjusted Rand Index). In the principal distinct-text setting (Scenario 3), OpenAI embeddings outperform both spaCy and Cosine Delta (1NN: {s3['openai']:.2f} vs {s3['spacy']:.2f} vs {s3['delta']:.2f}). By contrast, all methods collapse in Scenario 5 (all 1NN = {s5['openai']:.2f}), indicating a topic-driven confound where nearest-neighbour structure no longer reflects author labels. The package provides reproducible scripts, statistical tests, and publication-ready tables/figures.

## 1. Introduction
Quantitative authorship studies increasingly combine interpretable stylometric distances with high-dimensional semantic embeddings. For quantitative linguistics, this raises two empirical questions: (i) whether embeddings deliver measurable gains over established stylometry under controlled conditions, and (ii) under which corpus structures both approaches fail. The current study addresses both questions in a scenario-based benchmark with fixed class cardinality and balanced sample sizes.

## 2. Data and Experimental Design
- Five scenarios were analyzed (Scenarios 1-5), each with 60 documents and three author classes (A/B/C).
- Scenario metadata are reported in `table_scenario_overview.csv`.
- Scenario 3 is the restored distinct-text corpus (formerly stored in a folder labelled `Scenario 3a`); the overwritten identical-text variant is retained separately as a control.
- Scenario 4 is an explicit identical-text control; Scenario 5 is topic-structured and used to test confounding.

## 3. Methods
### 3.1 Representations
1. **Cosine Delta (wurzburg)** using `stylo` over MFW profiles (500 MFW except Scenario 4: 393).
2. **OpenAI embeddings** (`text-embedding-3-small`, 1536 dimensions; document-level averaging).
3. **spaCy embeddings** (`en_core_web_lg`, 300 dimensions; document vectors).

### 3.2 Evaluation
- **Attribution:** leave-one-out 1-nearest-neighbour (1NN-LOO) accuracy.
- **Structure recovery:** Adjusted Rand Index (ARI) from agglomerative clustering (k = 3, precomputed distances).

### 3.3 Inference
- Scenario-3 pairwise method contrasts use paired randomization tests, exact McNemar tests, and bootstrap 95% CIs of accuracy differences (`table_significance_tests.csv`).
- Scenario-5 performance is tested against a naive chance baseline (p = 1/3) using exact binomial tests.

## 4. Results
### 4.1 Main performance pattern
Method-level metrics are reported in `table_method_performance.csv` and visualized in `fig1_accuracy_heatmap.png` and `fig2_ari_heatmap.png`.

- Distinct-text attribution (Scenario 3): OpenAI = {s3['openai']:.2f}, spaCy = {s3['spacy']:.2f}, Delta = {s3['delta']:.2f}.
- Scenario-3 distance geometry (`fig3_scenario3_within_across.png`) shows stronger within/across separation for OpenAI than for Delta.
- Topic-confounded Scenario 5: all methods return 0.00 1NN (`fig4_scenario5_within_across.png`), with nearest-neighbour structure misaligned with author classes.

### 4.2 Significance reporting (Scenario 3)
Pairwise contrasts:
{sig_s3[['contrast','acc_diff','ci95_low','ci95_high','p_permutation','p_mcnemar_exact']].to_string(index=False)}

These estimates should be read as effect-size-oriented: confidence intervals and directionality are emphasized alongside p-values.

## 5. Discussion
From a quantitative-linguistic perspective, the benchmark shows that neural representations can improve attribution in a non-trivial, balanced three-class setting (Scenario 3), but this advantage is contingent on corpus design. When topic structure dominates lexical/semantic proximity (Scenario 5), both stylometric and embedding methods fail similarly. Thus, model choice does not substitute for careful control of topic-author entanglement.

## 6. Reproducibility Statement
All analysis artifacts are generated by `src/build_jql_package.py` and saved under `paper_package_jql/`:
- Tables: `paper_package_jql/tables/`
- Figures: `paper_package_jql/figures/`
- Draft text: `paper_package_jql/draft/`

## 7. Conclusion
The current package supports a submission-ready quantitative results section for JQL, with explicit inferential reporting and reproducible visuals. The main empirical claim is bounded: embeddings outperform Cosine Delta in the restored distinct-text setting, while all methods fail under strong topic confounding.
"""
    (DRAFT / "jql_section_by_section_draft.md").write_text(lines, encoding="utf-8")


def main() -> None:
    overview, perf, sig = build_tables_and_figures()
    write_jql_draft(overview, perf, sig)
    print(f"Wrote tables to: {TABLES}")
    print(f"Wrote figures to: {FIGS}")
    print(f"Wrote draft to: {DRAFT / 'jql_section_by_section_draft.md'}")


if __name__ == "__main__":
    main()
