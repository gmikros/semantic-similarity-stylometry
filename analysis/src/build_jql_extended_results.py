"""Build extended reproducibility artifacts matching the full JQL paper methods."""
from __future__ import annotations

from itertools import combinations
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import KNeighborsClassifier

import config
from classify import loo_1nn_accuracy, cluster_ari
from corpus import load_documents
from embedding_distances import load_embeddings
from reshape_distances import read_stylo_distance_table

PKG = config.ANALYSIS_DIR / "paper_package_jql_extended"
TABLES = PKG / "tables"
FIGS = PKG / "figures"


TARGET_MAP = {
    "1": (0.9, 0.1),
    "2": (0.7, 0.3),
    "3": (0.5, 0.5),
    "4": (0.3, 0.7),
    "5": (0.1, 0.9),
}
ROLE_MAP = {
    "1": "High-separation baseline",
    "2": "High separation maintained",
    "3": "Balanced, discriminative condition",
    "4": "Identical-text control",
    "5": "Topic confound, attribution collapse",
}


def _ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)


def _norm_label(lbl: str) -> str:
    s = str(lbl).strip().strip('"')
    if s.startswith("Sc") and "_" in s:
        # Sc3_A_12 -> A_12
        return s.split("_", 1)[1]
    return s


def _delta_path(scn: str):
    mfw = config.MFW_PER_SCENARIO[scn]
    p = config.OUTPUT_DIR / f"Scenario_{scn}_distance_table_{mfw}mfw_0c.csv"
    if p.exists():
        return p
    p2 = config.DATA_DIR / f"Scenario {scn}_distance_table_{mfw}mfw_0c.csv"
    if p2.exists():
        return p2
    p3 = config.DATA_DIR / f"Scenario {scn}_distance_table_{mfw}mfw_0c.txt"
    if p3.exists():
        return p3
    raise FileNotFoundError(f"No Delta table for scenario {scn}")


def _method_data(scn: str):
    # delta
    dm = read_stylo_distance_table(_delta_path(scn))
    dlabels_all = [_norm_label(x) for x in dm.index]
    keep = [i for i, x in enumerate(dlabels_all) if ("_" in x and x.split("_")[0] in {"A", "B", "C"} and x.split("_")[1].isdigit())]
    dm = dm.iloc[keep, keep]
    dlabels = [_norm_label(x) for x in dm.index]
    dgroups = [x.split("_")[0] for x in dlabels]
    out = {"delta": (dm.to_numpy(float), dgroups, dlabels)}

    # embeddings (use regenerated output for consistency)
    for method in ("openai", "spacy"):
        emb = load_embeddings(scn, method, source="output")
        labels = [_norm_label(x) for x in emb.index]
        groups = [x.split("_")[0] for x in labels]
        dist = cosine_distances(emb.to_numpy(float))
        out[method] = (dist, groups, labels)
    return out


def _within_between_stats(dist: np.ndarray, groups: list[str]):
    within, between = [], []
    for i, j in combinations(range(len(groups)), 2):
        if groups[i] == groups[j]:
            within.append(float(dist[i, j]))
        else:
            between.append(float(dist[i, j]))
    mw = float(np.mean(within)) if within else math.nan
    mb = float(np.mean(between)) if between else math.nan
    ratio = mb / mw if (within and abs(mw) > 1e-12) else math.inf
    return mw, mb, ratio


def _pair_type_decomposition(dist: np.ndarray, groups: list[str], labels: list[str]):
    within, same_topic_between, diff_topic_between = [], [], []
    topics = [int(x.split("_")[1]) for x in labels]
    for i, j in combinations(range(len(groups)), 2):
        d = float(dist[i, j])
        if groups[i] == groups[j]:
            within.append(d)
        elif topics[i] == topics[j]:
            same_topic_between.append(d)
        else:
            diff_topic_between.append(d)
    return {
        "mean_within": float(np.mean(within)) if within else math.nan,
        "mean_between_same_topic": float(np.mean(same_topic_between)) if same_topic_between else math.nan,
        "mean_between_diff_topic": float(np.mean(diff_topic_between)) if diff_topic_between else math.nan,
    }


def _knn_accuracy_curve(dist: np.ndarray, groups: list[str], ks=(1, 3, 5, 7)):
    dist = np.clip(dist, 0, None)
    y = np.array(groups)
    n = len(y)
    out = {}
    for k in ks:
        preds = []
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
            clf.fit(dist[np.ix_(tr, tr)], y[tr])
            preds.append(clf.predict(dist[np.ix_([i], tr)])[0])
        out[k] = float(np.mean(np.array(preds) == y))
    return out


def _silhouette(dist: np.ndarray, groups: list[str]) -> float:
    try:
        return float(silhouette_score(np.clip(dist, 0, None), groups, metric="precomputed"))
    except Exception:
        return math.nan


def _mantel_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 17):
    def tril(x):
        i, j = np.tril_indices_from(x, k=-1)
        return x[i, j]

    va = tril(a)
    vb = tril(b)
    obs = float(pearsonr(va, vb).statistic)
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        rb = float(pearsonr(va, tril(b[np.ix_(p, p)])).statistic)
        if abs(rb) >= abs(obs):
            ge += 1
    return obs, float((ge + 1) / (n_perm + 1))


def build_extended() -> None:
    _ensure_dirs()
    sns.set_theme(style="whitegrid")

    t1_rows, t2_rows, t3_rows, t4_rows, t5_rows, knn_rows = [], [], [], [], [], []

    for scn in config.ALL_SCENARIOS:
        md = _method_data(scn)

        # table 1
        docs = load_documents(scn)
        texts = [d.text() for d in docs]
        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())
        unique_texts = len({t.strip() for t in texts})
        intra, inter = TARGET_MAP[scn]
        t1_rows.append({
            "scenario": scn,
            "intra_overlap": intra,
            "inter_overlap": inter,
            "unique_texts": unique_texts,
            "vocab_size": len(vocab),
            "mean_words_per_doc": int(round(sum(len(t.split()) for t in texts) / len(texts))),
            "mfw_used": int(config.MFW_PER_SCENARIO[scn]),
        })

        # table 2 + table 4 based on delta
        d_dist, d_groups, d_labels = md["delta"]
        mw, mb, ratio = _within_between_stats(d_dist, d_groups)
        t2_rows.append({
            "scenario": scn,
            "within_author_delta": mw,
            "between_author_delta": mb,
            "separation_ratio": ratio,
            "role_in_the_design": ROLE_MAP[scn],
        })
        t4_rows.append({"scenario": scn, **_pair_type_decomposition(d_dist, d_groups, d_labels)})

        # table 3 and k-NN
        for method, (dist, groups, labels) in md.items():
            t3_rows.append({
                "scenario": scn,
                "representation": {"delta": "Cosine Delta", "openai": "OpenAI", "spacy": "spaCy"}[method],
                "acc_1nn_loo": loo_1nn_accuracy(dist, groups),
                "ari_k3": cluster_ari(dist, groups),
                "silhouette": _silhouette(dist, groups),
            })
            for k, acc in _knn_accuracy_curve(dist, groups).items():
                knn_rows.append({"scenario": scn, "method": method, "k": k, "knn_acc": acc})

        # table 5 correlations (pairwise aligned by labels)
        pairs = [("delta", "openai"), ("delta", "spacy"), ("openai", "spacy")]
        row = {"scenario": scn}
        for a, b in pairs:
            da, _, la = md[a]
            db, _, lb = md[b]
            ia = {x: i for i, x in enumerate(la)}
            ib = {x: i for i, x in enumerate(lb)}
            common = sorted([x for x in ia if x in ib])
            A = da[np.ix_([ia[x] for x in common], [ia[x] for x in common])]
            B = db[np.ix_([ib[x] for x in common], [ib[x] for x in common])]
            r, p = _mantel_pvalue(A, B)
            row[f"{a}_~_{b}"] = r
            row[f"p_{a}_~_{b}"] = p
        t5_rows.append(row)

    t1 = pd.DataFrame(t1_rows).sort_values("scenario")
    t2 = pd.DataFrame(t2_rows).sort_values("scenario")
    t3 = pd.DataFrame(t3_rows).sort_values(["scenario", "representation"])
    t4 = pd.DataFrame(t4_rows).sort_values("scenario")
    t5 = pd.DataFrame(t5_rows).sort_values("scenario")
    knn = pd.DataFrame(knn_rows).sort_values(["scenario", "method", "k"])

    t1.to_csv(TABLES / "table1_paper_style.csv", index=False)
    t2.to_csv(TABLES / "table2_paper_style.csv", index=False)
    t3.to_csv(TABLES / "table3_paper_style.csv", index=False)
    t4.to_csv(TABLES / "table4_pair_type_decomposition_delta.csv", index=False)
    t5.to_csv(TABLES / "table5_paper_style.csv", index=False)
    knn.to_csv(TABLES / "table_knn_curves.csv", index=False)

    # Inferential table used as Table 4 in the manuscript
    try:
        sig = pd.read_csv(config.ANALYSIS_DIR / "paper_package_jql" / "tables" / "table_significance_tests.csv")
        sig_out = pd.DataFrame({
            "Sc.": sig["scenario"],
            "Contrast": sig["contrast"].astype(str).str.replace(" - ", " vs ", regex=False).str.replace("(1/3)", "", regex=False),
            "Acc. A": sig["acc_a"],
            "Acc. B": sig["acc_b"],
            "Δ acc. [95% CI]": sig.apply(
                lambda r: (
                    f"{r['acc_diff']:.3f}"
                    if pd.isna(r["ci95_low"])
                    else f"{r['acc_diff']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
                ),
                axis=1,
            ),
            "p (perm.)": sig["p_permutation"],
            "p (exact)": sig["p_mcnemar_exact"].fillna(sig["p_permutation"]),
        })
        sig_out.to_csv(TABLES / "table4_inferential_paper_style.csv", index=False)
    except Exception:
        pass

    # figs
    plt.figure(figsize=(8, 4.2))
    sns.barplot(data=t4.melt(id_vars="scenario", var_name="pair_type", value_name="distance"), x="scenario", y="distance", hue="pair_type")
    plt.title("Delta Distance Decomposition by Pair Type")
    plt.tight_layout(); plt.savefig(FIGS / "fig4_delta_pair_type_decomposition.png", dpi=300); plt.close()

    kplot = knn[knn["scenario"].isin(["3", "5"])].copy()
    g = sns.relplot(data=kplot, x="k", y="knn_acc", hue="method", col="scenario", kind="line", marker="o", height=3.5, aspect=1.1)
    for ax in g.axes.flat:
        ax.axhline(1/3, ls="--", color="gray", lw=1)
    g.set_titles("Scenario {col_name}"); g.set_axis_labels("k", "k-NN accuracy")
    plt.tight_layout(); plt.savefig(FIGS / "fig5_knn_accuracy_curves_s3_s5.png", dpi=300); plt.close()

    corr_plot = t5[["scenario", "delta_~_openai", "delta_~_spacy", "openai_~_spacy"]].melt(id_vars="scenario", var_name="pair", value_name="r")
    plt.figure(figsize=(8, 4.0))
    sns.lineplot(data=corr_plot, x="scenario", y="r", hue="pair", marker="o")
    plt.ylabel("Pearson r (lower triangle)")
    plt.title("Inter-Representation Matrix Correlations by Scenario")
    plt.tight_layout(); plt.savefig(FIGS / "fig6_matrix_correlations_mantel.png", dpi=300); plt.close()

    # figure 1b
    plt.figure(figsize=(7.2, 4.0))
    sns.lineplot(data=t2, x="scenario", y="within_author_delta", marker="o", label="within")
    sns.lineplot(data=t2, x="scenario", y="between_author_delta", marker="o", label="between")
    plt.title("Realized Delta Geometry Across Scenarios")
    plt.tight_layout(); plt.savefig(FIGS / "fig1b_realized_delta_geometry.png", dpi=300); plt.close()

    # figure 1 combined
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    sns.lineplot(data=t1, x="scenario", y="intra_overlap", marker="o", label="intra", ax=axes[0])
    sns.lineplot(data=t1, x="scenario", y="inter_overlap", marker="o", label="inter", ax=axes[0])
    axes[0].set_ylim(0, 1); axes[0].set_title("(a) Design overlap targets")
    sns.lineplot(data=t2, x="scenario", y="within_author_delta", marker="o", label="within", ax=axes[1])
    sns.lineplot(data=t2, x="scenario", y="between_author_delta", marker="o", label="between", ax=axes[1])
    axes[1].set_title("(b) Realized Delta geometry")
    fig.tight_layout(); fig.savefig(FIGS / "fig1_design_targets_and_realized_geometry.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    print(f"Extended tables: {TABLES}")
    print(f"Extended figures: {FIGS}")


def main() -> None:
    build_extended()


if __name__ == "__main__":
    main()
