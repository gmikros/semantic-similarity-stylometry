# Investigating the Impact of Semantic Similarity on Stylometric Attribution

[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data%20license-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Made with stylo](https://img.shields.io/badge/stylometry-stylo%20(R)-success.svg)](https://github.com/computationalstylistics/stylo)

> Controlled-corpus benchmark comparing classical **Cosine-Delta** stylometry with
> **OpenAI** and **spaCy** embeddings for authorship attribution under systematically
> varied semantic similarity between authors.

Reproducibility package for:

> Mikros, G., Čech, R., Mutlová, P., & Mačutek, J. (2026). *Investigating the Impact of
> Semantic Similarity on Stylometric Attribution Using Controlled Artificial Texts and
> Delta Distances.* (Manuscript, submitted to the *Journal of Quantitative Linguistics*.)

## Contents

- [Headline result](#headline-result)
- [Repository layout](#repository-layout)
- [The five conditions (+ control)](#the-five-conditions--control)
- [Reproducing the results](#reproducing-the-results)
- [Methods in brief](#methods-in-brief)
- [Citing](#citing)
- [Licensing](#licensing)

This repository contains the controlled artificial corpora, the analysis code, and the
exact tables and figures behind the paper. It compares classical distance-based
stylometry (**Cosine Delta**, a.k.a. the Würzburg distance) against two dense embedding
representations (**OpenAI `text-embedding-3-small`** and **spaCy `en_core_web_lg`**) under
systematically varied semantic similarity between authors.

## Headline result

Five controlled conditions span an author-dominated to topic-dominated similarity
continuum. Attribution is perfect while authors stay internally coherent (Scenarios 1, 2),
becomes graded in the balanced condition (Scenario 3, where embeddings edge out Delta),
and collapses for all methods under strong topic confounding (Scenario 5), where Cosine
Delta nonetheless retains recoverable author structure beyond the nearest neighbour.

## Repository layout

```
.
├── Corpus/                     # input corpora (5 scenarios + 1 identical-text control)
│   ├── Scenario 1/ ... 5/      # 60 documents each (authors A/B/C x 20)
│   └── Scenario 3_identical_control/
├── Data/                       # reference outputs (original stylo distance tables,
│                               #   frequency tables, and embedding CSVs)
├── analysis/
│   ├── src/                    # Python pipeline (see analysis/README.md)
│   ├── R/                      # stylo_pipeline.R (Cosine Delta + consensus network)
│   ├── output/                 # regenerated artifacts (embeddings, distances, stylo)
│   ├── paper_package_jql/      # paper tables, figures, significance tests, draft
│   ├── paper_package_jql_extended/   # extended tables/figures (kNN, silhouette, Mantel)
│   └── requirements.txt        # Python dependencies
├── paper/                      # manuscript (Methods/Results) and abstract
├── environment/                # R dependency notes
├── reproduce.ps1 / reproduce.sh  # convenience end-to-end driver
├── LICENSE                     # MIT (code)
├── LICENSE-DATA                # CC BY 4.0 (corpora, data, figures)
└── CITATION.cff
```

## The five conditions (+ control)

| Scenario | Intra/Inter overlap | Unique texts | MFW | Role |
|----------|---------------------|--------------|-----|------|
| 1 | 0.90 / 0.10 | 60 | 500 | high-separation baseline |
| 2 | 0.70 / 0.30 | 60 | 500 | high separation maintained |
| 3 | 0.50 / 0.50 | 60 (59 unique) | 500 | balanced, discriminative |
| 4 | 0.30 / 0.70 | 3 | 393 | identical-text control |
| 5 | 0.10 / 0.90 | 60 | 500 | topic confound, collapse |
| 3_identical_control | — | 3 | 500 | preserved degenerate ex-`Scenario 3` |

Documents are named `Sc{N}_{GROUP}_{i}` (Scenarios 1, 3, 5) or `{GROUP}_{i}` (Scenarios 2,
4, and the control); both conventions are handled by the loader. In Scenario 3, two
documents (`A_12`, `A_16`) are identical, so the corpus holds 60 documents but 59 unique
texts. Scenario 4 has only ~400 distinct word types, so its Cosine-Delta profile is
necessarily capped at 393 MFW rather than 500.

### Supplementary: regenerated Scenario 4

`Corpus/Scenario 4_regenerated/` is a non-degenerate, distinct-text version of Scenario 4
(60 unique texts) produced by `analysis/src/generate_scenario.py`. It is **supplementary**
and not part of the five-condition pipeline reported in the paper. The block-architecture
recipe, the exact regeneration command, and the (non-linear) tipping-point finding are
documented in [`docs/scenario4_regeneration.md`](docs/scenario4_regeneration.md).

## Reproducing the results

There are two paths. The repository already ships every precomputed artifact, so you can
reproduce the paper tables/figures **without** an OpenAI key or R. Full regeneration from
the raw corpora additionally needs an OpenAI API key (embeddings) and R with `stylo`
(Cosine Delta).

### 1. Python environment

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
pip install -r analysis/requirements.txt
python -m spacy download en_core_web_lg
```

### 2a. Quick reproduction (uses shipped artifacts; no API key / no R)

```bash
python analysis/src/build_jql_package.py            # paper tables, figures, significance
python analysis/src/build_jql_extended_results.py   # extended tables/figures
```

Outputs are written under `analysis/paper_package_jql/` and
`analysis/paper_package_jql_extended/`.

### 2b. Full regeneration from the corpora

Requires `OPENAI_API_KEY` (for OpenAI embeddings) and R with `stylo` + `networkD3`
(see `environment/R_requirements.txt`). `Rscript` must be on `PATH`.

```bash
export OPENAI_API_KEY=sk-...        # PowerShell: $env:OPENAI_API_KEY = "sk-..."
python analysis/src/run_all.py      # chunk, embeddings, distances, stylo, verify, classify
python analysis/src/build_jql_package.py
python analysis/src/build_jql_extended_results.py
```

Or run the convenience driver: `./reproduce.sh` (or `./reproduce.ps1` on Windows).

### Verifying fidelity

`analysis/src/verify_outputs.py` compares regenerated stylo distance tables against the
reference tables in `Data/`. Scenarios 1, 3, 4, 5 reproduce the saved Cosine-Delta tables
exactly; Scenario 2 matches within rounding.

## Methods in brief

- **Cosine Delta** (`stylo`, `distance = "wurzburg"`): 500 MFW (393 for Scenario 4),
  0 culling, lowercased word unigrams.
- **OpenAI embeddings**: `text-embedding-3-small` (1536-d), document-level averaging.
- **spaCy embeddings**: `en_core_web_lg` (300-d) mean-pooled document vectors.
- **Evaluation**: leave-one-out 1NN (and k-NN for k = 1, 3, 5, 7) attribution accuracy,
  Adjusted Rand Index and silhouette from 3-cluster agglomerative clustering, separation
  ratios, pair-type distance decomposition, and Mantel-style matrix correlations.
- **Inference**: bootstrap CIs, paired randomization tests, exact McNemar tests
  (Scenario 3) and exact binomial tests against chance (Scenario 5).

See `analysis/README.md` for full per-script documentation.

## Citing

If you use this code or data, please cite the paper and this repository (see
`CITATION.cff`).

## Licensing

- **Code** (`analysis/`, scripts): MIT License — see [`LICENSE`](LICENSE).
- **Corpora, data, tables, and figures** (`Corpus/`, `Data/`, generated outputs):
  Creative Commons Attribution 4.0 International (CC BY 4.0) — see
  [`LICENSE-DATA`](LICENSE-DATA).
