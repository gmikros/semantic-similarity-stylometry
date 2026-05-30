# Mikros & Čech — Stylometry vs. Embeddings (Qualico 2025)

Reconstructed analysis workspace for the authorship-attribution experiments on the
controlled English corpus stored one level up in `../Corpus` and `../Data`.

The corpus consists of synthetic, author-controlled texts (groups **A**, **B**, **C**)
arranged into several **scenarios**. Each scenario is represented by 60 documents
(3 groups × 20 samples). The study compares two families of document representations:

1. **Classic stylometry** — Most-Frequent-Word (MFW) profiles with **Cosine Delta**
   (a.k.a. the *Würzburg* distance) computed with the R package
   [`stylo`](https://github.com/computationalstylistics/stylo).
2. **Dense embeddings** — document vectors from
   **OpenAI `text-embedding-3-small`** (1536-d) and **spaCy `en_core_web_lg`** (300-d),
   compared with cosine distance.

## Scenarios

| Scenario | Corpus form              | Docs | MFW used | Notes                          |
|----------|--------------------------|------|----------|--------------------------------|
| 1        | 3 group texts `A/B/C.txt`| 60   | 500      | split into 20 chunks per group |
| 2        | 3 group texts `A/B/C.txt`| 60   | 500      | split into 20 chunks per group |
| 3        | 60 ready files           | 60   | 500      | distinct texts (formerly `3a`) |
| 4        | 60 ready files           | 60   | 393      | identical-text control by design |
| 5        | 60 ready files           | 60   | 500      |                                |
| 3_identical_control | 60 ready files  | 60   | 500      | preserved degenerate ex-`Scenario 3` (3 unique texts) |

Document ids follow the pattern `Sc{N}_{GROUP}_{i}` (e.g. `Sc3_A_1`).

## What the original `../Data` files are

| File pattern                                             | Produced by                          |
|----------------------------------------------------------|--------------------------------------|
| `Scenario N_table_with_frequencies.txt`                  | `stylo` MFW relative-frequency table |
| `Scenario N_distance_table_<k>mfw_0c.csv/.txt`           | `stylo` Cosine-Delta distance table  |
| `Scenario N_Documents_CA_<k>_MFWs_Culled_0__wurzburg_EDGES.csv` | `stylo` bootstrap consensus network edges |
| `Wide_Format_Distance_Matrix_for_Scenario_N.csv`         | comma/wide reshape of the distance table |
| `stylo_features_Scenario N.xlsx`                         | MFW feature table (Excel)            |
| `Scenario N_Distances.xlsx`                              | distance table (Excel)               |
| `Sc{N}_OpenAI_average_embeddings.csv`                    | OpenAI `text-embedding-3-small`, averaged |
| `Sc{N}_Spacy_embeddings.csv`                             | spaCy `en_core_web_lg` doc vectors   |

> Note: `Wide_Format_Distance_Matrix_for_Scenario_N.csv` is numerically identical to
> the `stylo` `distance_table_<k>mfw_0c` file — it is the same Cosine-Delta matrix in a
> comma-separated wide layout, **not** an embedding distance matrix.

## Pipeline

```
1. chunk_corpus.py        split A/B/C group texts (scenarios 1, 2) into 20 chunks each
2. embed_spacy.py         spaCy en_core_web_lg document vectors      -> Sc{N}_Spacy_embeddings.csv
3. embed_openai.py        OpenAI text-embedding-3-small (averaged)   -> Sc{N}_OpenAI_average_embeddings.csv
4. embedding_distances.py cosine distance matrices from embeddings   -> *_embedding_distance_*.csv
5. R/stylo_pipeline.R     500-MFW Cosine Delta: freq table, distance table, consensus network edges
6. reshape_distances.py   wide-format reshape of any stylo distance table
7. verify_outputs.py      compare reconstructed outputs against ../Data
8. classify.py            attribution comparison (stylometry vs embeddings): 1NN-LOO + ARI
```

## Setup

> **Important (this machine):** the project lives under Dropbox, which locks files and
> breaks `pip` installs into a venv created *inside* the synced folder. Create the venv
> **outside** Dropbox. The environment used here is
> `C:\Users\USER01\.venvs\qualico_mikros_cech`.

```powershell
python -m venv C:\Users\USER01\.venvs\qualico_mikros_cech
& C:\Users\USER01\.venvs\qualico_mikros_cech\Scripts\python.exe -m pip install -r requirements.txt
& C:\Users\USER01\.venvs\qualico_mikros_cech\Scripts\python.exe -m spacy download en_core_web_lg
```

The OpenAI step needs an API key:

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

The R step needs R with `stylo` **and** `networkD3` (the latter is required by
`stylo.network` for the EDGES/NODES export). `Rscript` was not on PATH on this machine;
it lives at `C:\Program Files\R\R-4.3.1\bin\Rscript.exe`.

```r
install.packages(c("stylo", "networkD3"))
```

## Run

```bash
python src/run_all.py --scenarios 1 2 3 3a 4 5
# or individual steps, e.g.:
python src/embed_spacy.py --scenario 3
python src/embedding_distances.py --scenario 3 --backend spacy
Rscript R/stylo_pipeline.R 3 500
python src/verify_outputs.py --scenario 3
```

All reconstructed artifacts are written under `analysis/output/` and never overwrite the
original `../Data` files.

## Reproduction status (Cosine-Delta distance tables)

Regenerated `stylo` distance tables vs. the originals in `../Data`
(per-document corpus, Cosine Delta, 0 culling):

| Scenario | n  | max abs diff | Pearson | Status                          |
|----------|----|--------------|---------|---------------------------------|
| 1        | 60 | 0.000000     | 1.00000 | exact                           |
| 2        | 60 | 0.027        | 0.99994 | matches within rounding         |
| 3        | 60 | 0.000000     | 1.00000 | exact (after folder fix, below) |
| 4        | 60 | 0.000000     | 1.00000 | exact                           |
| 5        | 60 | 0.000000     | 1.00000 | exact                           |

**Scenario 3 — fixed.** The folder originally named `Corpus/Scenario 3` had been
overwritten with a degenerate *identical-text control* (only 3 unique texts, each
replicated 20× → 224 word-types), which could not reproduce the saved `500mfw` table.
The genuine Scenario 3 source texts survived in the folder `Corpus/Scenario 3a`. We
therefore:

- renamed the degenerate folder to **`Corpus/Scenario 3_identical_control`** (preserved), and
- promoted **`Corpus/Scenario 3a` → `Corpus/Scenario 3`**.

Regenerating from the restored `Scenario 3` now reproduces
`Scenario 3_distance_table_500mfw_0c.csv` **exactly** (max abs diff `0.000000`,
Pearson `1.0`).

Note that **Scenario 4 is itself an identical-text control by design** (3 unique texts,
exactly 393 word-types — matching its original `393mfw` filename), and its original
distance table reproduces exactly, so it was left unchanged.

## Attribution comparison: stylometry vs. embeddings

`src/classify.py` evaluates how well the three author groups are recovered by each
representation — **delta** (stylo Cosine Delta), **openai** (`text-embedding-3-small`),
and **spacy** (`en_core_web_lg`) — using leave-one-out 1-NN accuracy and the Adjusted
Rand Index of a 3-cluster agglomerative clustering. Results
(`output/classification_summary.csv`):

| Scenario | delta 1NN | openai 1NN | spacy 1NN | Notes |
|----------|-----------|------------|-----------|-------|
| 1        | 1.00 | 1.00 | 1.00 | chunks of 3 group texts — trivially separable |
| 2        | 1.00 | 1.00 | 1.00 | chunks of 3 group texts — trivially separable |
| 3        | **0.83** | **0.92** | **0.87** | *real* distinct-text task — **OpenAI > spaCy > Delta** |
| 4        | 1.00 | 1.00 | 1.00 | identical-text control |
| 5        | 0.00 | 0.00 | 0.00 | every doc's nearest neighbour is in another group |

(The preserved `Scenario 3_identical_control` folder is excluded from `ALL_SCENARIOS`;
pass it explicitly, e.g. `python src/classify.py --scenarios 3_identical_control`, to
score the degenerate control — it returns 1.00 trivially.)

Interpretation:
- The only genuinely discriminative attribution task is **Scenario 3**, where
  **OpenAI embeddings outperform** both spaCy and classic Cosine Delta (0.92 vs 0.87 vs 0.83).
- **Scenario 5 scores 0%** for *every* method: nearest neighbours are consistently
  cross-group. The group A/B/C texts are topic-aligned across groups (e.g. each group
  contains a text about the same place/topic), so similarity tracks topic rather than
  author — a clean illustration of the topic confound that defeats both stylometry and
  embeddings when labels are not the dominant axis of variation.
- Scenarios 1–4 are either trivial (chunked single sources) or identical-text controls,
  so their perfect scores are expected and not evidence of method quality.

## Findings about the original `../Data` embedding files

While reconstructing, two issues surfaced in the *original* embedding CSVs (verify with
`python src/verify_outputs.py` after generating fresh files):

- **Group-level, not document-level.** `../Data/Sc3_Spacy_embeddings.csv` contains only
  **3 unique vectors** (one per group A/B/C), each replicated 20 times;
  `../Data/Sc3_OpenAI_average_embeddings.csv` has only **20 unique rows** out of 60. The
  original embeddings were effectively computed by averaging *all* of a group's text into
  a single vector, so embedding-based **within-group distances are degenerate (≈0)**.
  (This is consistent with those `../Data/Sc3_*` files having been built from the
  identical-text corpus now preserved as `Scenario 3_identical_control`, not from the
  distinct-text Scenario 3 that produced the `../Data` *distance* table.)
- **Mislabelled.** The `Author` column in `../Data/Sc3_*` (and likely other scenarios)
  reads `Sc1_A/B/C` — a copy/paste leftover from scenario 1.

This pipeline defaults to **per-document** embeddings (60 distinct vectors), which is the
intended granularity for document-level distance/clustering. Pass
`--granularity group` to `embed_spacy.py` / `embed_openai.py` to reproduce the original
group-level behaviour for comparison.

## Reproduction decisions

These choices reproduce the original `../Data` artifacts; adjust in `src/config.py` if needed.

- **OpenAI model**: `text-embedding-3-small` (1536-d, matches the 1536 `WE*` columns).
- **spaCy model**: `en_core_web_lg` (English; 300-d `doc.vector`, matches `WE1..WE300`).
- **OpenAI "average" embedding**: each document is segmented (sentence-batched), every
  segment is embedded, and the segment vectors are averaged into one document vector.
- **Stylometry**: Cosine Delta (`distance = "wurzburg"`), 500 MFW (393 for scenario 4),
  0% culling, on lowercased word 1-grams — the `stylo` defaults implied by the filenames.
- **Embedding label column** is named `Author` and holds the group label (`Sc{N}_{GROUP}`)
  to match the originals; a `doc_id` column is also written for traceability.
