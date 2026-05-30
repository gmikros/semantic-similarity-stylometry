# Regenerating Scenario 4 (supplementary)

The original `Corpus/Scenario 4` is an **identical-text control** (3 unique base texts,
~393 distinct word types). This note documents how a non-degenerate, distinct-text
Scenario 4 can be regenerated with `analysis/src/generate_scenario.py`, and what its
geometry actually is. The regenerated corpus lives in `Corpus/Scenario 4_regenerated/`
and is **supplementary**: it is not part of the five-condition pipeline reported in the
paper and is provided for transparency and further experimentation.

## Block architecture (recovered from the surviving scenarios)

Every document is two verbatim blocks concatenated:

```
[ TOPIC block ]  +  [ AUTHOR block ]
```

- **TOPIC block** — shared by the three authors at the *same topic index* (`A_i`, `B_i`,
  `C_i` share it). Controls **between-author (inter)** overlap.
- **AUTHOR block** — held constant across one author's 20 documents. Controls
  **within-author (intra)** overlap.

The overlap ratio is simply the relative length of the two blocks, with
`intra + inter = 1`:

```
len(author_block) = round(intra * L)
len(topic_block)  = round(inter * L)
```

This recipe reproduces the surviving scenarios: Scenario 1 is almost all author block
(intra 0.90), Scenario 5 is a large topic block plus a short author signature (intra 0.10).
Scenario 4 is specified as intra 0.30 / inter 0.70.

## Regeneration command

```bash
cd analysis/src
python generate_scenario.py --intra 0.3 --inter 0.7 --length 300 \
       --topic-src 5 --author-src 3 --author-mode voice \
       --out "../../Corpus/Scenario 4_regenerated"
```

- `--topic-src 5` harvests the 20 topic blocks (verbatim shared prefix of `A_i/B_i/C_i`)
  from Scenario 5.
- `--author-src 3 --author-mode voice` harvests mild author-voice signatures from
  Scenario 3 (rather than Scenario 1's strongly topical author blocks), so the dominant
  shared signal is topic, not an artificially distinctive author register.

To fold it into the analysis, point the pipeline at the folder: stylo 500-MFW Cosine
Delta, then `embed_spacy.py` / `embed_openai.py` / `embedding_distances.py`, then
`classify.py`.

## What the regenerated Scenario 4 shows (the tipping-point finding)

At the designed 0.30 / 0.70 split the corpus is no longer degenerate (60 distinct texts,
non-zero within-author distance), but it does **not** sit at a smooth midpoint between
Scenario 3 (1NN ≈ 0.83) and Scenario 5 (1NN ≈ 0.00). It lands in the **topic-dominant
regime**: 1NN-LOO ≈ 0.08 and ARI ≈ −0.03, while k-NN accuracy recovers fully to 1.00 by
k = 5.

The reason is a sharp tipping point confirmed by an overlap-ratio sweep: when shared
content is organized *per topic*, attribution difficulty is **non-linear** in the overlap
ratio. As soon as inter-overlap exceeds intra-overlap (i.e. past the balanced 0.5/0.5 of
Scenario 3), every document's nearest neighbour becomes its cross-author topic twin and
1NN collapses toward zero. A faithful 0.3/0.7 Scenario 4 is therefore a **milder Scenario
5**, distinguished mainly by faster k-NN recovery, rather than an intermediate between
Scenarios 3 and 5. That non-linearity is itself a reportable result.

## Open design choice

If an *intermediate-difficulty* Scenario 4 (1NN strictly between 0.00 and 0.83) is wanted,
it cannot be obtained simply by setting 0.3/0.7 under the per-topic construction. Options:

1. Tune the construction (e.g. partially de-align the shared block from topic) to target a
   chosen 1NN level.
2. Rebuild Scenarios 2–4 on a single consistent block architecture so the overlap gradient
   is a clean continuous manipulation.
3. Switch to fresh LLM-generated text with an explicit prompt template.
