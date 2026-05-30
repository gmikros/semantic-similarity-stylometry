# Prompt templates for generating the controlled synthetic corpora

This note documents how to produce **fresh** LLM-generated texts with the same controlled
structure as the study corpora. It complements `analysis/src/generate_scenario.py`, which
assembles documents from verbatim blocks, and `tools/generate_blocks_llm.py`, which calls
an LLM to produce those blocks.

## Recovered architecture

Every document is two verbatim-reused blocks:

```
document = [ SHARED block ]  +  [ AUTHOR block ]
```

- **SHARED block** — byte-identical across authors at a given topic index. Controls
  **between-author (inter)** overlap.
- **AUTHOR block** — byte-identical across one author's 20 documents. Controls
  **within-author (intra)** overlap.

The overlap ratio is simply the relative length of the two blocks
(`intra + inter = 1`): `len(author_block) ≈ round(intra * L)`,
`len(shared_block) ≈ round(inter * L)`. The LLM only generates the building blocks;
deterministic assembly enforces exact overlap.

| Scenario | intra / inter | SHARED block | AUTHOR block |
|----------|---------------|--------------|--------------|
| 1 | 0.90 / 0.10 | per-author topic essay (~270 w) | short doc-specific tail |
| 3 | 0.50 / 0.50 | shared scene paragraph (~140 w) | ~120-w author voice profile |
| 4 (regenerated) | 0.30 / 0.70 | per-topic scene block (~210 w) | ~90-w author voice |
| 5 | 0.10 / 0.90 | per-topic scene block (~280 w) | ~30-w author signature |

Three author voices: **A** introspective/literary, **B** epic/adventure, **C**
analytical/measured.

## Parameters

`n_authors = 3`, `n_topics = 20`, `L ≈ 300` words/doc, `intra + inter = 1`,
`author_target = round(intra * L)`, `shared_target = round(inter * L)`.

## System prompt (all calls)

```
You are generating building blocks for a controlled stylometric corpus in English.
Rules:
- Output ONLY the requested text block(s), no titles, labels, or commentary.
- Use natural, fluent prose. No lists, no markdown, no headings.
- Never mention author identities, labels (A/B/C), or the word "author" inside a
  SHARED/topic block.
- Hit the requested word count within +/- 10%.
- Keep register and vocabulary self-consistent within each block.
```

## Template 1 — Author voice blocks (within-author signal)

Generate once per author; reused across all 20 of that author's documents.

```
Produce {n_authors} distinct author-style paragraphs, each {author_target} words,
each in a clearly different and internally consistent literary VOICE. Differentiate the
voices by sentence length, syntactic complexity, function-word habits, punctuation, and
rhythm -- NOT by topic. Do not name any topic or place; describe nothing concrete.

Voice 1 (Author A): introspective, emotional, reflective; longer sentences, hypotaxis.
Voice 2 (Author B): epic/adventurous, high-energy; vivid verbs, momentum.
Voice 3 (Author C): analytical, measured, precise; nominal style, restrained tone.

Return the three paragraphs separated by a line containing only "---".
```

## Template 2 — Shared topic/scene blocks (between-author signal)

Generate `n_topics` blocks; each is shared verbatim by all authors at that topic index.

```
Produce {n_topics} self-contained descriptive scene paragraphs, each {shared_target}
words, each about a DIFFERENT concrete subject (a place, season, object, or event).
Keep a neutral narrative register so the same paragraph could plausibly sit inside any
author's work. Do not reference people by name, do not use first person, and do not
describe a writing style.

Number them 1..{n_topics}. Separate paragraphs with a line containing only "---".
```

## Template 3 — One-shot full document (alternative)

Use only if you want whole documents directly; prefer Templates 1+2 plus deterministic
assembly for exact, reproducible overlap.

```
Write one ~{L}-word text for Author {GROUP} on Topic {i}.
- First ~{shared_target} words: reproduce this SHARED block verbatim:
  <<<{shared_block_topic_i}>>>
- Remaining ~{author_target} words: continue in Author {GROUP}'s VOICE, reproducing
  this voice block verbatim:
  <<<{author_block_GROUP}>>>
Output only the concatenated text.
```

## End-to-end workflow

```bash
# 1) Generate blocks with an LLM (needs OPENAI_API_KEY)
python tools/generate_blocks_llm.py --intra 0.5 --inter 0.5 --length 300 \
       --n-topics 20 --out blocks/scenario_new

# 2) Assemble the 60-document corpus from those blocks
python analysis/src/generate_scenario.py --intra 0.5 --inter 0.5 --length 300 \
       --blocks-dir blocks/scenario_new --out "Corpus/Scenario new"

# 3) Validate geometry with the standard pipeline
#    (stylo Cosine Delta + embeddings + classify.py)
```

## Validation targets

Cosine-Delta separation ratio runs from ~3.2 at 0.9/0.1 down to ~1.1 at 0.1/0.9; 1NN
attribution stays at ceiling until the balanced 0.5/0.5 tipping point, then collapses.
Because shared content is organized per topic, difficulty is **non-linear** in the ratio:
set `intra >= inter` for a graded, non-collapsed condition (see
`docs/scenario4_regeneration.md`).
