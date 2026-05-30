"""Document embeddings from spaCy ``en_core_web_lg`` (300-d).

For each document, ``Doc.vector`` (the average of its token vectors) is used as the
document representation, reproducing ``Sc{N}_Spacy_embeddings.csv``.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config
from corpus import load_documents


def _vector(nlp, text: str) -> np.ndarray:
    vec = nlp(text).vector
    if vec.shape[0] != config.SPACY_DIM:
        raise RuntimeError(
            f"Expected {config.SPACY_DIM}-d spaCy vector, got {vec.shape[0]}. "
            f"Is '{config.SPACY_MODEL}' installed (it ships word vectors)?"
        )
    return vec


def embed_scenario(scenario: str, nlp, granularity: str = "document") -> pd.DataFrame:
    """Build a 60-row embedding table.

    granularity="document" -> one (distinct) vector per document  [recommended]
    granularity="group"    -> one vector per group (all group text averaged),
                              replicated across the group's documents
                              (reproduces the degenerate original files).
    """
    docs = load_documents(scenario)
    rows, labels, ids = [], [], []

    if granularity == "group":
        from collections import defaultdict
        texts_by_group: dict[str, list[str]] = defaultdict(list)
        for doc in docs:
            texts_by_group[doc.group].append(doc.text())
        group_vec = {g: _vector(nlp, "\n".join(t)) for g, t in texts_by_group.items()}
        for doc in docs:
            rows.append(group_vec[doc.group])
            labels.append(doc.group_label)
            ids.append(doc.doc_id)
    else:
        for doc in docs:
            rows.append(_vector(nlp, doc.text()))
            labels.append(doc.group_label)
            ids.append(doc.doc_id)

    cols = [f"WE{i}" for i in range(1, config.SPACY_DIM + 1)]
    frame = pd.DataFrame(np.vstack(rows), columns=cols)
    frame.insert(0, "Author", labels)
    frame.insert(1, "doc_id", ids)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--granularity", choices=["document", "group"], default="document",
                        help="'document' (per-doc, recommended) or 'group' (reproduce original)")
    args = parser.parse_args()

    import spacy
    try:
        nlp = spacy.load(config.SPACY_MODEL, disable=["parser", "ner", "tagger", "lemmatizer"])
    except OSError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"spaCy model '{config.SPACY_MODEL}' not found. Install it with:\n"
            f"    python -m spacy download {config.SPACY_MODEL}"
        ) from exc

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    frame = embed_scenario(args.scenario, nlp, granularity=args.granularity)
    out = config.OUTPUT_DIR / f"Sc{args.scenario}_Spacy_embeddings.csv"
    frame.to_csv(out, index=False)
    print(f"Wrote {out} ({frame.shape[0]} docs x {config.SPACY_DIM} dims)")


if __name__ == "__main__":
    main()
