"""Document embeddings from OpenAI ``text-embedding-3-small`` (1536-d), averaged.

Long documents are split into character-bounded segments; each segment is embedded and
the per-segment vectors are averaged into a single document vector, reproducing
``Sc{N}_OpenAI_average_embeddings.csv``.

Requires the ``OPENAI_API_KEY`` environment variable.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from corpus import Document, load_documents


def segment(text: str, max_chars: int) -> list[str]:
    """Split text into <= max_chars segments without breaking sentences where possible."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else [" "]
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    segments: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 > max_chars and buf:
            segments.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        segments.append(buf.strip())
    return segments or [text[:max_chars]]


def embed_text(client, text: str) -> np.ndarray:
    segments = segment(text, config.OPENAI_SEGMENT_CHARS)
    resp = client.embeddings.create(model=config.OPENAI_MODEL, input=segments)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float64)
    return vectors.mean(axis=0)


def embed_document(client, doc: Document) -> np.ndarray:
    return embed_text(client, doc.text())


def embed_scenario(scenario: str, client, granularity: str = "document") -> pd.DataFrame:
    """granularity="document" (per-doc, recommended) or "group" (reproduce original)."""
    docs = load_documents(scenario)
    rows, labels, ids = [], [], []

    if granularity == "group":
        from collections import defaultdict
        texts_by_group: dict[str, list[str]] = defaultdict(list)
        for doc in docs:
            texts_by_group[doc.group].append(doc.text())
        group_vec = {g: embed_text(client, "\n".join(t))
                     for g, t in texts_by_group.items()}
        for doc in docs:
            rows.append(group_vec[doc.group])
            labels.append(doc.group_label)
            ids.append(doc.doc_id)
    else:
        for doc in tqdm(docs, desc=f"OpenAI Sc{scenario}"):
            rows.append(embed_document(client, doc))
            labels.append(doc.group_label)
            ids.append(doc.doc_id)

    cols = [f"WE{i}" for i in range(1, config.OPENAI_DIM + 1)]
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

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set the OPENAI_API_KEY environment variable to run this step.")

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    frame = embed_scenario(args.scenario, client, granularity=args.granularity)
    out = config.OUTPUT_DIR / f"Sc{args.scenario}_OpenAI_average_embeddings.csv"
    frame.to_csv(out, index=False)
    print(f"Wrote {out} ({frame.shape[0]} docs x {config.OPENAI_DIM} dims)")


if __name__ == "__main__":
    main()
