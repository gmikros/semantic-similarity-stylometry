"""Split the concatenated group texts (scenarios 1 & 2) into 20 chunks per group.

Reproduces the ``Corpus/Scenario {1,2}/output/Sc{N}_{GROUP}_{i}.txt`` files from the
original ``A.txt`` / ``B.txt`` / ``C.txt`` group texts by partitioning each group text
into ``SAMPLES_PER_GROUP`` contiguous, word-balanced chunks.
"""
from __future__ import annotations

import argparse
import re

import config


def split_into_chunks(text: str, n_chunks: int) -> list[str]:
    """Split ``text`` into ``n_chunks`` contiguous chunks balanced by word count."""
    words = text.split()
    if len(words) < n_chunks:
        raise ValueError(f"Text has only {len(words)} words; cannot make {n_chunks} chunks")
    # Balanced sizes: the first (len % n) chunks get one extra word.
    base, extra = divmod(len(words), n_chunks)
    chunks: list[str] = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append(" ".join(words[start:start + size]))
        start += size
    return chunks


def chunk_scenario(scenario: str) -> None:
    base = config.CORPUS_DIR / f"Scenario {scenario}"
    out_dir = base / "output"
    out_dir.mkdir(exist_ok=True)

    for group in config.GROUPS:
        src = base / f"{group}.txt"
        if not src.exists():
            print(f"  ! missing {src.name}, skipping group {group}")
            continue
        text = src.read_text(encoding="utf-8", errors="replace")
        chunks = split_into_chunks(text, config.SAMPLES_PER_GROUP)
        for i, chunk in enumerate(chunks, start=1):
            dest = out_dir / f"Sc{scenario}_{group}_{i}.txt"
            dest.write_text(chunk, encoding="utf-8")
        print(f"  group {group}: wrote {len(chunks)} chunks -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="*", default=list(config.SPLIT_SCENARIOS),
        help="Scenarios to chunk (only split scenarios are meaningful).",
    )
    args = parser.parse_args()
    for scenario in args.scenarios:
        print(f"Scenario {scenario}:")
        chunk_scenario(scenario)


if __name__ == "__main__":
    main()
