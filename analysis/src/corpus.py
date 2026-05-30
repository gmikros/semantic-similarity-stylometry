"""Helpers for discovering and loading the per-document corpus of a scenario."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import config


@dataclass(frozen=True)
class Document:
    scenario: str
    group: str          # "A" / "B" / "C"
    index: int          # 1..20
    path: Path

    @property
    def doc_id(self) -> str:
        return config.doc_id(self.scenario, self.group, self.index)

    @property
    def group_label(self) -> str:
        return config.group_label(self.scenario, self.group)

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")


# Accepts both naming conventions found in the corpus:
#   Sc3_A_1.txt  (scenarios 1, 3, 5)   and   A_1.txt  (scenarios 2, 3a, 4)
_FNAME_RE = re.compile(
    r"^(?:Sc(?P<scn>[0-9a-z]+)_)?(?P<grp>[A-C])_(?P<idx>\d+)\.txt$", re.IGNORECASE
)


def load_documents(scenario: str) -> list[Document]:
    """Return the 60 documents of a scenario sorted by (group, index)."""
    directory = config.scenario_corpus_dir(scenario)
    if not directory.exists():
        raise FileNotFoundError(f"Corpus directory not found: {directory}")

    docs: list[Document] = []
    for path in directory.glob("*.txt"):
        m = _FNAME_RE.match(path.name)
        if not m:
            continue
        docs.append(
            Document(
                scenario=scenario,
                group=m.group("grp").upper(),
                index=int(m.group("idx")),
                path=path,
            )
        )
    docs.sort(key=lambda d: (d.group, d.index))
    if not docs:
        raise RuntimeError(f"No 'Sc*_<group>_<idx>.txt' files found in {directory}")
    return docs
