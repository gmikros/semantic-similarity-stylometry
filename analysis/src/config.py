"""Central configuration for the Mikros & Cech stylometry/embeddings pipeline.

Paths are resolved relative to this file so the workspace can be moved freely as
long as it stays inside the ``Mikros, Cech`` project folder.
"""
from __future__ import annotations

from pathlib import Path

# analysis/src/config.py -> analysis/ -> Mikros, Cech/
ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ANALYSIS_DIR.parent

CORPUS_DIR = PROJECT_DIR / "Corpus"
DATA_DIR = PROJECT_DIR / "Data"           # original artifacts (read-only)
OUTPUT_DIR = ANALYSIS_DIR / "output"      # reconstructed artifacts

GROUPS = ("A", "B", "C")
SAMPLES_PER_GROUP = 20

# Scenarios whose corpus is delivered as three concatenated group texts (A/B/C.txt)
# and must be split into ``SAMPLES_PER_GROUP`` chunks each.
SPLIT_SCENARIOS = ("1", "2")

# Number of Most Frequent Words used by the original stylo runs, per scenario.
#
# Note: the folder originally named "Scenario 3a" held the genuine distinct-text
# Scenario 3 corpus and has been promoted to "Scenario 3" (it reproduces the saved
# Data/Scenario 3 distance table exactly). The earlier degenerate identical-text
# folder is preserved as "Scenario 3_identical_control" for reference.
MFW_PER_SCENARIO = {
    "1": 500,
    "2": 500,
    "3": 500,
    "4": 393,
    "5": 500,
    "3_identical_control": 500,
}

ALL_SCENARIOS = ("1", "2", "3", "4", "5")

# --- Embedding backends ---------------------------------------------------
OPENAI_MODEL = "text-embedding-3-small"   # 1536-d
OPENAI_DIM = 1536
SPACY_MODEL = "en_core_web_lg"            # 300-d
SPACY_DIM = 300

# How many characters per OpenAI request segment when averaging long documents.
OPENAI_SEGMENT_CHARS = 6000

# Stylometry
STYLO_DISTANCE = "wurzburg"               # Cosine Delta
STYLO_CULLING = 0


def scenario_corpus_dir(scenario: str) -> Path:
    """Directory holding the 60 per-document text files for a scenario."""
    base = CORPUS_DIR / f"Scenario {scenario}"
    if scenario in SPLIT_SCENARIOS:
        # split chunks live in the ``output`` subfolder
        return base / "output"
    return base


def doc_id(scenario: str, group: str, idx: int) -> str:
    return f"Sc{scenario}_{group}_{idx}"


def group_label(scenario: str, group: str) -> str:
    return f"Sc{scenario}_{group}"
