"""End-to-end driver for the Mikros & Cech reconstruction pipeline.

Steps per scenario (each can be turned off with a ``--no-*`` flag):
  chunk    -> split A/B/C group texts (scenarios 1 & 2 only)
  spacy    -> spaCy document embeddings
  openai   -> OpenAI averaged embeddings (needs OPENAI_API_KEY)
  dist     -> cosine distance matrices from the original ../Data embeddings
  stylo    -> R stylo Cosine-Delta pipeline (needs Rscript + stylo)
  verify   -> compare reconstructions against ../Data

Example:
  python src/run_all.py --scenarios 3 5 --no-openai
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config

SRC = Path(__file__).resolve().parent
R_SCRIPT = config.ANALYSIS_DIR / "R" / "stylo_pipeline.R"


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="*", default=list(config.ALL_SCENARIOS))
    parser.add_argument("--no-chunk", action="store_true")
    parser.add_argument("--no-spacy", action="store_true")
    parser.add_argument("--no-openai", action="store_true")
    parser.add_argument("--no-dist", action="store_true")
    parser.add_argument("--no-stylo", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-classify", action="store_true")
    args = parser.parse_args()

    py = [sys.executable]
    rscript = shutil.which("Rscript")

    for scn in args.scenarios:
        print(f"\n========== Scenario {scn} ==========")
        if not args.no_chunk and scn in config.SPLIT_SCENARIOS:
            run(py + [str(SRC / "chunk_corpus.py"), "--scenarios", scn])
        if not args.no_spacy:
            run(py + [str(SRC / "embed_spacy.py"), "--scenario", scn])
        if not args.no_openai:
            run(py + [str(SRC / "embed_openai.py"), "--scenario", scn])
        if not args.no_dist:
            for backend in ("openai", "spacy"):
                run(py + [str(SRC / "embedding_distances.py"),
                          "--scenario", scn, "--backend", backend, "--source", "data"])
        if not args.no_stylo:
            if rscript:
                run([rscript, str(R_SCRIPT), scn, str(config.MFW_PER_SCENARIO.get(scn, 500))])
            else:
                print("  ! Rscript not found on PATH; skipping stylo step.")
        if not args.no_verify:
            run(py + [str(SRC / "verify_outputs.py"), "--scenario", scn])

    if not args.no_classify:
        run(py + [str(SRC / "classify.py"), "--scenarios", *args.scenarios])


if __name__ == "__main__":
    main()
