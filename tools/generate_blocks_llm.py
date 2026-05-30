#!/usr/bin/env python3
"""
generate_blocks_llm.py -- generate the building blocks for a controlled synthetic
corpus with an LLM, following the prompt templates in docs/prompt_templates.md.

It writes, into --out:
    author_A.txt, author_B.txt, author_C.txt   (within-author voice blocks)
    topic_1.txt ... topic_{n_topics}.txt        (between-author shared scene blocks)

Those blocks are then assembled into a 60-document corpus by
analysis/src/generate_scenario.py --blocks-dir <out> ...

This tool is path-clean and reproducible given an OpenAI API key
(set the OPENAI_API_KEY environment variable). Output is non-deterministic unless the
chosen model/params make it so.

Example:
    python tools/generate_blocks_llm.py --intra 0.5 --inter 0.5 --length 300 \
           --n-topics 20 --out blocks/scenario_new
    python analysis/src/generate_scenario.py --intra 0.5 --inter 0.5 --length 300 \
           --blocks-dir blocks/scenario_new --out "Corpus/Scenario new"
"""
from __future__ import annotations

import argparse
import os
import re
import sys

SYSTEM_PROMPT = (
    "You are generating building blocks for a controlled stylometric corpus in English.\n"
    "Rules:\n"
    "- Output ONLY the requested text block(s), no titles, labels, or commentary.\n"
    "- Use natural, fluent prose. No lists, no markdown, no headings.\n"
    "- Never mention author identities, labels (A/B/C), or the word \"author\" inside a "
    "SHARED/topic block.\n"
    "- Hit the requested word count within +/- 10%.\n"
    "- Keep register and vocabulary self-consistent within each block."
)

DEFAULT_VOICES = [
    "Voice 1 (Author A): introspective, emotional, reflective; longer sentences, hypotaxis.",
    "Voice 2 (Author B): epic/adventurous, high-energy; vivid verbs, momentum.",
    "Voice 3 (Author C): analytical, measured, precise; nominal style, restrained tone.",
]


def author_prompt(author_target: int, voices: list[str]) -> str:
    return (
        f"Produce {len(voices)} distinct author-style paragraphs, each {author_target} "
        "words, each in a clearly different and internally consistent literary VOICE. "
        "Differentiate the voices by sentence length, syntactic complexity, function-word "
        "habits, punctuation, and rhythm -- NOT by topic. Do not name any topic or place; "
        "describe nothing concrete.\n\n"
        + "\n".join(voices)
        + "\n\nReturn the paragraphs separated by a line containing only \"---\"."
    )


def topic_prompt(n_topics: int, shared_target: int) -> str:
    return (
        f"Produce {n_topics} self-contained descriptive scene paragraphs, each "
        f"{shared_target} words, each about a DIFFERENT concrete subject (a place, season, "
        "object, or event). Keep a neutral narrative register so the same paragraph could "
        "plausibly sit inside any author's work. Do not reference people by name, do not "
        "use first person, and do not describe a writing style.\n\n"
        f"Number them 1..{n_topics}. Separate paragraphs with a line containing only \"---\"."
    )


def call_llm(client, model: str, prompt: str, temperature: float) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def split_blocks(text: str) -> list[str]:
    parts = re.split(r"(?m)^\s*-{3,}\s*$", text.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = re.sub(r"^\s*\d+[.)]\s*", "", p)  # strip leading "1." / "1)"
        out.append(" ".join(p.split()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intra", type=float, required=True)
    ap.add_argument("--inter", type=float, required=True)
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--n-topics", type=int, default=20)
    ap.add_argument("--model", default="gpt-4o-mini",
                    help="Any OpenAI chat model (default: gpt-4o-mini).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="Directory to write the blocks into.")
    args = ap.parse_args()

    if abs((args.intra + args.inter) - 1.0) > 1e-6:
        sys.exit("intra + inter must equal 1.0")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set the OPENAI_API_KEY environment variable to run this tool.")

    author_target = round(args.intra * args.length)
    shared_target = round(args.inter * args.length)

    from openai import OpenAI
    client = OpenAI()

    os.makedirs(args.out, exist_ok=True)

    # 1) Author voice blocks
    authors_raw = call_llm(client, args.model, author_prompt(author_target, DEFAULT_VOICES),
                           args.temperature)
    author_blocks = split_blocks(authors_raw)
    if len(author_blocks) < 3:
        sys.exit(f"Expected 3 author blocks, parsed {len(author_blocks)}. Raw output:\n{authors_raw}")
    for g, block in zip("ABC", author_blocks[:3]):
        with open(os.path.join(args.out, f"author_{g}.txt"), "w", encoding="utf-8") as fh:
            fh.write(block.strip() + "\n")

    # 2) Shared topic blocks
    topics_raw = call_llm(client, args.model, topic_prompt(args.n_topics, shared_target),
                          args.temperature)
    topic_blocks = split_blocks(topics_raw)
    if len(topic_blocks) < args.n_topics:
        sys.exit(f"Expected {args.n_topics} topic blocks, parsed {len(topic_blocks)}. "
                 f"Raw output:\n{topics_raw}")
    for i, block in enumerate(topic_blocks[: args.n_topics], start=1):
        with open(os.path.join(args.out, f"topic_{i}.txt"), "w", encoding="utf-8") as fh:
            fh.write(block.strip() + "\n")

    print(f"Wrote 3 author blocks (~{author_target}w) and {args.n_topics} topic blocks "
          f"(~{shared_target}w) to {args.out}")
    print("Next: assemble with analysis/src/generate_scenario.py --blocks-dir "
          f"{args.out} --intra {args.intra} --inter {args.inter} --length {args.length} "
          "--out \"Corpus/Scenario new\"")


if __name__ == "__main__":
    main()
