#!/usr/bin/env python3
"""
generate_scenario.py -- rebuild a controlled-overlap scenario for the
Mikros & Cech semantic-similarity study.

Each document is assembled as:   [ TOPIC block ] + [ AUTHOR block ]
  * TOPIC block  is shared by the three authors at the SAME topic index
                 -> controls BETWEEN-author (inter) overlap
  * AUTHOR block is shared across one author's 20 documents
                 -> controls WITHIN-author (intra) overlap

The split is set by the design ratio (intra, inter), intra + inter = 1:
      len(author_block) = round(intra * L)
      len(topic_block)  = round(inter * L)

This reproduces the construction recovered from the surviving scenarios
(Sc1 = pure author block, intra 0.9; Sc5 = big topic block, intra 0.1).
Blocks are harvested verbatim from the existing corpus so the prose stays
natural and LLM-generated; only the split ratio changes between scenarios.

Example (regenerate Scenario 4 at intra 0.3 / inter 0.7):
    python generate_scenario.py --intra 0.3 --inter 0.7 --length 300 \
           --topic-src 5 --author-src 3 --author-mode voice \
           --out "../../Corpus/Scenario 4_regenerated"
"""
import argparse, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.normpath(os.path.join(ROOT, "..", "..", "Corpus"))

def read(p):
    return open(p, encoding="utf-8", errors="ignore").read().strip()

def sentence_trim(words, target):
    """Trim a word list to ~target words, ending on a sentence boundary."""
    if len(words) <= target:
        return words
    for j in range(target, min(len(words), target + 40)):
        if re.search(r'[.!?]"?$', words[j]):
            return words[:j + 1]
    for j in range(target - 1, max(0, target - 40), -1):
        if re.search(r'[.!?]"?$', words[j]):
            return words[:j + 1]
    return words[:target]

def lcp_words(strs):
    seqs = [s.split() for s in strs]
    n = min(len(s) for s in seqs)
    k = 0
    for j in range(n):
        if len({s[j] for s in seqs}) == 1:
            k += 1
        else:
            break
    return seqs[0][:k]

def getf(sc, g, i):
    for c in ["Scenario %s/output/Sc%s_%s_%s.txt" % (sc, sc, g, i),
              "Scenario %s/%s_%s.txt" % (sc, g, i),
              "Scenario %s/Sc%s_%s_%s.txt" % (sc, sc, g, i)]:
        p = os.path.join(CORPUS, c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Sc%s %s_%s" % (sc, g, i))

def harvest_topic_blocks(src_scenario=5):
    """20 topic blocks = verbatim prefix shared by A_i, B_i, C_i."""
    return {i: lcp_words([read(getf(src_scenario, g, i)) for g in "ABC"])
            for i in range(1, 21)}

def harvest_author_blocks(src_scenario=1, mode="prefix"):
    """3 author blocks, one per author, constant across that author's texts.

    mode 'prefix' : verbatim prefix shared across an author's 20 texts (Sc1) ->
                    STRONG topical author block -> high author signal.
    mode 'voice'  : author-style remainder of one text after the block common to
                    all three authors (Sc3) -> MILD author-voice signature, the
                    register used for the Sc5 signatures.  Recommended for
                    overlap scenarios so the dominant shared signal is the topic.
    """
    blocks = {}
    if mode == "prefix":
        for g in "ABC":
            blocks[g] = lcp_words([read(getf(src_scenario, g, i)) for i in range(1, 21)])
    elif mode == "voice":
        glob = lcp_words([read(getf(src_scenario, g, 1)) for g in "ABC"])
        for g in "ABC":
            blocks[g] = read(getf(src_scenario, g, 1)).split()[len(glob):]
    else:
        raise ValueError("unknown author mode: %s" % mode)
    return blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra", type=float, required=True)
    ap.add_argument("--inter", type=float, required=True)
    ap.add_argument("--length", type=int, default=300)
    ap.add_argument("--topic-src", type=int, default=5)
    ap.add_argument("--author-src", type=int, default=3)
    ap.add_argument("--author-mode", choices=["prefix", "voice"], default="voice")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    L = args.length
    n_author = round(args.intra * L)
    n_topic = round(args.inter * L)
    topics = harvest_topic_blocks(args.topic_src)
    authors = harvest_author_blocks(args.author_src, args.author_mode)

    os.makedirs(args.out, exist_ok=True)
    written = 0
    for g in "ABC":
        a_block = sentence_trim(authors[g], n_author)
        for i in range(1, 21):
            t_block = sentence_trim(topics[i], n_topic)
            doc = " ".join(t_block) + " " + " ".join(a_block)
            with open(os.path.join(args.out, "%s_%s.txt" % (g, i)), "w", encoding="utf-8") as fh:
                fh.write(doc.strip() + "\n")
            written += 1
    print("Wrote %d documents to %s" % (written, args.out))
    print("  L=%d: topic block ~%dw (inter=%.2f), author block ~%dw (intra=%.2f)"
          % (L, n_topic, args.inter, n_author, args.intra))

if __name__ == "__main__":
    main()
