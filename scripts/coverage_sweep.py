"""How big should the akshara table be?

Every akshara in the table takes one vocabulary slot. A full table (~5,000 aksharas on IndicCorp)
leaves few slots for merges at small vocabularies. This sweep keeps only the most frequent aksharas
(by coverage of occurrences) and encodes the rest as consonant cluster + vowel sign.

Reports, per setting:
  tokens/word, words broken (any cut inside an akshara), and the TYPE of cut:
    conjunct  - inside a consonant cluster (next to a virama or nukta): the harmful kind
    vowel     - between a consonant (cluster) and its vowel sign / nasal mark
    byte      - inside a single UTF-8 character (only byte-level tokenizers do this)

    python scripts/coverage_sweep.py --train-file data/clean/tokenizer_train.txt --train-mb 30 \
        --test-file data/clean/heldout.txt --max-test 10000 --vocab 8000 16000 32000
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aksharapiece.akshara import akshara_boundaries  # noqa: E402
from aksharapiece.tokenizer import AksharaPiece, train_plain  # noqa: E402
from audit import AksharaPieceAdapter, HFAdapter, char_to_byte_offsets, load_mono  # noqa: E402
from pilot import read_train  # noqa: E402

VIRAMA_NUKTA_JOINER = set("़्‌‍")
VOWEL_SIGNS = re.compile(r"[ऺऻा-ौॎॏॕ-ॗॢॣऀ-ः]")


def classify_cut(text: str, byte_to_char: dict, cut: int) -> str:
    if cut not in byte_to_char:
        return "byte"
    i = byte_to_char[cut]
    prev_c, next_c = text[i - 1], text[i]
    if prev_c in VIRAMA_NUKTA_JOINER or next_c in VIRAMA_NUKTA_JOINER:
        return "conjunct"
    if VOWEL_SIGNS.match(next_c):
        return "vowel"
    return "other"


def score(adapter, sents):
    tokens = words = 0
    deva_words = 0
    broken = {"any": 0, "conjunct": 0, "vowel": 0, "byte": 0}
    for s in sents:
        spans, _, _ = adapter.spans(s)
        tokens += len(spans)
        words += len(s.split())
        good = akshara_boundaries(s, unit="byte")
        offs = char_to_byte_offsets(s)
        b2c = {b: i for i, b in enumerate(offs)}
        cuts = {c for sp in spans for c in sp} - {0, offs[-1]}
        bad = [(c, classify_cut(s, b2c, c)) for c in cuts if c not in good]
        for m in re.finditer(r"\S+", s):
            if not re.search(r"[ऀ-ॿ]", m.group()):
                continue
            deva_words += 1
            ws, we = offs[m.start()], offs[m.end()]
            kinds = {k for c, k in bad if ws < c < we}
            if kinds:
                broken["any"] += 1
                for k in kinds & {"conjunct", "vowel", "byte"}:
                    broken[k] += 1
    return {"fertility": tokens / words, **{f"words_broken_{k}": v / deva_words for k, v in broken.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--train-mb", type=float, default=30)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--max-test", type=int, default=10000)
    ap.add_argument("--vocab", nargs="+", type=int, default=[8000, 16000, 32000])
    ap.add_argument("--coverage", nargs="+", type=float, default=[0.95, 0.99, 0.995, 0.999, 1.0])
    ap.add_argument("--out", default=str(ROOT / "results" / "coverage_sweep"))
    args = ap.parse_args()

    train = read_train(args.train_file, args.train_mb)
    test = [s for s, _ in load_mono([args.test_file], args.max_test)]
    print(f"train {len(train):,} lines ({args.train_mb} MB) | test {len(test):,} sentences")
    rows = []
    for v in args.vocab:
        t0 = time.time()
        r = {"vocab": v, "setting": "plain BPE", "table": 0, **score(HFAdapter("plain", train_plain(train, v)), test)}
        rows.append(r)
        print(f"  {v:6d}  {'plain BPE':22s} tok/word {r['fertility']:.3f}  broken {100 * r['words_broken_any']:.2f}% "
              f"(conjunct {100 * r['words_broken_conjunct']:.2f}%)  {time.time() - t0:.0f}s")
        for cov in args.coverage:
            t0 = time.time()
            ak = AksharaPiece.train(train, v, "bpe", coverage=None if cov >= 1 else cov, rare="split")
            name = f"Akshara-BPE {'all' if cov >= 1 else f'{cov:.1%}'}"
            r = {"vocab": v, "setting": name, "table": len(ak.table),
                 **score(AksharaPieceAdapter(name, ak), test)}
            rows.append(r)
            print(f"  {v:6d}  {name:22s} table {len(ak.table):5d}  tok/word {r['fertility']:.3f}  "
                  f"broken {100 * r['words_broken_any']:.2f}% (conjunct {100 * r['words_broken_conjunct']:.2f}%, "
                  f"vowel {100 * r['words_broken_vowel']:.2f}%)  {time.time() - t0:.0f}s")
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", args.out + ".csv")


if __name__ == "__main__":
    main()
