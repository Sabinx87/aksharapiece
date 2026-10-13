"""Clean raw Nepali text and write tokenizer / pretraining / held-out splits.

Usage (on Kaggle or a laptop):
    python scripts/clean.py --inputs raw/indiccorp_ne.txt raw/ne_wiki.txt \
        --exclude data/raw/wikipedia_en_ne_si_test_sets/*.ne \
        --out data/clean --tok-mb 200 --heldout-mb 50 --pretrain-mb 2000

Inputs can be .txt (one document or sentence per line), .jsonl (field "text"),
or .parquet (column "text"). Files ending in .gz are read transparently.

Steps per line:
  1. normalize()  : NFC, invisible-char removal, vowel-sign typo fixes, spaces
  2. strip URLs, e-mail addresses and HTML tags
  3. filters      : length, Devanagari share, repeated-character junk
  4. exact dedupe : 8-byte hash of the normalized line
  5. leakage      : drop lines whose hash matches any test-set line (--exclude)
Lines are then shuffled with a fixed seed and written in byte budgets:
held-out first, then the tokenizer sample, then pretraining text, so the
held-out set never overlaps the others.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aksharapiece.akshara import normalize  # noqa: E402

URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
HTML_RE = re.compile(r"<[^>]{1,200}>")
LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
DEVA_LETTER_RE = re.compile(r"[ऀ-ॣॱ-ॿ]")
REPEAT_RE = re.compile(r"(.)\1{9,}")  # same char 10+ times in a row
# private-use characters: leftover symbol-font bullets (e.g. U+F06E) that no tokenizer can represent;
# AksharaPiece also uses plane-15 private-use code points internally, so they must not appear in text
PRIVATE_USE_RE = re.compile("[-\U000F0000-\U0010FFFF]")


def read_lines(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    stem = path[:-3] if path.endswith(".gz") else path
    if stem.endswith(".parquet"):
        import pyarrow.parquet as pq  # only needed for parquet input
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t:
                    yield from t.splitlines()
        return
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        if stem.endswith(".jsonl"):
            for row in f:
                try:
                    t = json.loads(row).get("text", "")
                except json.JSONDecodeError:
                    continue
                yield from t.splitlines()
        else:
            yield from f


def h64(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def clean_line(line: str, min_chars: int, min_deva: float, keep_mixed: bool) -> str | None:
    line = HTML_RE.sub(" ", URL_RE.sub(" ", line))
    line = normalize(PRIVATE_USE_RE.sub(" ", line))
    if len(line) < min_chars or REPEAT_RE.search(line):
        return None
    letters = LETTER_RE.findall(line)
    if not letters:
        return None
    deva_share = len(DEVA_LETTER_RE.findall(line)) / len(letters)
    if deva_share < (0.0 if keep_mixed else min_deva):
        return None
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="test-set files whose lines must not appear in training")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--min-deva", type=float, default=0.6, help="minimum share of letters that are Devanagari")
    ap.add_argument("--keep-mixed", action="store_true", help="keep code-mixed / Romanized lines too")
    ap.add_argument("--heldout-mb", type=float, default=50)
    ap.add_argument("--tok-mb", type=float, default=200)
    ap.add_argument("--pretrain-mb", type=float, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    banned = set()
    for pattern in args.exclude:
        for p in glob.glob(pattern):
            for line in read_lines(p):
                banned.add(h64(normalize(line)))
    print(f"excluded test lines: {len(banned):,}")

    seen, kept = set(), []
    stats = dict(read=0, filtered=0, duplicate=0, test_overlap=0)
    for pattern in args.inputs:
        for p in sorted(glob.glob(pattern)):
            for raw in read_lines(p):
                stats["read"] += 1
                line = clean_line(raw, args.min_chars, args.min_deva, args.keep_mixed)
                if line is None:
                    stats["filtered"] += 1
                    continue
                k = h64(line)
                if k in banned:
                    stats["test_overlap"] += 1
                    continue
                if k in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(k)
                kept.append(line)
    random.Random(args.seed).shuffle(kept)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budgets = [("heldout.txt", args.heldout_mb), ("tokenizer_train.txt", args.tok_mb), ("pretrain.txt", args.pretrain_mb)]
    i, written = 0, {}
    for name, mb in budgets:
        limit, size = int(mb * 1e6), 0
        with open(out / name, "w", encoding="utf-8") as f:
            while i < len(kept) and size < limit:
                b = kept[i] + "\n"
                f.write(b)
                size += len(b.encode("utf-8"))
                i += 1
        written[name] = round(size / 1e6, 2)
    stats.update(kept=len(kept), written_mb=written)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
