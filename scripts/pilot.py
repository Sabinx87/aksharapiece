"""Pilot: AksharaPiece vs plain subword tokenizers trained on identical data.

Default (no arguments): train on FLORES v1 Nepali dev + devtest, test on FLORES v1 test.
This small run (~4k sentences) checks direction, not size, of effects.

At scale (e.g. on Kaggle, after scripts/clean.py):
    python scripts/pilot.py --train-file data/clean/tokenizer_train.txt \
        --test-file data/clean/heldout.txt --max-test 20000 \
        --vocab 8000 16000 32000 --models bpe --out results/pilot_indiccorp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aksharapiece.akshara import normalize  # noqa: E402
from aksharapiece.tokenizer import AksharaPiece, train_plain  # noqa: E402
from audit import AksharaPieceAdapter, HFAdapter, audit, fmt, load_mono, load_pairs  # noqa: E402

D = ROOT / "data" / "raw" / "wikipedia_en_ne_si_test_sets"


def read_train(path, max_mb=None):
    out, size = [], 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(line)
            size += len(line.encode("utf-8"))
            if max_mb and size >= max_mb * 1e6:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", nargs="+", type=int, default=[2000, 4000, 8000])
    ap.add_argument("--models", nargs="+", default=["bpe", "unigram"], choices=["bpe", "unigram"])
    ap.add_argument("--train-file", default=None, help="Nepali text, one sentence/line (default: FLORES dev+devtest)")
    ap.add_argument("--train-mb", type=float, default=None, help="use only the first N MB of the training file")
    ap.add_argument("--test-file", default=None, help="Nepali-only held-out text (default: FLORES test with English)")
    ap.add_argument("--max-test", type=int, default=20000)
    ap.add_argument("--min-akshara-freq", type=int, default=2)
    ap.add_argument("--coverage", type=float, default=None, help="keep aksharas covering this share (e.g. 0.999)")
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot_flores_v1"))
    args = ap.parse_args()

    if args.train_file:
        train = read_train(args.train_file, args.train_mb)
    else:
        train = sorted({normalize(l) for s in ("dev", "devtest")
                        for l in open(D / f"wikipedia.{s}.ne-en.ne", encoding="utf-8") if l.strip()})
    train_set = set(normalize(t) for t in train)
    if args.test_file:
        test = [p for p in load_mono([args.test_file], args.max_test) if p[0] not in train_set]
    else:
        test = [p for p in load_pairs([D / "wikipedia.test.ne-en.ne"], [D / "wikipedia.test.ne-en.en"])
                if p[0] not in train_set]
    mb = sum(len(t.encode("utf-8")) for t in train) / 1e6
    print(f"train: {len(train):,} lines ({mb:.1f} MB)  |  test: {len(test):,} unseen sentences")

    tok_dir = Path(args.out + "_tokenizers")
    tok_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in args.vocab:
        for model in args.models:
            t0 = time.time()
            plain = train_plain(train, v, model)
            plain.save(str(tok_dir / f"A_{model}_{v}.json"))
            ak = AksharaPiece.train(train, v, model, min_akshara_freq=args.min_akshara_freq, coverage=args.coverage)
            ak.save(str(tok_dir / f"B_akshara_{model}_{v}.aksharapiece.json"))
            for a in (HFAdapter(f"A. plain {model.upper()} {v}", plain),
                      AksharaPieceAdapter(f"B. Akshara-{model.upper()} {v}", ak)):
                r = audit(a, test)
                rows.append(r)
                print(f"  {a.name:28s} vocab {a.vocab_size:6d}  tokens/word {r['fertility']:.3f}  "
                      f"breaks {100 * r['akshara_break_rate']:.1f}%  words broken {100 * r['words_broken']:.1f}%"
                      f"  ({time.time() - t0:.0f}s)")
            print(f"  akshara table size: {len(ak.table):,}")

    head = ["Tokenizer", "Vocab", "Tokens/word", "Chars/token", "Akshara breaks", "Words broken",
            "UNK", "Rényi eff.", "Parity vs EN"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt(r)) + " |" for r in rows]
    Path(args.out + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import csv
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
