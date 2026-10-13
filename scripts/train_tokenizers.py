"""Train the tokenizers used for language-model training, all on the same text.

    python scripts/train_tokenizers.py --train-file data/clean/tokenizer_train.txt --train-mb 100 \
        --vocab 8000 16000 --out data/tok

Writes, per vocabulary size V:
    plain_bpe_{V//1000}k.json                         (variant A: BPE over characters)
    akshara_bpe_{V//1000}k.aksharapiece.json          (variant B: BPE over aksharas, coverage-capped table)
Coverage defaults follow the sweep on IndicCorp: 99.5% for vocabularies under 16k, 99.9% otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aksharapiece.tokenizer import AksharaPiece, train_plain  # noqa: E402
from pilot import read_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--train-mb", type=float, default=100)
    ap.add_argument("--vocab", nargs="+", type=int, default=[16000])
    ap.add_argument("--model", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--coverage", type=float, default=None, help="override the default coverage")
    ap.add_argument("--out", default=str(ROOT / "data" / "tok"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts = read_train(args.train_file, args.train_mb)
    print(f"training tokenizers on {len(texts):,} lines ({args.train_mb} MB)")
    for v in args.vocab:
        tag = f"{args.model}_{v // 1000}k"
        p_plain, p_ak = out / f"plain_{tag}.json", out / f"akshara_{tag}.aksharapiece.json"
        if not p_plain.exists():
            t0 = time.time()
            train_plain(texts, v, args.model).save(str(p_plain))
            print(f"  plain_{tag}: {time.time() - t0:.0f}s")
        if not p_ak.exists():
            cov = args.coverage if args.coverage is not None else (0.995 if v < 16000 else 0.999)
            t0 = time.time()
            ak = AksharaPiece.train(texts, v, args.model, coverage=cov, rare="split")
            ak.save(str(p_ak))
            print(f"  akshara_{tag}: table {len(ak.table):,} aksharas (coverage {cov:.1%}), {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
