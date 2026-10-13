"""Tokenize the SAME text with each tokenizer and save token ids for language-model training.

For every tokenizer it writes  <out>/<name>/{train.bin, val.bin, meta.json}
  - ids are uint32; one separator id (= tokenizer vocab size) is placed after every line,
    so every tokenizer sees identical line breaks and the model vocab is vocab_size + 1
  - meta.json records the exact UTF-8 byte count of each split (line bytes + 1 per newline),
    which is what bits-per-byte is divided by, and the unknown-token rate

Tokenizer specs (repeat --tok):
    name=aksharapiece:path/to/file.aksharapiece.json
    name=hf:path/to/tokenizer.json
    name=hub:google/muril-base-cased          (needs internet + transformers)
    name=tiktoken:o200k_base
    name=bytes:

    python scripts/prepare_tokens.py --train-file data/clean/pretrain.txt --train-mb 600 \
        --val-file data/clean/heldout.txt --val-mb 5 --out data/tokens \
        --tok plain_bpe_16k=hf:tok/plain_bpe_16k.json --tok akshara_bpe_16k=aksharapiece:tok/akshara_bpe_16k.aksharapiece.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_ENC = None  # per-process encoder


def build_encoder(spec: str):
    """Return (encode_batch(lines) -> list[list[int]], vocab_size, unk_ids)."""
    kind, _, arg = spec.partition(":")
    if kind == "aksharapiece":
        from aksharapiece.tokenizer import AksharaPiece
        ap = AksharaPiece.load(arg)
        unk = {ap.tok.token_to_id("<unk>")} - {None}
        return (lambda lines: ap.encode_batch_ids(lines, normalized=True)), ap.vocab_size, unk
    if kind == "hf":
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(arg)
        unk = {t.token_to_id(u) for u in ("<unk>", "[UNK]")} - {None}
        return (lambda lines: [e.ids for e in t.encode_batch(lines, add_special_tokens=False)]), t.get_vocab_size(), unk
    if kind == "hub":
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained(arg).backend_tokenizer
        unk = {t.token_to_id(u) for u in ("<unk>", "[UNK]")} - {None}
        return (lambda lines: [e.ids for e in t.encode_batch(lines, add_special_tokens=False)]), t.get_vocab_size(), unk
    if kind == "tiktoken":
        import tiktoken
        e = tiktoken.get_encoding(arg)
        return (lambda lines: e.encode_ordinary_batch(lines)), e.n_vocab, set()
    if kind == "bytes":
        return (lambda lines: [list(l.encode("utf-8")) for l in lines]), 256, set()
    raise ValueError(f"unknown tokenizer kind in {spec!r}")


def _init_worker(spec):
    global _ENC
    _ENC = build_encoder(spec)[0]


def _encode(lines):
    return _ENC(lines)


def read_lines(path, max_mb):
    out, size = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            out.append(line)
            size += len(line.encode("utf-8")) + 1
            if size >= max_mb * 1e6:
                break
    return out, size


def write_split(lines, spec, vocab, unk_ids, path, workers, batch=4000):
    sep = vocab
    n_tok = n_unk = 0
    chunks = [lines[i:i + batch] for i in range(0, len(lines), batch)]
    with open(path, "wb") as f, ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(spec,)) as ex:
        for ids_batch in ex.map(_encode, chunks):
            flat = []
            for ids in ids_batch:
                flat.extend(ids)
                flat.append(sep)
                if unk_ids:
                    n_unk += sum(1 for i in ids if i in unk_ids)
            arr = np.asarray(flat, dtype=np.uint32)
            arr.tofile(f)
            n_tok += len(arr)
    return n_tok, n_unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--train-mb", type=float, default=600)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--val-mb", type=float, default=5)
    ap.add_argument("--tok", action="append", required=True, help="name=kind:path")
    ap.add_argument("--out", default=str(ROOT / "data" / "tokens"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    train, train_bytes = read_lines(args.train_file, args.train_mb)
    val, val_bytes = read_lines(args.val_file, args.val_mb)
    print(f"train: {len(train):,} lines, {train_bytes / 1e6:.1f} MB | val: {len(val):,} lines, {val_bytes / 1e6:.2f} MB")
    for item in args.tok:
        name, _, spec = item.partition("=")
        d = Path(args.out) / name
        if (d / "meta.json").exists():
            print(f"[{name}] already prepared, skipping")
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _, vocab, unk_ids = build_encoder(spec)
        tr_tok, tr_unk = write_split(train, spec, vocab, unk_ids, d / "train.bin", args.workers)
        va_tok, va_unk = write_split(val, spec, vocab, unk_ids, d / "val.bin", args.workers)
        np.asarray([len(l.encode("utf-8")) + 1 for l in val], dtype=np.uint32).tofile(d / "val_line_bytes.bin")
        meta = {"name": name, "spec": spec, "tokenizer_vocab": vocab, "model_vocab": vocab + 1, "sep_id": vocab,
                "train_tokens": tr_tok, "train_bytes": train_bytes, "val_tokens": va_tok, "val_bytes": val_bytes,
                "train_unk_rate": tr_unk / tr_tok, "val_unk_rate": va_unk / va_tok,
                "bytes_per_token": train_bytes / tr_tok}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[{name}] {tr_tok:,} train tokens ({meta['bytes_per_token']:.2f} bytes/token), "
              f"{va_tok:,} val tokens, unk {100 * meta['val_unk_rate']:.3f}%  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
