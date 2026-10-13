"""Tokenizer audit for Nepali (Table 1 of the plan).

Measures, on parallel Nepali-English sentences:
  fertility            Nepali tokens per whitespace word                (lower is better)
  chars_per_token      Unicode code points per token                    (higher is better)
  bytes_per_token      UTF-8 bytes per token                            (higher is better)
  akshara_break_rate   share of token boundaries that fall INSIDE an akshara (lower; 0 is ideal)
  words_broken         share of Devanagari words with at least one broken akshara
  unk_rate             share of tokens that are an unknown token
  renyi_eff            Renyi efficiency (alpha=2.5) of the token distribution, Zouhar et al. 2023
  parity_vs_en         Nepali tokens / English tokens on the same sentences (1.0 = equal cost)

Every tokenizer is reduced to a list of UTF-8 byte spans per sentence, so byte-level,
SentencePiece and WordPiece tokenizers are compared on exactly the same footing.

Usage:
    python scripts/audit.py --ne data/raw/.../wikipedia.*.ne-en.ne --en data/raw/.../wikipedia.*.ne-en.en
Extra Hugging Face tokenizers (when you have internet, e.g. on Kaggle):
    python scripts/audit.py ... --hf google/muril-base-cased ai4bharat/IndicBERTv2-MLM-only
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aksharapiece.akshara import akshara_boundaries, aksharas, normalize  # noqa: E402

BYTE_PIECE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def char_to_byte_offsets(text: str):
    offs = [0]
    for ch in text:
        offs.append(offs[-1] + len(ch.encode("utf-8")))
    return offs


# --------------------------------------------------------------------------- adapters
class Adapter:
    """name, vocab_size, and spans(text) -> (list of (byte_start, byte_end), n_unk, token_ids)."""
    name = "?"
    vocab_size = 0

    def spans(self, text):
        raise NotImplementedError


class TiktokenAdapter(Adapter):
    def __init__(self, name, enc):
        self.name, self.enc, self.vocab_size = name, enc, enc.n_vocab

    def spans(self, text):
        ids = self.enc.encode_ordinary(text)
        out, pos = [], 0
        for i in ids:
            n = len(self.enc.decode_single_token_bytes(i))
            out.append((pos, pos + n))
            pos += n
        return out, 0, ids


class SentencePieceAdapter(Adapter):
    def __init__(self, name, path):
        import sentencepiece as spm
        self.name, self.sp = name, spm.SentencePieceProcessor(model_file=str(path))
        self.vocab_size = self.sp.get_piece_size()

    def spans(self, text):
        ids = self.sp.encode(text)
        pieces = [self.sp.id_to_piece(i) for i in ids]
        bs = []
        for p in pieces:
            m = BYTE_PIECE.match(p)
            bs.append(bytes([int(m.group(1), 16)]) if m else p.replace("▁", " ").encode("utf-8"))
        joined = b"".join(bs)
        target = text.encode("utf-8")
        shift = len(joined) - len(target)  # dummy prefix space(s) added by SentencePiece
        if joined[shift:] != target:
            raise ValueError(f"{self.name}: pieces do not reconstruct the input")
        out, pos = [], -shift
        for b in bs:
            s, e = pos, pos + len(b)
            out.append((max(s, 0), max(e, 0)))
            pos = e
        unk = sum(1 for i in ids if i == self.sp.unk_id())
        return [sp for sp in out if sp[1] > sp[0]], unk, ids


class HFAdapter(Adapter):
    def __init__(self, name, tok):
        self.name, self.tok = name, tok
        self.vocab_size = tok.get_vocab_size()
        self.unk_ids = {tok.token_to_id(t) for t in ("[UNK]", "<unk>", "<UNK>") if tok.token_to_id(t) is not None}

    def spans(self, text):
        enc = self.tok.encode(text, add_special_tokens=False)
        b = char_to_byte_offsets(text)
        out = [(b[s], b[e]) for s, e in enc.offsets if e > s]
        unk = sum(1 for i in enc.ids if i in self.unk_ids)
        return out, unk, enc.ids


class AksharaPieceAdapter(Adapter):
    def __init__(self, name, ap):
        self.name, self.ap, self.vocab_size = name, ap, ap.vocab_size
        self.fallback = 0

    def spans(self, text):
        _, spans, ids, fb = self.ap.encode(text)  # text is already normalized by load_pairs
        self.fallback += fb
        b = char_to_byte_offsets(text)
        return [(b[s], b[e]) for s, e in spans], 0, ids


class BytesAdapter(Adapter):
    name, vocab_size = "UTF-8 bytes (no tokenizer)", 256

    def spans(self, text):
        n = len(text.encode("utf-8"))
        return [(i, i + 1) for i in range(n)], 0, list(text.encode("utf-8"))


class CharAdapter(Adapter):
    name, vocab_size = "Unicode characters (reference)", 0

    def spans(self, text):
        b = char_to_byte_offsets(text)
        return [(b[i], b[i + 1]) for i in range(len(text))], 0, [ord(c) for c in text]


class AksharaAdapter(Adapter):
    name, vocab_size = "Aksharas, no merges (reference)", 0

    def spans(self, text):
        b, pos, out = char_to_byte_offsets(text), 0, []
        for a in aksharas(text):
            out.append((b[pos], b[pos + len(a)]))
            pos += len(a)
        return out, 0, [hash(a) for a in aksharas(text)]


# --------------------------------------------------------------------------- loading
def load_default_adapters(hf_names=()):
    adapters = [BytesAdapter(), CharAdapter(), AksharaAdapter()]
    cache = ROOT / "data" / "tokenizers" / "tiktoken_cache"
    if cache.exists():
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
    try:
        import tiktoken
        for enc_name, label in [("p50k_base", "GPT-3 (p50k_base)"), ("cl100k_base", "GPT-4 (cl100k_base)"),
                                ("o200k_base", "GPT-4o (o200k_base)")]:
            try:
                adapters.append(TiktokenAdapter(label, tiktoken.get_encoding(enc_name)))
            except Exception as e:  # file not cached and no internet
                print(f"skip {label}: {e}", file=sys.stderr)
    except ImportError:
        print("skip tiktoken tokenizers (pip install tiktoken)", file=sys.stderr)
    llama = ROOT / "data" / "tokenizers" / "llama2_tokenizer.model"
    if llama.exists():
        adapters.append(SentencePieceAdapter("Llama 2 (SentencePiece BPE)", llama))
    try:
        import mistral_common
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
        p = Path(mistral_common.__file__).parent / "data" / "tekken_240911.json"
        tk = Tekkenizer.from_file(str(p))
        a = TiktokenAdapter("Mistral Tekken (2024)", tk._model)
        a.vocab_size = tk.n_words
        adapters.append(a)
    except Exception as e:
        print(f"skip Mistral Tekken: {e}", file=sys.stderr)
    try:
        import nepalitokenizers
        from tokenizers import Tokenizer
        d = Path(nepalitokenizers.__file__).parent / "models"
        adapters.append(HFAdapter("nepalitokenizers WordPiece", Tokenizer.from_file(str(d / "WordPiece.json"))))
        adapters.append(HFAdapter("nepalitokenizers SentencePiece (Unigram)", Tokenizer.from_file(str(d / "SentencePiece.json"))))
    except Exception as e:
        print(f"skip nepalitokenizers: {e}", file=sys.stderr)
    adapters += load_dir_adapters(ROOT / "data" / "tokenizers")
    for name in hf_names:  # needs internet + transformers; gated models need a Hugging Face token
        try:
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(name)
            adapters.append(HFAdapter(name, t.backend_tokenizer))
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
    return adapters


def load_dir_adapters(folder):
    """Every *.json tokenizer in a folder: AksharaPiece files (*.aksharapiece.json) or plain HF tokenizers."""
    from tokenizers import Tokenizer
    from aksharapiece.tokenizer import AksharaPiece
    out = []
    for path in sorted(glob.glob(str(Path(folder) / "*.json"))):
        name = Path(path).name.replace(".aksharapiece.json", "").replace(".json", "")
        if path.endswith(".aksharapiece.json"):
            out.append(AksharaPieceAdapter(name, AksharaPiece.load(path)))
        else:
            out.append(HFAdapter(name, Tokenizer.from_file(path)))
    return out


def load_mono(files, max_sentences=None, min_chars=10):
    """Nepali-only test text (e.g. a held-out split): pairs with en=None, so parity is not computed."""
    pairs, seen = [], set()
    for f in files:
        for line in open(f, encoding="utf-8"):
            a = normalize(line)
            if len(a) >= min_chars and a not in seen:
                seen.add(a)
                pairs.append((a, None))
                if max_sentences and len(pairs) >= max_sentences:
                    return pairs
    return pairs


def load_pairs(ne_files, en_files):
    pairs, seen = [], set()
    for nf, ef in zip(sorted(ne_files), sorted(en_files)):
        ne = open(nf, encoding="utf-8").read().splitlines()
        en = open(ef, encoding="utf-8").read().splitlines()
        assert len(ne) == len(en), f"{nf} and {ef} are not aligned"
        for a, b in zip(ne, en):
            a, b = normalize(a), b.strip()
            # FLORES v1 repeats some Nepali sentences with a second English translation:
            # keep each Nepali sentence once so no Nepali text is counted twice.
            if a and b and a not in seen:
                seen.add(a)
                pairs.append((a, b))
    return pairs


# --------------------------------------------------------------------------- metrics
def renyi_efficiency(counts: Counter, vocab_size: int, alpha: float = 2.5):
    total = sum(counts.values())
    if total == 0 or vocab_size <= 1:
        return float("nan")
    s = sum((c / total) ** alpha for c in counts.values())
    h = math.log(s) / (1 - alpha)
    return h / math.log(vocab_size)


def audit(adapter, pairs):
    ne_tokens = en_tokens = words = chars = nbytes = 0
    boundaries = bad = unk = 0
    deva_words = broken_words = 0
    counts = Counter()
    for ne, en in pairs:
        spans, n_unk, ids = adapter.spans(ne)
        ne_tokens += len(spans)
        unk += n_unk
        counts.update(ids)
        words += len(ne.split())
        chars += len(ne)
        nbytes += len(ne.encode("utf-8"))
        good = akshara_boundaries(ne, unit="byte")
        cuts = set()
        for s, e in spans:
            cuts.add(s)
            cuts.add(e)
        total_bytes = len(ne.encode("utf-8"))
        cuts.discard(0)
        cuts.discard(total_bytes)
        bad_cuts = {c for c in cuts if c not in good}
        boundaries += len(cuts)
        bad += len(bad_cuts)
        # words with a broken akshara
        b = char_to_byte_offsets(ne)
        for m in re.finditer(r"\S+", ne):
            if re.search(r"[ऀ-ॿ]", m.group()):
                deva_words += 1
                ws, we = b[m.start()], b[m.end()]
                if any(ws < c < we for c in bad_cuts):
                    broken_words += 1
        if en is not None:
            en_tokens += len(adapter.spans(en)[0])
    return {
        "tokenizer": adapter.name,
        "vocab": adapter.vocab_size or "",
        "fertility": ne_tokens / words,
        "chars_per_token": chars / ne_tokens,
        "bytes_per_token": nbytes / ne_tokens,
        "akshara_break_rate": bad / boundaries if boundaries else 0.0,
        "words_broken": broken_words / deva_words if deva_words else 0.0,
        "unk_rate": unk / ne_tokens,
        "renyi_eff": renyi_efficiency(counts, adapter.vocab_size) if adapter.vocab_size else float("nan"),
        "parity_vs_en": ne_tokens / en_tokens if en_tokens else float("nan"),
        "ne_tokens": ne_tokens,
    }


def fmt(r):
    pct = lambda x: f"{100 * x:.1f}%"
    num = lambda x, d=2: "–" if isinstance(x, float) and math.isnan(x) else f"{x:.{d}f}"
    return [r["tokenizer"], str(r["vocab"]), num(r["fertility"]), num(r["chars_per_token"]),
            pct(r["akshara_break_rate"]), pct(r["words_broken"]), pct(r["unk_rate"]),
            num(r["renyi_eff"], 3), num(r["parity_vs_en"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", nargs="+", required=True, help="Nepali files (aligned with --en when given)")
    ap.add_argument("--en", nargs="*", default=[], help="English translations; omit for Nepali-only text")
    ap.add_argument("--hf", nargs="*", default=[], help="Hugging Face tokenizer names to add")
    ap.add_argument("--tokenizer-dir", nargs="*", default=[], help="extra folders of *.json / *.aksharapiece.json")
    ap.add_argument("--max-sentences", type=int, default=None, help="cap for large Nepali-only files")
    ap.add_argument("--out", default=str(ROOT / "results" / "audit"))
    args = ap.parse_args()

    if args.en:
        pairs = load_pairs(args.ne, args.en)
        print(f"{len(pairs):,} unique Nepali-English sentence pairs")
    else:
        pairs = load_mono(args.ne, args.max_sentences)
        print(f"{len(pairs):,} unique Nepali sentences (no English: parity not computed)")
    adapters = load_default_adapters(args.hf)
    for d in args.tokenizer_dir:
        adapters += load_dir_adapters(d)
    rows = []
    for a in adapters:
        r = audit(a, pairs)
        rows.append(r)
        print(f"  {a.name:45s} fertility {r['fertility']:.2f}  break {100 * r['akshara_break_rate']:.1f}%")
    rows.sort(key=lambda r: r["fertility"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    head = ["Tokenizer", "Vocab", "Tokens/word", "Chars/token", "Akshara breaks", "Words broken",
            "UNK", "Rényi eff.", "Parity vs EN"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt(r)) + " |" for r in rows]
    Path(f"{out}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
