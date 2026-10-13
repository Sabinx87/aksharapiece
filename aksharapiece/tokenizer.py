"""AksharaPiece: BPE / Unigram over aksharas instead of characters or bytes.

How it works
------------
1. Normalize text and split it into aksharas.
2. Give every akshara seen in training its own private-use code point (Unicode plane 15),
   so the subword trainer sees each akshara as ONE symbol and can never split it.
3. Train an ordinary Hugging Face BPE or Unigram model on that mapped text.
4. At encode time, map text the same way; tokens are mapped back for output.

Aksharas never seen in training fall back to their individual characters (rare; reported as
`fallback_aksharas` so it can be measured).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from .akshara import aksharas, normalize

PUA_START = 0xF0000  # Supplementary Private Use Area-A (65,534 code points)
PUA_END = 0xFFFFD


import re

# vowel signs and nasal/visarga marks at the end of an akshara
_VOWEL_ENDING = re.compile(r"[ऺऻा-ौॎॏॕ-ॗॢॣऀ-ः]+$")


class AksharaPiece:
    """rare = how aksharas outside the table are encoded:
        "split" : consonant cluster (one symbol if it is in the table) + its vowel sign(s);
                  never cuts inside a conjunct, so only the cluster|vowel boundary is exposed
        "chars" : individual Unicode characters (original prototype behaviour)
    """

    def __init__(self, tokenizer: Tokenizer, table: List[str], rare: str = "chars"):
        self.tok = tokenizer
        self.table = table                                   # index -> akshara
        self.rare = rare
        self.to_pua = {a: chr(PUA_START + i) for i, a in enumerate(table)}
        self.from_pua = {v: k for k, v in self.to_pua.items()}

    # ------------------------------------------------------------------ mapping
    def _map(self, text: str) -> Tuple[str, List[int], int]:
        """Return mapped text, and for each mapped char the number of original chars it covers."""
        out, widths, fallback = [], [], 0
        for a in aksharas(text):
            if len(a) > 1 and a in self.to_pua:
                out.append(self.to_pua[a])
                widths.append(len(a))
                continue
            if len(a) > 1:
                fallback += 1
                if self.rare == "split":
                    m = _VOWEL_ENDING.search(a)
                    base = a[:m.start()] if m else a
                    if m and len(base) > 1 and base in self.to_pua:
                        out.append(self.to_pua[base])
                        widths.append(len(base))
                        out.extend(m.group())
                        widths.extend([1] * len(m.group()))
                        continue
            out.extend(a)
            widths.extend([1] * len(a))
        return "".join(out), widths, fallback

    def _unmap(self, s: str) -> str:
        return "".join(self.from_pua.get(c, c) for c in s)

    # ------------------------------------------------------------------ public API
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 16000, model: str = "bpe",
              min_akshara_freq: int = 2, max_aksharas: int = 60000, coverage: float | None = None,
              rare: str = "split") -> "AksharaPiece":
        """coverage: keep the most frequent aksharas until they cover this share of all akshara
        occurrences (e.g. 0.99); the rest are encoded as cluster + vowel sign (rare="split").
        Every table entry uses a vocabulary slot, so a smaller table leaves more room for merges."""
        texts = [normalize(t) for t in texts]
        counts = Counter(a for t in texts for a in aksharas(t) if len(a) > 1)
        ranked = [(a, c) for a, c in counts.most_common(max_aksharas) if c >= min_akshara_freq]
        if coverage is not None:
            total, cum, keep = sum(counts.values()), 0, []
            for a, c in ranked:
                keep.append((a, c))
                cum += c
                if cum / total >= coverage:
                    break
            ranked = keep
        table = [a for a, _ in ranked]
        self = cls(Tokenizer(models.BPE()), table, rare=rare)
        mapped = [self._map(t)[0] for t in texts]

        if model == "bpe":
            tok = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=False))
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                          min_frequency=2, limit_alphabet=100000, show_progress=False)
        elif model == "unigram":
            tok = Tokenizer(models.Unigram())
            trainer = trainers.UnigramTrainer(vocab_size=vocab_size, unk_token="<unk>",
                                              special_tokens=["<unk>", "<s>", "</s>", "<pad>"], show_progress=False,
                                              max_piece_length=16)
        else:
            raise ValueError(model)
        tok.normalizer = normalizers.Sequence([])
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        tok.train_from_iterator(mapped, trainer=trainer)
        self.tok = tok
        return self

    def encode(self, text: str):
        """Return (tokens as original text, char spans in the normalized text, ids, n_fallback)."""
        text = normalize(text)
        mapped, widths, fallback = self._map(text)
        enc = self.tok.encode(mapped, add_special_tokens=False)
        # cumulative original-char offset for each mapped char position
        cum = [0]
        for w in widths:
            cum.append(cum[-1] + w)
        spans = [(cum[s], cum[e]) for s, e in enc.offsets if e > s]
        tokens = [self._unmap(t) for t in enc.tokens]
        return tokens, spans, enc.ids, fallback

    def encode_batch_ids(self, texts: List[str], normalized: bool = False) -> List[List[int]]:
        """Fast path for training data: ids only, many lines at once."""
        mapped = [self._map(t if normalized else normalize(t))[0] for t in texts]
        return [e.ids for e in self.tok.encode_batch(mapped, add_special_tokens=False)]

    def decode(self, ids) -> str:
        return self._unmap(self.tok.decode(ids))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def save(self, path: str):
        Path(path).write_text(json.dumps({"format": "aksharapiece-v1", "rare": self.rare, "table": self.table,
                                          "tokenizer": json.loads(self.tok.to_str())}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str) -> "AksharaPiece":
        d = json.loads(Path(path).read_text())
        # v0 files (first Kaggle run) predate the "rare" setting and used character fallback
        return cls(Tokenizer.from_str(json.dumps(d["tokenizer"])), d["table"], rare=d.get("rare", "chars"))


def train_plain(texts: Iterable[str], vocab_size: int = 16000, model: str = "bpe") -> Tokenizer:
    """Same trainer and settings as AksharaPiece but on characters: the fair in-house baseline (variant A)."""
    texts = [normalize(t) for t in texts]
    if model == "bpe":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
                                      min_frequency=2, show_progress=False)
    else:
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, unk_token="<unk>",
                                          special_tokens=["<unk>", "<s>", "</s>", "<pad>"], show_progress=False,
                                          max_piece_length=16)
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    tok.train_from_iterator(texts, trainer=trainer)
    return tok
