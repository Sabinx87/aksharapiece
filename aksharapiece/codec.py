"""One interface for every tokenizer file the project produces: encode text to ids, decode ids to text."""
from __future__ import annotations

from pathlib import Path
from typing import List


class Codec:
    def __init__(self, path: str):
        self.path = str(path)
        if self.path.endswith(".aksharapiece.json"):
            from .tokenizer import AksharaPiece
            self._ap = AksharaPiece.load(self.path)
            self._tok = self._ap.tok
            self.vocab_size = self._ap.vocab_size
        else:
            from tokenizers import Tokenizer
            self._ap = None
            self._tok = Tokenizer.from_file(self.path)
            self.vocab_size = self._tok.get_vocab_size()
        self.sep_id = self.vocab_size            # line separator used in training (model vocab = vocab + 1)
        self.name = Path(self.path).name.replace(".aksharapiece.json", "").replace(".json", "")

    def encode(self, text: str) -> List[int]:
        from .akshara import normalize
        text = normalize(text)
        if self._ap is not None:
            return self._ap.encode_batch_ids([text], normalized=True)[0]
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        from .akshara import normalize
        texts = [normalize(t) for t in texts]
        if self._ap is not None:
            return self._ap.encode_batch_ids(texts, normalized=True)
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def pieces(self) -> List[str]:
        """Surface text of every token id, with the word-start marker shown as a space."""
        out = []
        for i in range(self.vocab_size):
            t = self._tok.id_to_token(i) or ""
            if self._ap is not None:
                t = self._ap._unmap(t)
            out.append(t.replace("▁", " "))
        return out

    def decode(self, ids: List[int]) -> str:
        ids = [i for i in ids if i < self.vocab_size]
        if self._ap is not None:
            return self._ap.decode(ids)
        return self._tok.decode(ids)
