"""Devanagari normalization and akshara (orthographic syllable) segmentation.

An akshara is the unit a Nepali reader sees as one written syllable:
    consonant cluster (consonants joined by virama) + optional vowel sign + optional nasal/visarga marks
    or an independent vowel + optional marks.

Examples:
    नमस्ते        -> न | म | स्ते
    क्षेत्र        -> क्षे | त्र
    विद्यालयहरूमा -> वि | द्या | ल | य | ह | रू | मा

Every character outside Devanagari (Latin letters, digits, spaces, punctuation, emoji)
is its own unit, so akshara boundaries are defined for any text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

# --- Character classes (Unicode Devanagari block, U+0900-U+097F) -------------------------
_CONS = "क-हक़-य़ॸ-ॿ"   # consonants (incl. precomposed nukta forms, extended)
_NUKTA = "़"
_VIRAMA = "्"
_JOINERS = "‌‍"                             # ZWNJ, ZWJ (half forms, eyelash ra)
_MATRA = "ऺऻा-ौॎॏॕ-ॗॢॣ"  # dependent vowel signs
_MARKS = "ऀ-ः॑-॔"                 # chandrabindu, anusvara, visarga, vedic accents
_INDEP_VOWEL = "ऄ-औॠॡॲ-ॷ"

AKSHARA_RE = re.compile(
    rf"""
    (?:[{_CONS}]{_NUKTA}?{_VIRAMA}‍?)*            # zero or more half consonants (conjunct; ZWJ keeps it joined)
    [{_CONS}]{_NUKTA}?                                 # base consonant
    (?:{_VIRAMA}(?:‌|‍?(?![{_CONS}]))        # visible halant: word-final (छन्) or forced by ZWNJ ...
     |[{_MATRA}]*)                                     # ... or vowel signs
    [{_MARKS}]*                                        # nasalization / visarga
  | [{_INDEP_VOWEL}][{_MATRA}]*[{_MARKS}]*             # independent vowel (+ rare signs)
  | [{_MATRA}{_MARKS}{_NUKTA}{_VIRAMA}][{_MARKS}]*     # orphan sign (typo / broken text) as its own unit
  | .                                                  # any other single character
    """,
    re.VERBOSE | re.DOTALL,
)

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Characters dropped during normalization (soft hyphen, BOM, directional marks).
_DROP = dict.fromkeys(map(ord, "­﻿‎‏"), None)


# Two-sign sequences that are common typing/font-conversion errors in Nepali text
# (e.g. legacy Preeti font converted to Unicode): ा + े looks like ो.
_VOWEL_FIXES = {
    "ाे": "ो",  # ा + े -> ो
    "ाै": "ौ",  # ा + ै -> ौ
    "ेा": "ो",  # े + ा -> ो
}


def normalize(text: str, *, nfc: bool = True, collapse_spaces: bool = True,
              devanagari_digits: str = "keep", fix_vowel_signs: bool = True) -> str:
    """Normalize Nepali text consistently.

    - NFC composition. Note: NFC *decomposes* precomposed nukta letters (U+0958-U+095F),
      because Unicode lists them as composition exclusions, so क़ is always क + ़ afterwards.
    - Removes invisible junk characters (soft hyphen, BOM, LRM/RLM). ZWJ/ZWNJ are kept:
      they change how conjuncts render (e.g. eyelash ra र्‍).
    - fix_vowel_signs: repairs ा+े -> ो style errors common in converted legacy-font text.
    - devanagari_digits: "keep", "to_ascii" (०-९ -> 0-9) or "to_devanagari".
    """
    if nfc:
        text = unicodedata.normalize("NFC", text)
    text = text.translate(_DROP)
    if fix_vowel_signs:
        for bad, good in _VOWEL_FIXES.items():
            text = text.replace(bad, good)
    if devanagari_digits == "to_ascii":
        text = text.translate({0x0966 + i: ord("0") + i for i in range(10)})
    elif devanagari_digits == "to_devanagari":
        text = text.translate({ord("0") + i: 0x0966 + i for i in range(10)})
    if collapse_spaces:
        text = re.sub(r"[ \t ]+", " ", text).strip()
    return text


def aksharas(text: str) -> List[str]:
    """Split text into aksharas; non-Devanagari characters are single units.

    The pieces always join back to the input exactly.
    """
    return AKSHARA_RE.findall(text)


def akshara_spans(text: str) -> List[Tuple[int, int]]:
    """Character (code point) spans of each akshara."""
    return [m.span() for m in AKSHARA_RE.finditer(text)]


def akshara_boundaries(text: str, unit: str = "char") -> set:
    """Set of offsets where one akshara ends and the next begins (including 0 and len).

    unit="char" gives code point offsets; unit="byte" gives UTF-8 byte offsets.
    A tokenizer boundary that is NOT in this set falls inside an akshara.
    """
    spans = akshara_spans(text)
    if unit == "char":
        cuts = {0, len(text)}
        for s, e in spans:
            cuts.add(s)
            cuts.add(e)
        return cuts
    if unit == "byte":
        # cumulative byte offset of every char position
        offs = [0]
        for ch in text:
            offs.append(offs[-1] + len(ch.encode("utf-8")))
        cuts = {0, offs[-1]}
        for s, e in spans:
            cuts.add(offs[s])
            cuts.add(offs[e])
        return cuts
    raise ValueError("unit must be 'char' or 'byte'")


_MARK_START = re.compile(rf"^[{_MATRA}{_MARKS}{_NUKTA}{_VIRAMA}]")
_VOWEL_SIGN = re.compile(rf"[{_MATRA}]")


def malformed_aksharas(text: str) -> List[str]:
    """Aksharas that cannot be written correctly in Devanagari:
    - an orphan sign: a vowel sign, nasal mark, nukta or halant with no consonant/vowel before it
      (e.g. after a space, or a vowel sign after a halant: "क्ा")
    - two or more vowel signs on one syllable (e.g. "काे", "किु")
    """
    bad = []
    for a in aksharas(text):
        if _MARK_START.match(a) or len(_VOWEL_SIGN.findall(a)) > 1:
            bad.append(a)
    return bad


def script_validity(text: str) -> dict:
    """Counts for measuring how well-formed a (generated) Devanagari text is."""
    deva = [a for a in aksharas(text) if DEVANAGARI_RE.match(a)]
    bad = malformed_aksharas(text)
    words = [w for w in text.split() if DEVANAGARI_RE.search(w)]
    bad_words = sum(1 for w in words if malformed_aksharas(w))
    return {"aksharas": len(deva), "malformed": len(bad), "words": len(words), "malformed_words": bad_words}


def is_devanagari(s: str) -> bool:
    return bool(DEVANAGARI_RE.search(s))


if __name__ == "__main__":
    for w in ["नमस्ते", "क्षेत्र", "विद्यालयहरूमा", "काठमाडौं", "छन्", "र्‍याली", "ॐ", "AI र नेपाली"]:
        print(w, "->", " | ".join(aksharas(normalize(w))))
