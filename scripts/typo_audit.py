"""How often does real Nepali text contain malformed syllables, and what kinds?

Classifies every malformed akshara (see aksharapiece.akshara.malformed_aksharas) by what comes before it:
  doubled_sign       the same vowel sign twice              दुुई, कुुरा
  nasal_then_vowel   nasal mark typed before the vowel sign  आफंैले (should be आफैंले)
  sign_after_halant  vowel sign or mark right after a halant  निर्धा्रित, क्ा
  two_vowel_signs    two different vowel signs on a syllable  काे, िु
  word_start         sign with no letter before it            ेदार, ँ
  other
Also counts how many would be repaired by two simple normalization rules
(drop a repeated vowel sign; put a nasal mark after the vowel sign).

    python scripts/typo_audit.py --files data/clean/heldout.txt data/clean/pretrain.txt --mb 120 --out results/typo_audit
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aksharapiece.akshara import AKSHARA_RE, DEVANAGARI_RE, _VOWEL_SIGN, malformed_aksharas, normalize  # noqa: E402

MATRA = re.compile(r"[ऺ-ौॎॏॕ-ॗॢॣ]")
NASAL = "ँंः"
VIRAMA = "्"
CATS = ["doubled_sign", "nasal_then_vowel", "sign_after_halant", "two_vowel_signs", "word_start", "other"]
NAMES = {"doubled_sign": "Same vowel sign twice", "nasal_then_vowel": "Nasal mark before vowel sign",
         "sign_after_halant": "Sign after halant", "two_vowel_signs": "Two different vowel signs",
         "word_start": "Sign at start of word", "other": "Other"}

_DOUBLE = re.compile(r"([ा-ौ])\1+")
_NASAL_SWAP = re.compile(r"([ँं])([ा-ौ])")


def repair(text):
    """Two conservative repairs for legacy-font conversion errors."""
    text = _DOUBLE.sub(r"\1", text)
    return _NASAL_SWAP.sub(r"\2\1", text)


def classify(text):
    """Yield (category, akshara, word) for every malformed akshara in text."""
    for m in AKSHARA_RE.finditer(text):
        a = m.group()
        starts_with_sign = a[0] in "ऀँंः" or MATRA.match(a) or a[0] in (VIRAMA, "़")
        n_vs = len(_VOWEL_SIGN.findall(a))
        if not starts_with_sign and n_vs <= 1:
            continue
        s, e = m.span()
        ws = text.rfind(" ", 0, s) + 1
        we = text.find(" ", e)
        word = text[ws:we if we >= 0 else len(text)]
        prev = text[s - 1] if s > 0 else " "
        if not starts_with_sign:                         # several vowel signs inside one akshara
            vs = _VOWEL_SIGN.findall(a)
            cat = "doubled_sign" if len(set(vs)) < len(vs) else "two_vowel_signs"
        elif prev in " \n\t" or not DEVANAGARI_RE.match(prev):
            cat = "word_start"
        elif prev in NASAL and MATRA.match(a):
            cat = "nasal_then_vowel"
        elif prev == VIRAMA:
            cat = "sign_after_halant"
        elif MATRA.match(prev) and MATRA.match(a):
            cat = "doubled_sign" if prev == a[0] else "two_vowel_signs"
        else:
            cat = "other"
        yield cat, a, word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--mb", type=float, default=120, help="MB to read from each file")
    ap.add_argument("--normalized", action="store_true", help="text is already normalized (clean.py output)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = {}
    for f in args.files:
        cats, examples, words_ex = Counter(), defaultdict(Counter), defaultdict(Counter)
        aks = lines = bad_lines = mal = mal_after_repair = size = 0
        for line in open(f, encoding="utf-8"):
            size += len(line.encode("utf-8"))
            if size > args.mb * 1e6:
                break
            t = line.strip() if args.normalized else normalize(line)
            lines += 1
            aks += sum(1 for a in AKSHARA_RE.findall(t) if DEVANAGARI_RE.match(a))
            found = list(classify(t))
            if found:
                bad_lines += 1
                mal += len(found)
                for cat, a, w in found:
                    cats[cat] += 1
                    examples[cat][a] += 1
                    words_ex[cat][w] += 1
                mal_after_repair += len(malformed_aksharas(repair(t)))
        name = Path(f).name
        res[name] = {"mb_read": round(min(size, args.mb * 1e6) / 1e6, 1), "lines": lines, "aksharas": aks,
                     "malformed": mal, "per_1000": 1000 * mal / max(1, aks),
                     "lines_with_malformed_%": 100 * bad_lines / max(1, lines),
                     "categories": {c: cats[c] for c in CATS},
                     "malformed_after_repair": mal_after_repair,
                     "repaired_%": 100 * (1 - mal_after_repair / max(1, mal)),
                     "top_words": {c: words_ex[c].most_common(12) for c in CATS}}
        print(f"{name}: {mal:,} malformed in {aks:,} aksharas ({res[name]['per_1000']:.3f}/1000), "
              f"{res[name]['lines_with_malformed_%']:.2f}% of lines; two repair rules fix {res[name]['repaired_%']:.1f}%")
        for c in CATS:
            print(f"   {NAMES[c]:30s} {cats[c]:7,d}  {100 * cats[c] / max(1, mal):5.1f}%   "
                  + " ".join(w for w, _ in words_ex[c].most_common(6)))

    Path(args.out + ".json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    md = ["| Text | Aksharas | Malformed | per 1000 | Lines affected | " + " | ".join(NAMES[c] for c in CATS)
          + " | Fixed by 2 repair rules |", "|---" * (6 + len(CATS)) + "|"]
    for n, r in res.items():
        md.append(f"| {n} | {r['aksharas']:,} | {r['malformed']:,} | {r['per_1000']:.3f} | {r['lines_with_malformed_%']:.2f}% | "
                  + " | ".join(f"{100 * r['categories'][c] / max(1, r['malformed']):.1f}%" for c in CATS)
                  + f" | {r['repaired_%']:.1f}% |")
    md.append("")
    for n, r in res.items():
        md.append(f"**Most frequent malformed words in {n}**\n")
        for c in CATS:
            if r["top_words"][c]:
                md.append(f"- {NAMES[c]}: " + ", ".join(f"{w} ({k})" for w, k in r["top_words"][c][:8]))
        md.append("")
    Path(args.out + ".md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
