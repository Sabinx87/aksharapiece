"""Compare the kinds of malformed syllables in generated text with those in the training corpus.

    python scripts/error_types.py --samples "results/kaggle_generation/zip/gen_free/*.samples.txt" \
        --corpus results/typo_audit.json --out results/error_types
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from typo_audit import CATS, NAMES, classify  # noqa: E402


def generated_errors(path):
    """(category, word) for malformed syllables in the generated part only (prompt errors subtracted)."""
    out = []
    for line in open(path, encoding="utf-8"):
        p, t = line.rstrip("\n").split(" ||| ", 1)
        full = Counter((c, a) for c, a, _ in classify(p + t))
        full.subtract(Counter((c, a) for c, a, _ in classify(p)))
        words = {a: w for c, a, w in classify(p + t)}
        for (c, a), k in full.items():
            out += [(c, words.get(a, a))] * max(0, k)
    return out


def chi2_test(a, b):
    """Chi-square test that two category count vectors have the same distribution."""
    from scipy.stats import chi2_contingency
    rows = [[a[c] for c in CATS], [b[c] for c in CATS]]
    keep = [i for i in range(len(CATS)) if rows[0][i] + rows[1][i] > 0]
    stat, p, dof, _ = chi2_contingency([[r[i] for i in keep] for r in rows])
    return stat, p, dof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-key", default="pretrain.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    groups = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(Counter))
    for f in sorted(glob.glob(args.samples)):
        tok = "AksharaPiece" if Path(f).name.startswith("akshara") else "Plain BPE"
        for c, w in generated_errors(f):
            groups[tok][c] += 1
            examples[tok][c][w] += 1
    corpus = Counter(json.load(open(args.corpus))[args.corpus_key]["categories"])
    groups = {"Training corpus": corpus, **groups}
    res = {"categories": CATS, "names": NAMES, "counts": {g: {c: v[c] for c in CATS} for g, v in groups.items()},
           "examples": {g: {c: examples[g][c].most_common(8) for c in CATS} for g in examples}, "tests": {}}
    lines = ["| Error type | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
    for c in CATS:
        lines.append(f"| {NAMES[c]} | " + " | ".join(f"{100 * v[c] / max(1, sum(v.values())):.0f}% ({v[c]})" for v in groups.values()) + " |")
    lines.append("| **Total** | " + " | ".join(f"{sum(v.values()):,}" for v in groups.values()) + " |")
    lines += ["", "Chi-square test, same mix of error types as the training corpus?", ""]
    for g in groups:
        if g == "Training corpus":
            continue
        stat, p, dof = chi2_test(corpus, groups[g])
        res["tests"][g] = {"chi2": stat, "p": p, "dof": dof}
        lines.append(f"- {g} vs corpus: chi2 = {stat:.1f}, dof = {dof}, p = {p:.3g}")
    if {"Plain BPE", "AksharaPiece"} <= set(groups):
        stat, p, dof = chi2_test(groups["Plain BPE"], groups["AksharaPiece"])
        res["tests"]["Plain vs AksharaPiece"] = {"chi2": stat, "p": p, "dof": dof}
        lines.append(f"- Plain BPE vs AksharaPiece: chi2 = {stat:.1f}, dof = {dof}, p = {p:.3g}")
    lines += [""]
    for g in examples:
        lines.append(f"**{g} examples**")
        for c in CATS:
            if examples[g][c]:
                lines.append(f"- {NAMES[c]}: " + ", ".join(f"{w} ({k})" for w, k in examples[g][c].most_common(6)))
        lines.append("")
    Path(args.out + ".json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    Path(args.out + ".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
