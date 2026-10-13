"""Paper figures and tables from the saved results.

    python scripts/make_figures.py --out paper/figures
Writes fig*.pdf / fig*.png and tables.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import chi2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# validated categorical pair (dataviz reference palette, slots 1-2) + neutrals
AKS, PLAIN = "#2a78d6", "#eb6834"
OTHER, INK, INK2, GRID = "#b4b2a9", "#0b0b0b", "#52514e", "#e6e4dd"
NAME = {"akshara_bpe_16k": "AksharaPiece BPE", "plain_bpe_16k": "Plain BPE"}
COL = {"akshara_bpe_16k": AKS, "plain_bpe_16k": PLAIN}
ARCH = {"transformer": "Transformer", "mamba": "Mamba-2"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------- 1. tokenizer audit
AUDIT_SHOW = [
    ("GPT-4 (cl100k_base)", "GPT-4 (cl100k)"), ("Qwen/Qwen2.5-0.5B", "Qwen 2.5"),
    ("meta-llama/Llama-3.2-1B", "Llama 3.2"), ("Mistral Tekken (2024)", "Mistral Tekken"),
    ("sarvamai/sarvam-1", "Sarvam-1"), ("GPT-4o (o200k_base)", "GPT-4o (o200k)"),
    ("google/gemma-3-1b-it", "Gemma 3"), ("ai4bharat/IndicBERTv2-MLM-only", "IndicBERT v2"),
    ("nepalitokenizers SentencePiece (Unigram)", "nepalitokenizers SP"), ("google/muril-base-cased", "MuRIL"),
    ("A_bpe_32000", "Plain BPE 32k (ours)"), ("B_akshara_bpe_32000", "AksharaPiece BPE 32k (ours)"),
]


def fig_audit(out):
    rows = {r["tokenizer"]: r for r in csv.DictReader(open(R / "audit_flores_v1_with_pilot.csv"))}
    sel = [(lab, rows[k]) for k, lab in AUDIT_SHOW if k in rows]
    sel.sort(key=lambda x: -float(x[1]["fertility"]))
    labels = [s[0] for s in sel]
    colors = [AKS if "AksharaPiece" in l else (PLAIN if "Plain" in l else OTHER) for l in labels]
    y = np.arange(len(sel))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for ax, key, title, fmt in [(axes[0], "fertility", "Tokens per word (lower is better)", "{:.2f}"),
                                (axes[1], "words_broken", "Words with a syllable cut (%)", "{:.0f}%")]:
        v = np.array([float(s[1][key]) for s in sel]) * (100 if key == "words_broken" else 1)
        ax.barh(y, v, color=colors, height=0.62)
        for yi, vi in zip(y, v):
            ax.text(vi + v.max() * 0.015, yi, fmt.format(vi) if vi >= 0.5 or key == "fertility" else f"{vi:.1f}%",
                    va="center", fontsize=7.5, color=INK2)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, v.max() * 1.18)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle("Nepali tokenization cost on FLORES (6,511 sentences)", x=0.01, ha="left", fontsize=10)
    save(fig, out, "fig1_tokenizer_audit")
    return sel


# ----------------------------------------------------------------------------- 2. bits per byte
def load_bpb():
    d = {}
    for f in glob.glob(str(R / "kaggle_lm_run1/lm/*.json")):
        j = json.load(open(f))
        d[(j["tokenizer"], j["arch"], 0)] = j["bpb"]
    for f in glob.glob(str(R / "kaggle_lm_seeds/lm/*.json")):
        j = json.load(open(f))
        d[(j["tokenizer"], j["arch"], int(j["seed"]))] = j["bpb"]
    return d


def fig_bpb(out):
    d = load_bpb()
    sig = {(s["arch"], s["seed"]): s for s in json.load(open(R / "kaggle_lm_seeds/significance_bpe_16k.json"))}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, arch in zip(axes, ["transformer", "mamba"]):
        for i, seed in enumerate([0, 1, 2]):
            a, p = d.get(("akshara_bpe_16k", arch, seed)), d.get(("plain_bpe_16k", arch, seed))
            if a is None or p is None:
                continue
            diff = 100 * (a - p) / p
            s = sig.get((arch, seed))
            if s:
                ax.errorbar(diff, i, xerr=[[diff - s["ci_low_%"]], [s["ci_high_%"] - diff]], fmt="o", color=AKS,
                            ms=6, capsize=3, lw=1.5)
            else:
                ax.plot(diff, i, "o", color=AKS, ms=6, mfc="white", mew=1.5)
            ax.text(diff, i - 0.28, f"{diff:+.2f}%", ha="center", fontsize=7.5, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks([0, 1, 2], ["seed 0*", "seed 1", "seed 2"])
        ax.set_ylim(2.6, -0.6)
        ax.set_xlim(-1.1, 1.1)
        ax.set_title(ARCH[arch], fontsize=9, loc="left")
        ax.set_xlabel("Bits per byte, AksharaPiece vs plain BPE (%)")
        ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.16, "Negative = AksharaPiece better. Bars: 95% paired bootstrap interval over 5 MB of held-out text. "
             "*seed 0 trained in bfloat16, no per-line scores.", fontsize=7, color=INK2)
    save(fig, out, "fig2_bpb_difference")
    return d


# ----------------------------------------------------------------------------- 3. classification
def fig_cls(out):
    rows = json.load(open(R / "kaggle_eval/zip/classification/cls_metrics.json"))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = 0
    ticks, labels = [], []
    for arch in ["transformer", "mamba"]:
        for tok in ["plain_bpe_16k", "akshara_bpe_16k"]:
            v = [100 * r["accuracy"] for r in rows if r["tokenizer"] == tok and r["arch"] == arch]
            ax.plot([x] * len(v), v, "o", color=COL[tok], ms=5, alpha=0.9)
            ax.hlines(np.mean(v), x - 0.22, x + 0.22, color=COL[tok], lw=2)
            ax.text(x + 0.27, np.mean(v), f"{np.mean(v):.1f}", va="center", fontsize=7.5, color=INK2)
            ticks.append(x)
            labels.append(NAME[tok].replace(" BPE", ""))
            x += 1
        x += 0.6
    ax.set_xticks(ticks, labels, fontsize=8)
    tr = ax.get_xaxis_transform()
    ax.text(0.5, -0.17, "Transformer", ha="center", fontsize=8.5, transform=tr)
    ax.text(2.1, -0.17, "Mamba-2", ha="center", fontsize=8.5, transform=tr)
    ax.set_ylim(82.8, 85)
    ax.set_ylabel("Test accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("News topic classification (np20ng, 20 classes)", fontsize=9, loc="left")
    save(fig, out, "fig3_classification")
    return rows


# ----------------------------------------------------------------------------- 4. generation validity
def load_generation():
    """Pooled counts from the generation run's summary (zip output; falls back to the printed log)."""
    js = R / "kaggle_generation/zip/generation_summary.json"
    if js.exists():
        d = json.load(open(js))
        pooled = {}
        for k, v in d["pooled"].items():
            tok, arch, dec = k.split("|")
            pooled[(tok, arch, dec)] = {"malformed": v["malformed"], "aksharas": v["aksharas"],
                                        "malformed_words_%": 100 * v["malformed_words"] / max(1, v["words"]),
                                        "nonword_%": 100 * v["nonword_w"] / max(1, v["words"])}
        r = [x for x in d["rows"] if x["arch"] == "-"][0]
        return pooled, {"malformed": r["malformed"], "aksharas": r["aksharas"], "nonword_%": r["nonword_%"]}
    log = (R / "kaggle_generation/generation_log.txt").read_text(encoding="utf-8")
    part = log.split("generation_stats.py", 1)[1]
    pooled = {}
    for m in re.finditer(r"\| (\w+) \| (\w+) \| (free|constrained) \| (\d) \| (\d+) / ([\d,]+) \| [\d.]+ \| ([\d.]+)% \| ([\d.]+)% \|", part):
        pooled[(m[1], m[2], m[3])] = {"malformed": int(m[5]), "aksharas": int(m[6].replace(",", "")),
                                      "malformed_words_%": float(m[7]), "nonword_%": float(m[8])}
    ref = re.search(r"Human-written reference: (\d+) malformed in ([\d,]+) aksharas.*non-words ([\d.]+)%", part)
    human = {"malformed": int(ref[1]), "aksharas": int(ref[2].replace(",", "")), "nonword_%": float(ref[3])}
    return pooled, human


def poisson_ci(k, alpha=0.05):
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return lo, hi


def fig_generation(out):
    pooled, human = load_generation()
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    xs, xt = [], []
    x = 0
    for arch in ["transformer", "mamba"]:
        for dec in ["free", "constrained"]:
            for k, tok in enumerate(["plain_bpe_16k", "akshara_bpe_16k"]):
                p = pooled[(tok, arch, dec)]
                r = 1000 * p["malformed"] / p["aksharas"]
                lo, hi = (1000 * v / p["aksharas"] for v in poisson_ci(p["malformed"]))
                ax.bar(x + k * 0.36, r, width=0.34, color=COL[tok], label=NAME[tok] if (arch, dec) == ("transformer", "free") else None)
                ax.errorbar(x + k * 0.36, r, yerr=[[r - lo], [hi - r]], color=INK, lw=1, capsize=2.5)
                ax.text(x + k * 0.36, hi + 0.012, str(p["malformed"]), ha="center", fontsize=7, color=INK2)
            xt.append(x + 0.18)
            xs.append(f"{ARCH[arch]}\n{'free sampling' if dec == 'free' else 'script-aware'}")
            x += 1.05
        x += 0.25
    hr = 1000 * human["malformed"] / human["aksharas"]
    ax.axhline(hr, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(x - 0.4, hr + 0.01, f"human-written text {hr:.2f}", ha="right", fontsize=7.5, color=INK2)
    ax.set_xticks(xt, xs, fontsize=7.5)
    ax.set_ylabel("Malformed syllables per 1,000")
    ax.set_ylim(0, 0.42)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Broken syllables in generated text (≈210k syllables per bar)", fontsize=9, loc="left", pad=14)
    fig.text(0.01, -0.04, "Numbers above bars: malformed count (2 models × 3,000 continuations). Bars: exact 95% Poisson interval.",
             fontsize=7, color=INK2)
    save(fig, out, "fig4_generation_validity")
    return pooled, human


# ----------------------------------------------------------------------------- 5. typos in training text
CAT_NAMES = {"doubled_sign": "Same vowel sign twice", "two_vowel_signs": "Two different vowel signs",
             "nasal_then_vowel": "Nasal mark before vowel sign", "sign_after_halant": "Sign after halant",
             "word_start": "Sign at start of word", "other": "Other"}


def fig_typos(out):
    t = json.load(open(R / "typo_audit.json"))["pretrain.txt"]
    cats = sorted(CAT_NAMES, key=lambda c: -t["categories"][c] if c != "other" else 1)
    v = [100 * t["categories"][c] / t["malformed"] for c in cats]
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    y = np.arange(len(cats))
    ax.barh(y, v, color=[INK2 if c != "other" else OTHER for c in cats], height=0.6)
    for yi, vi, c in zip(y, v, cats):
        ax.text(vi + 0.6, yi, f"{vi:.0f}%  ({t['categories'][c]:,})", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [CAT_NAMES[c] for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, max(v) * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of malformed syllables")
    ax.set_title(f"Malformed syllables in IndicCorp Nepali: {t['per_1000']:.2f} per 1,000 "
                 f"({t['mb_read']:.0f} MB)", fontsize=9, loc="left")
    save(fig, out, "fig5_corpus_typos")
    return t


# ----------------------------------------------------------------------------- 6. error types: models vs corpus
def fig_error_types(out):
    f = R / "error_types.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    cats = ["doubled_sign", "two_vowel_signs", "nasal_then_vowel", "sign_after_halant", "word_start", "other"]
    groups = [("Training corpus", OTHER), ("Plain BPE", PLAIN), ("AksharaPiece", AKS)]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    y = np.arange(len(cats))
    h = 0.26
    for gi, (g, col) in enumerate(groups):
        cnt = d["counts"][g]
        tot = sum(cnt.values())
        v = [100 * cnt[c] / tot for c in cats]
        lab = f"{g} (n={tot:,})"
        ax.barh(y + (gi - 1) * (h + 0.02), v, height=h, color=col, label=lab)
    ax.set_yticks(y, [CAT_NAMES[c] for c in cats])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of malformed syllables (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, fontsize=7.5)
    t = d["tests"]
    ax.set_title("Kinds of broken syllables: generated text vs training corpus", fontsize=9, loc="left", pad=22)
    fig.text(0.01, -0.06, f"Same mix as corpus? Plain BPE: chi2 p = {t['Plain BPE']['p']:.2f}; "
             f"AksharaPiece: p = {t['AksharaPiece']['p']:.1g}. Free sampling, 4 models per tokenizer.", fontsize=7, color=INK2)
    save(fig, out, "fig6_error_types")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = fig_audit(out)
    bpb = fig_bpb(out)
    cls = fig_cls(out)
    pooled, human = fig_generation(out)
    typos = fig_typos(out)
    errs = fig_error_types(out)

    # ---------------- tables
    T = ["## Table 1. Tokenizer audit on FLORES Nepali", "",
         "| Tokenizer | Vocab | Tokens/word | Words with syllable cut | Tokens vs English |", "|---|---|---|---|---|"]
    for lab, r in sel:
        nepali_only = "ours" in lab or "nepalitokenizers" in lab      # English parity is meaningless for these
        T.append(f"| {lab} | {int(float(r['vocab'])):,} | {float(r['fertility']):.2f} | {100 * float(r['words_broken']):.1f}% | "
                 + ("n/a |" if nepali_only else f"{float(r['parity_vs_en']):.2f}× |"))
    T += ["", "## Table 2. Bits per byte (5 MB held-out IndicCorp), mean ± sd over 3 seeds", "",
          "| Architecture | Plain BPE 16k | AksharaPiece BPE 16k | Difference |", "|---|---|---|---|"]
    for arch in ["transformer", "mamba"]:
        p = [bpb[("plain_bpe_16k", arch, s)] for s in (0, 1, 2)]
        a = [bpb[("akshara_bpe_16k", arch, s)] for s in (0, 1, 2)]
        diffs = [100 * (x - y) / y for x, y in zip(a, p)]
        T.append(f"| {ARCH[arch]} | {np.mean(p):.4f} ± {np.std(p, ddof=1):.4f} | {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f} | "
                 f"{np.mean(diffs):+.2f}% ± {np.std(diffs, ddof=1):.2f} |")
    T += ["", "## Table 3. News classification (np20ng), mean ± sd over 2 seeds", "",
          "| Architecture | Plain BPE | AksharaPiece BPE |", "|---|---|---|"]
    for arch in ["transformer", "mamba"]:
        cell = []
        for tok in ["plain_bpe_16k", "akshara_bpe_16k"]:
            acc = [100 * r["accuracy"] for r in cls if r["tokenizer"] == tok and r["arch"] == arch]
            f1 = [100 * r["macro_f1"] for r in cls if r["tokenizer"] == tok and r["arch"] == arch]
            cell.append(f"{np.mean(acc):.2f} ± {np.std(acc, ddof=1):.2f} acc / {np.mean(f1):.2f} F1")
        T.append(f"| {ARCH[arch]} | {cell[0]} | {cell[1]} |")
    T += ["", "## Table 4. Generated-text validity (2 seeds × 3,000 continuations per cell)", "",
          f"Human-written reference: {human['malformed']} malformed in {human['aksharas']:,} syllables "
          f"({1000 * human['malformed'] / human['aksharas']:.2f}/1000), non-words {human['nonword_%']:.2f}%.", "",
          "| Architecture | Decoding | Plain BPE malformed | AksharaPiece malformed | Plain non-words | AksharaPiece non-words |",
          "|---|---|---|---|---|---|"]
    for arch in ["transformer", "mamba", "all"]:
        for dec in ["free", "constrained"]:
            p, a = pooled[("plain_bpe_16k", arch, dec)], pooled[("akshara_bpe_16k", arch, dec)]
            T.append(f"| {ARCH.get(arch, 'Both')} | {dec} | {p['malformed']} / {p['aksharas']:,} | {a['malformed']} / {a['aksharas']:,} | "
                     f"{p['nonword_%']:.2f}% | {a['nonword_%']:.2f}% |")
    T += ["", "## Table 5. Malformed syllables in IndicCorp Nepali", "",
          f"{typos['malformed']:,} malformed in {typos['aksharas']:,} syllables ({typos['per_1000']:.3f}/1000; "
          f"{typos['lines_with_malformed_%']:.2f}% of lines). Two repair rules fix {typos['repaired_%']:.0f}%.", ""]
    if errs:
        T += ["## Table 6. Kinds of malformed syllables (share; count)", "",
              "| Type | Training corpus | Plain BPE models | AksharaPiece models |", "|---|---|---|---|"]
        for c in ["doubled_sign", "two_vowel_signs", "nasal_then_vowel", "sign_after_halant", "word_start", "other"]:
            T.append(f"| {CAT_NAMES[c]} | " + " | ".join(
                f"{100 * errs['counts'][g][c] / sum(errs['counts'][g].values()):.0f}% ({errs['counts'][g][c]})"
                for g in ("Training corpus", "Plain BPE", "AksharaPiece")) + " |")
        T += ["", "Chi-square vs corpus: " + "; ".join(f"{g}: p = {v['p']:.3g}" for g, v in errs["tests"].items()), ""]
    (out / "tables.md").write_text("\n".join(T) + "\n", encoding="utf-8")
    print("\n".join(T))


if __name__ == "__main__":
    main()
