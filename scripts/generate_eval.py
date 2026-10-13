"""Generate Nepali text from trained models and measure how well-formed it is.

For every model: sample a continuation for each prompt (same prompts, same sampling seed for all models),
then measure
  malformed_per_1000   malformed aksharas per 1,000 aksharas (orphan vowel signs/halants, stacked vowel signs)
  malformed_words_%    share of generated Devanagari words containing a malformed akshara
  nonword_%            share of generated words never seen (>= --min-count times) in a reference corpus
Human-written reference text is scored the same way for comparison.

    python scripts/generate_eval.py --models "models/*.pt" --tokenizers tokenizers \
        --prompts flores_test.ne --wordlist-text ref.txt --out results/generation
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aksharapiece.akshara import malformed_aksharas, normalize, script_validity  # noqa: E402
from aksharapiece.codec import Codec  # noqa: E402
from aksharapiece.models import LM, ModelConfig  # noqa: E402

WORD_RE = re.compile(r"[ऀ-ॣ०-ॿ‌‍]+")
DEVA_RE = re.compile(r"[ऀ-ॿ]")


def load_model(path, device):
    ck = torch.load(path, map_location="cpu")
    cfg = ModelConfig(**ck["config"])
    m = LM(cfg)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    return m.to(device).eval(), ck


# ----------------------------------------------------------------------------- script-aware decoding
# Character classes: C consonant, K nukta, M vowel sign, N nasal/visarga, V virama, I independent vowel, O other
def char_class(ch):
    o = ord(ch)
    if 0x0915 <= o <= 0x0939 or 0x0958 <= o <= 0x095F or 0x0978 <= o <= 0x097F:
        return "C"
    if o == 0x093C:
        return "K"
    if o == 0x094D:
        return "V"
    if 0x093A <= o <= 0x094C or 0x094E <= o <= 0x094F or 0x0955 <= o <= 0x0957 or 0x0962 <= o <= 0x0963:
        return "M"
    if 0x0900 <= o <= 0x0903 or 0x0951 <= o <= 0x0954:
        return "N"
    if 0x0904 <= o <= 0x0914 or o in (0x0960, 0x0961) or 0x0972 <= o <= 0x0977:
        return "I"
    return "O"


CLASSES = "CKMNVIO"
# which characters may legally follow which (only signs are restricted)
NEEDS = {"M": "CK", "V": "CK", "K": "C", "N": "CKMI"}


def script_mask(pieces, n_model_vocab):
    """allowed[c, t] = token t may follow text whose last character has class c.
    Also returns each token's last-character class (or -1 for tokens with no text)."""
    V = len(pieces)
    allowed = torch.ones(len(CLASSES), n_model_vocab, dtype=torch.bool)
    last = torch.full((n_model_vocab,), -1, dtype=torch.long)
    for t, s in enumerate(pieces):
        if not s:
            continue
        last[t] = CLASSES.index(char_class(s[-1]))
        first = char_class(s[0])
        inner = ("क" + s) if first in NEEDS else s
        if malformed_aksharas(inner):                     # token is malformed on its own (e.g. two vowel signs)
            allowed[:, t] = False
        elif first in NEEDS:
            for ci, c in enumerate(CLASSES):
                if c not in NEEDS[first]:
                    allowed[ci, t] = False
    return allowed, last


@torch.no_grad()
def generate(model, prompts_ids, sep_id, max_new, temperature, top_p, seed, device, batch=32, amp=None,
             constraint=None, prompt_last=None):
    """Batched sampling with right padding (safe for causal models). Stops a row at the line separator.
    constraint = (allowed, last) from script_mask: never emit a sign that cannot attach to the previous character."""
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for i in range(0, len(prompts_ids), batch):
        chunk = prompts_ids[i:i + batch]
        lens = torch.tensor([len(p) for p in chunk], device=device)
        L = int(lens.max()) + max_new
        x = torch.full((len(chunk), L), sep_id, dtype=torch.long, device=device)
        for r, p in enumerate(chunk):
            x[r, :len(p)] = torch.tensor(p, device=device)
        done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
        cur = lens.clone()
        if constraint is not None:
            allowed, tok_last = constraint
            lastc = torch.tensor(prompt_last[i:i + batch], device=device)
        for _ in range(max_new):
            upto = int(cur.max())
            with amp:
                h = model.hidden(x[:, :upto])
            last = h[torch.arange(len(chunk), device=device), cur - 1]
            logits = model.logits_from_hidden(last).float() / temperature
            if constraint is not None:
                logits = logits.masked_fill(~allowed[lastc], float("-inf"))
            probs = torch.softmax(logits, -1)
            sp, si = probs.sort(-1, descending=True)
            keep = sp.cumsum(-1) - sp < top_p
            sp = sp * keep
            nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=g)).squeeze(-1)
            nxt = torch.where(done, torch.full_like(nxt, sep_id), nxt)
            x[torch.arange(len(chunk), device=device), cur] = nxt
            done |= nxt == sep_id
            if constraint is not None:
                tl = tok_last[nxt]
                lastc = torch.where(tl >= 0, tl, lastc)
            cur = cur + (~done).long() + done.long() * 0
            cur = torch.clamp(cur, max=L - 1)
            if done.all():
                break
        for r, p in enumerate(chunk):
            gen = x[r, len(p):].tolist()
            if sep_id in gen:
                gen = gen[:gen.index(sep_id)]
            out.append(gen)
    return out


def generated_part(prompt, full):
    """Split prompt+generation into (generated words, akshara/malformed counts of the generated part).
    Scoring the generation together with its prompt matters: a continuation that extends the prompt's
    last word (बसाइ + ँसराइ -> बसाइँसराइ) is valid text, not a broken syllable."""
    vf, vp = script_validity(full), script_validity(prompt)
    pw, fw = prompt.split(), full.split()
    k = len(pw) if (len(fw) >= len(pw) and pw and fw[len(pw) - 1] == pw[-1]) else max(0, len(pw) - 1)
    return fw[k:], vf["aksharas"] - vp["aksharas"], max(0, vf["malformed"] - vp["malformed"])


def text_metrics(texts, vocab, min_count):
    """texts: plain strings (scored whole) or (prompt, prompt+generation) pairs (generated part scored)."""
    aks = mal = words = mal_words = nonword = 0
    for t in texts:
        if isinstance(t, tuple):
            ws, a, m = generated_part(*t)
        else:
            v = script_validity(t)
            ws, a, m = t.split(), v["aksharas"], v["malformed"]
        aks += a
        mal += m
        for w in ws:
            if not DEVA_RE.search(w):
                continue
            words += 1
            mal_words += bool(malformed_aksharas(w))
            for core in WORD_RE.findall(w):
                if vocab is not None and vocab.get(core, 0) < min_count:
                    nonword += 1
                    break
    return {
        "aksharas": aks, "words": words, "malformed": mal, "malformed_words": mal_words,
        "malformed_per_1000": 1000 * mal / max(1, aks),
        "malformed_words_%": 100 * mal_words / max(1, words),
        "nonword_%": 100 * nonword / max(1, words) if vocab is not None else None,
        "avg_words": words / max(1, len(texts)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="glob of .pt files saved by train_lm.py --save-model")
    ap.add_argument("--tokenizers", required=True, help="folder with the tokenizer files used in training")
    ap.add_argument("--prompts", required=True, help="Nepali text file, one sentence per line")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--prompt-words", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--constrain", choices=["off", "on", "both"], default="off",
                    help="script-aware decoding: block vowel signs/halants that cannot attach to the previous character")
    ap.add_argument("--wordlist-text", default=None, help="large reference text for the non-word rate")
    ap.add_argument("--wordlist-mb", type=float, default=200)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--reference", default=None, help="human text to score the same way (e.g. held-out lines)")
    ap.add_argument("--out", default=str(ROOT / "results" / "generation"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    amp = torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp)

    lines = [normalize(l) for l in open(args.prompts, encoding="utf-8") if len(l.split()) > args.prompt_words + 3]
    prompts = list(dict.fromkeys(" ".join(l.split()[:args.prompt_words]) for l in lines))[:args.n]
    print(f"{len(prompts)} prompts, e.g. {prompts[0]!r}")

    vocab = None
    if args.wordlist_text:
        vocab, size = Counter(), 0
        for line in open(args.wordlist_text, encoding="utf-8"):
            vocab.update(WORD_RE.findall(line))
            size += len(line.encode("utf-8"))
            if size > args.wordlist_mb * 1e6:
                break
        print(f"reference word list: {sum(1 for c in vocab.values() if c >= args.min_count):,} words seen >= {args.min_count} times")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.reference:
        ref = [normalize(l) for l in open(args.reference, encoding="utf-8")][:5000]
        r = {"model": "human-written reference", "tokenizer": "-", "arch": "-", "seed": "-", "decoding": "-", **text_metrics(ref, vocab, args.min_count)}
        rows.append(r)
        print(f"  reference: malformed {r['malformed_per_1000']:.2f}/1000, nonword {r['nonword_%']}")

    codecs = {}
    for path in sorted(glob.glob(args.models)):
        t0 = time.time()
        model, ck = load_model(path, device)
        tname = ck["tokenizer"]
        if tname not in codecs:
            cands = glob.glob(str(Path(args.tokenizers) / f"{tname}.aksharapiece.json")) + \
                glob.glob(str(Path(args.tokenizers) / f"{tname}.json"))
            if not cands:
                print(f"  skip {path}: tokenizer {tname} not found")
                continue
            codecs[tname] = Codec(cands[0])
        codec = codecs[tname]
        ids = codec.encode_batch(prompts)
        modes = {"off": [False], "on": [True], "both": [False, True]}[args.constrain]
        for constrained in modes:
            t1 = time.time()
            cons, plast = None, None
            if constrained:
                allowed, tok_last = script_mask(codec.pieces(), model.c.vocab_size)
                cons = (allowed.to(device), tok_last.to(device))
                plast = [CLASSES.index(char_class(normalize(p)[-1])) for p in prompts]
            gens = generate(model, ids, codec.sep_id, args.max_new, args.temperature, args.top_p, args.seed, device,
                            batch=args.batch, amp=amp, constraint=cons, prompt_last=plast)
            pairs = []
            for pid, g in zip(ids, gens):
                ptxt = codec.decode(pid)
                pairs.append((ptxt, codec.decode(list(pid) + list(g))))
            texts = pairs
            name = Path(path).stem + ("+constrained" if constrained else "")
            parts = Path(path).stem.split("__")
            r = {"model": name, "tokenizer": tname, "arch": parts[1] if len(parts) > 1 else "",
                 "seed": parts[2] if len(parts) > 2 else "", "decoding": "constrained" if constrained else "free",
                 **text_metrics(texts, vocab, args.min_count), "minutes": (time.time() - t1) / 60}
            rows.append(r)
            with open(out / f"{name}.samples.txt", "w", encoding="utf-8") as f:
                for ptxt, full in texts:          # continuation may join the prompt's last word
                    f.write(f"{ptxt} ||| {full[len(ptxt):] if full.startswith(ptxt) else full}\n")
            print(f"  {name}: malformed {r['malformed']} in {r['aksharas']:,} aksharas ({r['malformed_per_1000']:.2f}/1000), "
                  f"nonword {r['nonword_%'] if r['nonword_%'] is None else round(r['nonword_%'], 2)}%  ({r['minutes']:.1f} min)",
                  flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (out / "generation_metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    from generation_stats import summarize
    head = "| Model | Malformed | Aksharas | per 1000 | Words with malformed syllable | Non-words | Words per sample |\n|---|---|---|---|---|---|---|"
    body = [f"| {r['model']} | {r['malformed']} | {r['aksharas']:,} | {r['malformed_per_1000']:.2f} | {r['malformed_words_%']:.2f}% | "
            f"{'' if r['nonword_%'] is None else format(r['nonword_%'], '.2f') + '%'} | {r['avg_words']:.1f} |" for r in rows]
    md = "\n".join([head] + body) + "\n\n**Pooled over seeds**\n\n" + summarize(rows)[0]
    (out / "generation_metrics.md").write_text(md, encoding="utf-8")
    print(md)

if __name__ == "__main__":
    main()
