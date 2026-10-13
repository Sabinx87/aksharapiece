"""Package the trained models, tokenizers and evaluation outputs for Hugging Face, and upload them.

Creates two public repositories under your account:
  <user>/<models-repo>   model repo: 2 tokenizers, 8 models (safetensors + config), loading code, model card
  <user>/<data-repo>     dataset repo: generated samples (free and script-aware), metrics, per-line LM scores

    python scripts/hf_release.py --models-dir models --lm-zip aksharapiece_lm_results.zip \
        --gen-zip aksharapiece_gen_results.zip --out release            # build only (dry run)
    HF_TOKEN=<write token> python scripts/hf_release.py ... --upload    # build and upload
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PAPER_TITLE = ("Keeping Syllables Whole Is Not Enough: Akshara-Aware Tokenization, "
               "Corpus Typos, and Script-Aware Decoding for Nepali Language Models")
AUTHORS = "Sabin Bhattarai and Nabin Aadhikari"

EXAMPLE_PY = '''"""Generate Nepali text with a released AksharaPiece model.

    pip install torch safetensors tokenizers huggingface_hub
    python example.py --model akshara_bpe_16k__transformer__s1 --prompt "नेपालको राजधानी" --constrained
"""
import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repository root (holds models/ and tokenizers/)
sys.path.insert(0, HERE)
from aksharapiece.codec import Codec              # noqa: E402
from aksharapiece.models import LM, ModelConfig   # noqa: E402
from generate_eval import CLASSES, char_class, generate, script_mask  # noqa: E402


def load(name, root=ROOT, device="cpu"):
    """Load one model and its tokenizer from the repository folders."""
    cfg = json.load(open(os.path.join(root, "models", name, "config.json")))
    model = LM(ModelConfig(**cfg["model"]))
    weights = load_file(os.path.join(root, "models", name, "model.safetensors"))
    model.load_state_dict({k: v.float() for k, v in weights.items()})
    codec = Codec(os.path.join(root, "tokenizers", cfg["tokenizer_file"]))
    return model.to(device).eval(), codec


def complete(model, codec, prompts, max_new=64, temperature=0.8, top_p=0.95, seed=0, constrained=False, device="cpu"):
    """Continue each prompt. constrained=True blocks vowel signs and halants that cannot attach (no broken syllables)."""
    ids = codec.encode_batch(prompts)
    cons = plast = None
    if constrained:
        allowed, last = script_mask(codec.pieces(), model.c.vocab_size)
        cons = (allowed.to(device), last.to(device))
        plast = [CLASSES.index(char_class(codec.decode(p)[-1])) for p in ids]
    gens = generate(model, ids, codec.sep_id, max_new, temperature, top_p, seed, torch.device(device),
                    amp=torch.autocast(device_type=device, enabled=False), constraint=cons, prompt_last=plast)
    return [codec.decode(list(p) + list(g)) for p, g in zip(ids, gens)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="akshara_bpe_16k__transformer__s1")
    ap.add_argument("--prompt", default="नेपालको राजधानी")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--max-new", type=int, default=64)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, c = load(a.model, device=dev)
    print(complete(m, c, [a.prompt], max_new=a.max_new, constrained=a.constrained, device=dev)[0])
'''


def read_json_from_zip(z, suffix):
    for n in z.namelist():
        if n.endswith(suffix):
            return json.loads(z.read(n))
    return None


def model_card(user, models_repo, data_repo, github_url, rows):
    """README.md with YAML metadata for the model repository."""
    tbl = "\n".join(f"| `{r['name']}` | {r['tokenizer_label']} | {r['arch_label']} | {r['seed']} | {r['bpb']:.4f} |"
                    for r in rows)
    gh = f"[{github_url}]({github_url})" if github_url else "the project's GitHub repository"
    return f"""---
language:
- ne
license: apache-2.0
library_name: pytorch
tags:
- nepali
- devanagari
- tokenizer
- mamba
- mamba-2
- transformer
- low-resource
datasets:
- ai4bharat/IndicCorpV2
---

# AksharaPiece: Nepali tokenizers and small language models

Tokenizers and the 8 matched language models from the paper **"{PAPER_TITLE}"** ({AUTHORS}).

**AksharaPiece** is a BPE tokenizer that never splits a Devanagari syllable (*akshara*): every frequent
akshara is mapped to one symbol before BPE, so no token boundary can fall inside a syllable. It is compared
with **plain BPE** trained on the same text with the same settings.

Code, paper sources and all scripts: {gh}. Generated samples and per-line scores:
[`{user}/{data_repo}`](https://huggingface.co/datasets/{user}/{data_repo}).

## What is here

| Folder | Contents |
|---|---|
| `tokenizers/` | `plain_bpe_16k.json` (Hugging Face `tokenizers` file) and `akshara_bpe_16k.aksharapiece.json` (AksharaPiece) |
| `models/<name>/` | `model.safetensors` (float16 weights) and `config.json` for each of the 8 models |
| `code/` | the `aksharapiece` package, the generation code, and `example.py` |

## Models

All models: width 512, about 27M non-embedding parameters (35M with the tied 16k embedding), trained for
one pass over the same 600 MB of cleaned IndicCorp v2 Nepali text (about 48M tokens) on Kaggle T4 GPUs.
Bits per byte is measured on 5 MB of held-out IndicCorp text (lower is better).

| Model | Tokenizer | Architecture | Seed | Bits per byte |
|---|---|---|---|---|
{tbl}

The Transformer is Llama-style (8 layers, RoPE, RMSNorm, SwiGLU). Mamba-2 has 16 layers (state size 64) and
uses a plain-PyTorch implementation of the chunked SSD scan, included in `code/aksharapiece/models.py`.

## Main findings

- Keeping syllables whole did **not** change language-modelling quality (bits per byte differ by less than
  0.4%, with no consistent direction across seeds) or news-classification accuracy (np20ng, about 84%).
- Generated text from all models has fewer broken syllables than human-written text (0.12 to 0.28 versus 0.30
  per 1,000). AksharaPiece does not reduce them further.
- **Script-aware decoding** (blocking a vowel sign or halant that cannot attach) removes every broken syllable
  for both tokenizers. Use `constrained=True` in the example below.

## Usage

```bash
pip install torch safetensors tokenizers huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

root = snapshot_download("{user}/{models_repo}")
sys.path.insert(0, root + "/code")
from example import load, complete

model, codec = load("akshara_bpe_16k__transformer__s1", root=root)
print(complete(model, codec, ["नेपालको राजधानी"], constrained=True)[0])

# Tokenizer only
print(codec.encode("विद्यालयहरूमा पढ्ने विद्यार्थी"))
```

The tokenizers can also be used without the models: `aksharapiece.codec.Codec("tokenizers/akshara_bpe_16k.aksharapiece.json")`.

## Limitations

These are small research models trained on 600 MB of web text. They are meant for studying tokenization,
not for real applications: their output is often not factual or fluent, and IndicCorp is web-crawled text
that may contain offensive content.

## Citation

```bibtex
@misc{{bhattarai2026aksharapiece,
  title  = {{{PAPER_TITLE}}},
  author = {{Bhattarai, Sabin and Aadhikari, Nabin}},
  year   = {{2026}},
  note   = {{Models: https://huggingface.co/{user}/{models_repo}}}
}}
```
"""


def data_card(user, models_repo, data_repo, github_url):
    gh = f"[{github_url}]({github_url})" if github_url else "the project's GitHub repository"
    return f"""---
language:
- ne
license: cc-by-sa-4.0
tags:
- nepali
- devanagari
- tokenization
- evaluation
pretty_name: AksharaPiece evaluation outputs
---

# AksharaPiece evaluation outputs

Evaluation data from the paper **"{PAPER_TITLE}"** ({AUTHORS}). Models: [`{user}/{models_repo}`](https://huggingface.co/{user}/{models_repo}). Code: {gh}.

| Folder | Contents |
|---|---|
| `generation/free/` | 3,000 continuations per model with nucleus sampling (top-p 0.95, temperature 0.8), one file per model: `opening ||| continuation` |
| `generation/constrained/` | the same openings with script-aware decoding |
| `generation/*.json, *.md` | malformed-syllable and non-word counts, pooled tables, exact rate tests, every broken word |
| `lm_per_line/` | per-line negative log-likelihood on the 5 MB held-out set (`.lines.npy`) and bytes per line, for the paired bootstrap |

Openings are the first six words of FLORES Nepali sentences (Guzmán et al., 2019), so this dataset is shared
under CC BY-SA 4.0. Continuations are machine-generated by small research models and may be non-factual or
contain offensive text.
"""


def build(args):
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    mdir, ddir = out / "models_repo", out / "data_repo"
    (mdir / "models").mkdir(parents=True)
    (mdir / "tokenizers").mkdir()
    (mdir / "code").mkdir()

    # code needed to load and run the models
    shutil.copytree(ROOT / "aksharapiece", mdir / "code" / "aksharapiece",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "scripts" / "generate_eval.py", mdir / "code")
    (mdir / "code" / "example.py").write_text(EXAMPLE_PY, encoding="utf-8")

    # tokenizers (and per-line scores) from the LM results zip
    with zipfile.ZipFile(args.lm_zip) as z:
        for n in z.namelist():
            if n.startswith("tokenizers/") and not n.endswith("/"):
                (mdir / "tokenizers" / Path(n).name).write_bytes(z.read(n))
            if n.endswith(".lines.npy") or n.endswith("val_line_bytes.bin"):
                p = ddir / "lm_per_line" / Path(n).name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(z.read(n))
    tok_files = {p.name.replace(".aksharapiece.json", "").replace(".json", ""): p.name for p in (mdir / "tokenizers").iterdir()}

    # models: .pt -> safetensors + config.json
    import torch
    from safetensors.torch import save_file
    rows = []
    for pt in sorted(glob.glob(os.path.join(args.models_dir, "*.pt"))):
        ck = torch.load(pt, map_location="cpu")
        name = Path(pt).stem
        d = mdir / "models" / name
        d.mkdir()
        save_file({k: v.contiguous() for k, v in ck["state_dict"].items()}, str(d / "model.safetensors"))
        tok = ck["tokenizer"]
        parts = name.split("__")
        cfg = {"model": ck["config"], "tokenizer": tok, "tokenizer_file": tok_files.get(tok, ""),
               "bits_per_byte": ck.get("bpb"), "arch": parts[1] if len(parts) > 1 else ck["config"]["arch"],
               "seed": int(parts[2][1:]) if len(parts) > 2 else None, "training_text": "IndicCorp v2 Nepali, 600 MB, one pass",
               "sep_id": ck["config"]["vocab_size"] - 1}
        (d / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        rows.append({"name": name, "bpb": ck.get("bpb") or float("nan"), "seed": cfg["seed"],
                     "tokenizer_label": "AksharaPiece BPE 16k" if tok.startswith("akshara") else "Plain BPE 16k",
                     "arch_label": "Mamba-2" if cfg["arch"] == "mamba" else "Transformer"})
        print(f"  {name}: {sum(v.numel() for v in ck['state_dict'].values()) / 1e6:.1f}M params, bpb {cfg['bits_per_byte']}")
    if not rows:
        raise SystemExit(f"no .pt models found in {args.models_dir}")
    rows.sort(key=lambda r: (r["arch_label"], r["tokenizer_label"], r["seed"] or 0))
    (mdir / "README.md").write_text(model_card(args.user, args.models_repo, args.data_repo, args.github_url, rows), encoding="utf-8")

    # generation outputs
    if args.gen_zip and os.path.exists(args.gen_zip):
        with zipfile.ZipFile(args.gen_zip) as z:
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                if n.startswith("gen_free/"):
                    dest = ddir / "generation" / "free" / Path(n).name
                elif n.startswith("gen_constrained/"):
                    dest = ddir / "generation" / "constrained" / Path(n).name
                elif n in ("generation_summary.json", "generation_summary.md", "broken_words.txt"):
                    dest = ddir / "generation" / n
                else:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
    ddir.mkdir(parents=True, exist_ok=True)
    (ddir / "README.md").write_text(data_card(args.user, args.models_repo, args.data_repo, args.github_url), encoding="utf-8")
    for d in (mdir, ddir):
        files = [p for p in d.rglob("*") if p.is_file()]
        print(f"{d.name}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    return mdir, ddir


def upload(args, mdir, ddir):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for repo, folder, kind in ((args.models_repo, mdir, "model"), (args.data_repo, ddir, "dataset")):
        rid = f"{args.user}/{repo}"
        api.create_repo(rid, repo_type=kind, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=rid, repo_type=kind, folder_path=str(folder), ignore_patterns=["**/__pycache__/**", "*.pyc"],
                          commit_message="Release AksharaPiece " + ("models and tokenizers" if kind == "model" else "evaluation outputs"))
        url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{rid}"
        print(f"uploaded {kind}: {url}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--lm-zip", required=True, help="aksharapiece_lm_results.zip (tokenizers, per-line scores)")
    ap.add_argument("--gen-zip", default=None, help="aksharapiece_gen_results.zip (generated samples)")
    ap.add_argument("--out", default="release")
    ap.add_argument("--user", default=None, help="Hugging Face user name (default: from the token)")
    ap.add_argument("--models-repo", default="aksharapiece-nepali")
    ap.add_argument("--data-repo", default="aksharapiece-nepali-eval")
    ap.add_argument("--github-url", default="")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    if args.user is None:
        if args.upload:
            from huggingface_hub import HfApi
            args.user = HfApi(token=os.environ.get("HF_TOKEN")).whoami()["name"]
        else:
            args.user = "USERNAME"
    print("Hugging Face user:", args.user)
    mdir, ddir = build(args)
    if args.upload:
        upload(args, mdir, ddir)
    else:
        print("dry run: nothing uploaded (add --upload)")


if __name__ == "__main__":
    sys.exit(main())
