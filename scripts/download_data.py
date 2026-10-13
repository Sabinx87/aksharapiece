"""Download the public data and tokenizer files used by the project (run where you have internet, e.g. Kaggle).

    python scripts/download_data.py            # FLORES v1 + Llama 2 tokenizer + IndicCorp v2 Nepali
    python scripts/download_data.py --no-indiccorp

IndicCorp v2 (CC-0) may ask you to log in to Hugging Face and accept its terms first:
    huggingface-cli login
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TOK = ROOT / "data" / "tokenizers"

FLORES_V1 = ("https://raw.githubusercontent.com/facebookresearch/flores/main/previous_releases/"
             "floresv1/data/wikipedia_en_ne_si_test_sets.tgz")
LLAMA2_SP = "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.model"


def fetch(url: str, dest: Path):
    if dest.exists():
        print(f"have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-indiccorp", action="store_true")
    args = ap.parse_args()

    tgz = RAW / "flores_v1.tgz"
    fetch(FLORES_V1, tgz)
    with tarfile.open(tgz) as t:
        try:
            t.extractall(RAW, filter="data")   # safe extraction (Python 3.12+)
        except TypeError:
            t.extractall(RAW)
    fetch(LLAMA2_SP, TOK / "llama2_tokenizer.model")

    if not args.no_indiccorp:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="ai4bharat/IndicCorpV2", filename="data/ne.txt",
                               repo_type="dataset", local_dir=str(RAW / "indiccorp_v2"))
        print(f"IndicCorp v2 Nepali at {path}")


if __name__ == "__main__":
    main()
