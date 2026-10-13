| Tokenizer | Vocab | Tokens/word | Chars/token | Akshara breaks | Words broken | UNK | Rényi eff. | Parity vs EN |
|---|---|---|---|---|---|---|---|---|
| nepalitokenizers WordPiece | 30522 | 1.33 | 4.84 | 2.9% | 5.3% | 0.0% | 0.482 | – |
| nepalitokenizers SentencePiece (Unigram) | 32000 | 1.35 | 4.76 | 3.5% | 4.2% | 0.0% | 0.480 | – |
| google/muril-base-cased | 197258 | 1.42 | 4.55 | 3.7% | 7.1% | 0.0% | 0.408 | – |
| ai4bharat/IndicBERTv2-MLM-only | 250000 | 1.44 | 4.47 | 3.9% | 7.9% | 0.0% | 0.401 | – |
| google/gemma-3-1b-it | 262145 | 2.05 | 3.15 | 11.4% | 19.1% | 0.0% | 0.407 | – |
| GPT-4o (o200k_base) | 200019 | 2.14 | 3.01 | 24.5% | 35.6% | 0.0% | 0.449 | – |
| sarvamai/sarvam-1 | 68096 | 2.36 | 2.73 | 15.1% | 27.4% | 0.0% | 0.441 | – |
| Mistral Tekken (2024) | 131072 | 3.03 | 2.13 | 33.1% | 59.7% | 0.0% | 0.435 | – |
| meta-llama/Llama-3.2-1B | 128256 | 3.59 | 1.80 | 54.7% | 86.1% | 0.0% | 0.380 | – |
| Aksharas, no merges (reference) |  | 4.09 | 1.58 | 0.0% | 0.0% | 0.0% | – | – |
| Qwen/Qwen2.5-0.5B | 151665 | 6.21 | 1.04 | 38.0% | 86.1% | 0.0% | 0.285 | – |
| Unicode characters (reference) |  | 6.45 | 1.00 | 36.8% | 86.1% | 0.0% | – | – |
| GPT-4 (cl100k_base) | 100277 | 6.64 | 0.97 | 63.0% | 99.6% | 0.0% | 0.294 | – |
| Llama 2 (SentencePiece BPE) | 32000 | 7.17 | 0.90 | 43.5% | 91.9% | 0.0% | 0.279 | – |
| GPT-3 (p50k_base) | 50281 | 10.39 | 0.62 | 69.7% | 100.0% | 0.0% | 0.192 | – |
| UTF-8 bytes (no tokenizer) | 256 | 17.28 | 0.37 | 76.5% | 100.0% | 0.0% | 0.290 | – |
