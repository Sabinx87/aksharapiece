| Tokenizer | Vocab | Tokens/word | Chars/token | Akshara breaks | Words broken | UNK | Rényi eff. | Parity vs EN |
|---|---|---|---|---|---|---|---|---|
| nepalitokenizers WordPiece | 30522 | 1.53 | 4.22 | 5.1% | 9.8% | 0.0% | 0.484 | 0.55 |
| nepalitokenizers SentencePiece (Unigram) | 32000 | 1.56 | 4.13 | 6.4% | 8.4% | 0.0% | 0.461 | 0.51 |
| GPT-4o (o200k_base) | 200019 | 2.24 | 2.88 | 28.2% | 41.0% | 0.0% | 0.441 | 1.60 |
| Mistral Tekken (2024) | 131072 | 2.97 | 2.17 | 35.5% | 62.1% | 0.0% | 0.431 | 2.07 |
| Aksharas, no merges (reference) |  | 4.01 | 1.60 | 0.0% | 0.0% | 0.0% | – | 0.58 |
| Unicode characters (reference) |  | 6.44 | 1.00 | 38.1% | 88.3% | 0.0% | – | 0.93 |
| GPT-4 (cl100k_base) | 100277 | 6.58 | 0.98 | 63.7% | 99.5% | 0.0% | 0.293 | 4.63 |
| Llama 2 (SentencePiece BPE) | 32000 | 6.99 | 0.92 | 43.7% | 90.9% | 0.0% | 0.277 | 4.34 |
| GPT-3 (p50k_base) | 50281 | 10.39 | 0.62 | 70.7% | 100.0% | 0.0% | 0.189 | 7.27 |
| UTF-8 bytes (no tokenizer) | 256 | 17.22 | 0.37 | 77.0% | 100.0% | 0.0% | 0.289 | 2.47 |
