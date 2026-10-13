| Tokenizer | Architecture | Bits per byte | Loss/token | Bytes/token | Tokens seen (M) | Params (M, non-emb + emb) | Minutes | Tokens/s |
|---|---|---|---|---|---|---|---|---|
| akshara_bpe_16k | mamba | **0.5066** | 4.319 | 12.31 | 48.7 | 26.5 + 8.2 | 143.1 | 5,677 |
| plain_bpe_16k | mamba | **0.5074** | 4.395 | 12.50 | 48.0 | 26.5 + 8.2 | 146.2 | 5,469 |
| akshara_bpe_16k | transformer | **0.5116** | 4.362 | 12.31 | 48.7 | 27.3 + 8.2 | 115.0 | 7,061 |
| plain_bpe_16k | transformer | **0.5158** | 4.467 | 12.50 | 48.0 | 27.3 + 8.2 | 108.8 | 7,349 |
