## Table 1. Tokenizer audit on FLORES Nepali

| Tokenizer | Vocab | Tokens/word | Words with syllable cut | Tokens vs English |
|---|---|---|---|---|
| GPT-4 (cl100k) | 100,277 | 6.58 | 99.5% | 4.63× |
| Qwen 2.5 | 151,665 | 6.14 | 88.3% | 4.32× |
| Llama 3.2 | 128,256 | 3.54 | 88.3% | 2.49× |
| Mistral Tekken | 131,072 | 2.97 | 62.1% | 2.07× |
| Sarvam-1 | 68,096 | 2.38 | 32.9% | 1.47× |
| GPT-4o (o200k) | 200,019 | 2.24 | 41.0% | 1.60× |
| Gemma 3 | 262,145 | 2.08 | 21.8% | 1.51× |
| IndicBERT v2 | 250,000 | 1.56 | 11.2% | 1.10× |
| nepalitokenizers SP | 32,000 | 1.56 | 8.4% | n/a |
| MuRIL | 197,258 | 1.47 | 9.1% | 1.01× |
| AksharaPiece BPE 32k (ours) | 32,000 | 1.44 | 0.3% | n/a |
| Plain BPE 32k (ours) | 32,000 | 1.42 | 6.2% | n/a |

## Table 2. Bits per byte (5 MB held-out IndicCorp), mean ± sd over 3 seeds

| Architecture | Plain BPE 16k | AksharaPiece BPE 16k | Difference |
|---|---|---|---|
| Transformer | 0.5152 ± 0.0017 | 0.5136 ± 0.0029 | -0.31% ± 0.46 |
| Mamba-2 | 0.5070 ± 0.0004 | 0.5065 ± 0.0004 | -0.09% ± 0.08 |

## Table 3. News classification (np20ng), mean ± sd over 2 seeds

| Architecture | Plain BPE | AksharaPiece BPE |
|---|---|---|
| Transformer | 83.43 ± 0.11 acc / 83.38 F1 | 83.56 ± 0.37 acc / 83.54 F1 |
| Mamba-2 | 84.06 ± 0.51 acc / 84.01 F1 | 84.14 ± 0.48 acc / 84.08 F1 |

## Table 4. Generated-text validity (2 seeds × 3,000 continuations per cell)

Human-written reference: 156 malformed in 512,377 syllables (0.30/1000), non-words 4.24%.

| Architecture | Decoding | Plain BPE malformed | AksharaPiece malformed | Plain non-words | AksharaPiece non-words |
|---|---|---|---|---|---|
| Transformer | free | 26 / 216,190 | 61 / 217,063 | 4.15% | 4.30% |
| Transformer | constrained | 0 / 215,789 | 0 / 216,724 | 4.24% | 4.24% |
| Mamba-2 | free | 40 / 203,972 | 41 / 208,287 | 4.03% | 4.15% |
| Mamba-2 | constrained | 0 / 205,732 | 0 / 209,225 | 3.84% | 4.03% |
| Both | free | 66 / 420,162 | 102 / 425,350 | 4.09% | 4.23% |
| Both | constrained | 0 / 421,521 | 0 / 425,949 | 4.04% | 4.14% |

## Table 5. Malformed syllables in IndicCorp Nepali

5,572 malformed in 17,506,452 syllables (0.318/1000; 2.71% of lines). Two repair rules fix 34%.

## Table 6. Kinds of malformed syllables (share; count)

| Type | Training corpus | Plain BPE models | AksharaPiece models |
|---|---|---|---|
| Same vowel sign twice | 25% (1386) | 27% (18) | 18% (18) |
| Two different vowel signs | 26% (1446) | 18% (12) | 18% (18) |
| Nasal mark before vowel sign | 13% (731) | 14% (9) | 2% (2) |
| Sign after halant | 6% (317) | 8% (5) | 21% (21) |
| Sign at start of word | 10% (582) | 12% (8) | 24% (24) |
| Other | 20% (1110) | 21% (14) | 19% (19) |

Chi-square vs corpus: AksharaPiece: p = 2.79e-13; Plain BPE: p = 0.803; Plain vs AksharaPiece: p = 0.0039

