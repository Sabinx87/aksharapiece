| Model | Malformed | Aksharas | per 1000 | Words with malformed syllable | Non-words | Words per sample |
|---|---|---|---|---|---|---|
| human-written reference | 156 | 512,377 | 0.30 | 0.09% | 4.24% | 33.6 |
| akshara_bpe_16k__mamba__s1 | 22 | 109,886 | 0.20 | 0.05% | 3.99% | 13.3 |
| akshara_bpe_16k__mamba__s2 | 19 | 98,401 | 0.19 | 0.05% | 4.33% | 12.0 |
| akshara_bpe_16k__transformer__s1 | 37 | 109,142 | 0.34 | 0.08% | 4.32% | 13.3 |
| akshara_bpe_16k__transformer__s2 | 24 | 107,921 | 0.22 | 0.05% | 4.27% | 13.3 |
| plain_bpe_16k__mamba__s1 | 19 | 96,311 | 0.20 | 0.05% | 4.08% | 11.9 |
| plain_bpe_16k__mamba__s2 | 21 | 107,661 | 0.20 | 0.05% | 3.98% | 13.0 |
| plain_bpe_16k__transformer__s1 | 9 | 107,659 | 0.08 | 0.02% | 4.31% | 13.3 |
| plain_bpe_16k__transformer__s2 | 17 | 108,531 | 0.16 | 0.04% | 4.00% | 13.4 |

**Pooled over seeds**

| Tokenizer | Arch | Decoding | Models | Malformed / aksharas | per 1000 | Words with malformed syllable | Non-words |
|---|---|---|---|---|---|---|---|
| akshara_bpe_16k | all | free | 4 | 102 / 425,350 | 0.240 | 0.058% | 4.23% |
| plain_bpe_16k | all | free | 4 | 66 / 420,162 | 0.157 | 0.041% | 4.09% |
| akshara_bpe_16k | mamba | free | 2 | 41 / 208,287 | 0.197 | 0.052% | 4.15% |
| plain_bpe_16k | mamba | free | 2 | 40 / 203,972 | 0.196 | 0.051% | 4.03% |
| akshara_bpe_16k | transformer | free | 2 | 61 / 217,063 | 0.281 | 0.064% | 4.30% |
| plain_bpe_16k | transformer | free | 2 | 26 / 216,190 | 0.120 | 0.031% | 4.15% |

**Exact rate tests (malformed syllables)**

| Comparison | Arch | Counts | Rate ratio | p-value |
|---|---|---|---|---|
| plain vs AksharaPiece (free decoding) | mamba | 40 vs 41 | 1.00 | 1 |
| plain vs AksharaPiece (free decoding) | transformer | 26 vs 61 | 0.43 | 0.00022 |
| plain vs AksharaPiece (free decoding) | all | 66 vs 102 | 0.66 | 0.0085 |
