Human-written reference: 156 malformed in 512,377 aksharas (0.304/1000), non-words 4.24%

| Tokenizer | Arch | Decoding | Models | Malformed / aksharas | per 1000 | Words with malformed syllable | Non-words |
|---|---|---|---|---|---|---|---|
| akshara_bpe_16k | all | constrained | 4 | 0 / 425,949 | 0.000 | 0.000% | 4.14% |
| plain_bpe_16k | all | constrained | 4 | 0 / 421,521 | 0.000 | 0.000% | 4.04% |
| akshara_bpe_16k | mamba | constrained | 2 | 0 / 209,225 | 0.000 | 0.000% | 4.03% |
| plain_bpe_16k | mamba | constrained | 2 | 0 / 205,732 | 0.000 | 0.000% | 3.84% |
| akshara_bpe_16k | transformer | constrained | 2 | 0 / 216,724 | 0.000 | 0.000% | 4.24% |
| plain_bpe_16k | transformer | constrained | 2 | 0 / 215,789 | 0.000 | 0.000% | 4.24% |
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
| plain vs AksharaPiece (constrained) | mamba | 0 vs 0 | nan | 1 |
| plain: free vs constrained | mamba | 40 vs 0 | nan | 7.7e-13 |
| AksharaPiece: free vs constrained | mamba | 41 vs 0 | nan | 4.1e-13 |
| plain vs AksharaPiece (free decoding) | transformer | 26 vs 61 | 0.43 | 0.00022 |
| plain vs AksharaPiece (constrained) | transformer | 0 vs 0 | nan | 1 |
| plain: free vs constrained | transformer | 26 vs 0 | nan | 3e-08 |
| AksharaPiece: free vs constrained | transformer | 61 vs 0 | nan | 8.7e-19 |
| plain vs AksharaPiece (free decoding) | all | 66 vs 102 | 0.66 | 0.0085 |
| plain vs AksharaPiece (constrained) | all | 0 vs 0 | nan | 1 |
| plain: free vs constrained | all | 66 vs 0 | nan | 1.2e-20 |
| AksharaPiece: free vs constrained | all | 102 vs 0 | nan | 1.8e-31 |
