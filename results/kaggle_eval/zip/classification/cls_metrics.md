20 classes, 20,000 train / 4,000 test documents, first 600 characters.

| Model | Accuracy | Macro-F1 | Tokens per doc | Truncated |
|---|---|---|---|---|
| akshara_bpe_16k__mamba__s1 | 84.47% | 84.43 | 124 | 0.0% |
| akshara_bpe_16k__mamba__s2 | 83.80% | 83.73 | 124 | 0.0% |
| akshara_bpe_16k__transformer__s1 | 83.83% | 83.81 | 124 | 0.0% |
| akshara_bpe_16k__transformer__s2 | 83.30% | 83.26 | 124 | 0.0% |
| plain_bpe_16k__mamba__s1 | 84.42% | 84.39 | 122 | 0.0% |
| plain_bpe_16k__mamba__s2 | 83.70% | 83.62 | 122 | 0.0% |
| plain_bpe_16k__transformer__s1 | 83.50% | 83.47 | 122 | 0.0% |
| plain_bpe_16k__transformer__s2 | 83.35% | 83.30 | 122 | 0.0% |

**Mean ± sd over pretraining seeds**

| Tokenizer | Arch | Seeds | Accuracy | Macro-F1 |
|---|---|---|---|---|
| akshara_bpe_16k | mamba | 2 | 84.14 ± 0.48 | 84.08 ± 0.50 |
| plain_bpe_16k | mamba | 2 | 84.06 ± 0.51 | 84.01 ± 0.54 |
| akshara_bpe_16k | transformer | 2 | 83.56 ± 0.37 | 83.54 ± 0.39 |
| plain_bpe_16k | transformer | 2 | 83.43 ± 0.11 | 83.38 ± 0.12 |
