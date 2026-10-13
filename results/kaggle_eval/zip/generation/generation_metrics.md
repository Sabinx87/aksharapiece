| Model | Tokenizer | Arch | Seed | Malformed / 1000 aksharas | Words with malformed syllable | Non-words | Words per sample |
|---|---|---|---|---|---|---|---|
| human-written reference | - | - | - | 0.30 | 0.09% | 4.37% | 33.6 |
| akshara_bpe_16k__mamba__s1 | akshara_bpe_16k | mamba | s1 | 0.09 | 0.02% | 2.80% | 13.4 |
| akshara_bpe_16k__mamba__s2 | akshara_bpe_16k | mamba | s2 | 0.09 | 0.03% | 3.36% | 13.2 |
| akshara_bpe_16k__transformer__s1 | akshara_bpe_16k | transformer | s1 | 0.09 | 0.03% | 3.31% | 13.0 |
| akshara_bpe_16k__transformer__s2 | akshara_bpe_16k | transformer | s2 | 0.18 | 0.05% | 2.87% | 13.9 |
| plain_bpe_16k__mamba__s1 | plain_bpe_16k | mamba | s1 | 0.31 | 0.08% | 3.20% | 12.1 |
| plain_bpe_16k__mamba__s2 | plain_bpe_16k | mamba | s2 | 1.05 | 0.28% | 3.38% | 13.0 |
| plain_bpe_16k__transformer__s1 | plain_bpe_16k | transformer | s1 | 0.49 | 0.13% | 4.05% | 12.7 |
| plain_bpe_16k__transformer__s2 | plain_bpe_16k | transformer | s2 | 0.49 | 0.13% | 3.78% | 12.6 |

**Mean over seeds**

| Tokenizer | Arch | Seeds | Malformed / 1000 | Malformed words | Non-words |
|---|---|---|---|---|---|
| akshara_bpe_16k | mamba | 2 | 0.09 | 0.03% | 3.08% |
| plain_bpe_16k | mamba | 2 | 0.68 | 0.18% | 3.29% |
| akshara_bpe_16k | transformer | 2 | 0.14 | 0.04% | 3.09% |
| plain_bpe_16k | transformer | 2 | 0.49 | 0.13% | 3.92% |
