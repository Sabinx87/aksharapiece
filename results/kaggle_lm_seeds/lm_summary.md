| Architecture | Tokenizer | Seeds | Bits per byte (mean ± sd) | Per seed | Bytes/token | Minutes/run | Tokens/s |
|---|---|---|---|---|---|---|---|
| mamba | akshara_bpe_16k | 2 | **0.5065** ± 0.0005 | s1: 0.5061, s2: 0.5068 | 12.31 | 96 | 8,510 |
| mamba | plain_bpe_16k | 2 | **0.5068** ± 0.0001 | s1: 0.5067, s2: 0.5068 | 12.50 | 95 | 8,470 |
| transformer | akshara_bpe_16k | 2 | **0.5146** ± 0.0034 | s1: 0.5169, s2: 0.5122 | 12.31 | 30 | 26,956 |
| transformer | plain_bpe_16k | 2 | **0.5148** ± 0.0023 | s1: 0.5165, s2: 0.5132 | 12.50 | 28 | 28,856 |
