Negative = akshara_bpe_16k needs fewer bits per byte than plain_bpe_16k (better).

| Architecture | Seed | BPB plain_bpe_16k | BPB akshara_bpe_16k | Difference | 95% interval | p (bootstrap) |
|---|---|---|---|---|---|---|
| mamba | 1 | 0.5067 | 0.5061 | -0.13% | [-0.20%, -0.05%] | 0.003 |
| mamba | 2 | 0.5068 | 0.5068 | +0.00% | [-0.08%, +0.08%] | 0.960 |
| mamba | **mean of 2** | | | **-0.06%** | [-0.87%, +0.74%] (t-interval over seeds) | 1/2 seeds favour akshara_bpe_16k |
| transformer | 1 | 0.5165 | 0.5169 | +0.09% | [+0.01%, +0.17%] | 0.020 |
| transformer | 2 | 0.5132 | 0.5122 | -0.20% | [-0.28%, -0.11%] | 0.001 |
| transformer | **mean of 2** | | | **-0.05%** | [-1.90%, +1.79%] (t-interval over seeds) | 1/2 seeds favour akshara_bpe_16k |
