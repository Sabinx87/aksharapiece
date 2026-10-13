import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aksharapiece.models import LM, ModelConfig  # noqa: E402
from train_lm import evaluate  # noqa: E402
from significance import bootstrap_diff  # noqa: E402


def test_per_line_nll_sums_to_total():
    torch.manual_seed(0)
    sep = 50
    rng = np.random.default_rng(0)
    stream = []
    for _ in range(37):                                   # 37 lines of random length, each ending in sep
        stream += list(rng.integers(0, 50, rng.integers(1, 30))) + [sep]
    val = np.array(stream, dtype=np.uint32)
    m = LM(ModelConfig(arch="transformer", vocab_size=51, d_model=32, n_layers=1, n_heads=2, max_seq_len=16))
    amp = torch.autocast("cpu", enabled=False)
    total, ntok, per_line = evaluate(m, val, 16, 4, torch.device("cpu"), amp, sep_id=sep)
    assert len(per_line) == 37
    assert ntok == len(val) - 1
    assert per_line.sum() == pytest.approx(total, rel=1e-6)


def test_bootstrap_detects_a_clear_difference():
    rng = np.random.default_rng(1)
    a = rng.uniform(50, 100, 2000)
    b = a * 0.98 + rng.normal(0, 0.5, 2000)               # b is ~2% better on every line
    point, lo, hi, p = bootstrap_diff(a, b, np.ones(2000), n_boot=500)
    assert -2.2 < point < -1.8 and hi < 0 and p < 0.01
