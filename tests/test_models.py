import pytest

torch = pytest.importorskip("torch")

from aksharapiece.models import LM, ModelConfig, ssd_reference, ssd_scan  # noqa: E402


@pytest.mark.parametrize("l,chunk", [(32, 8), (64, 16), (48, 48)])
def test_chunked_scan_matches_recurrence(l, chunk):
    torch.manual_seed(0)
    b, h, p, n = 2, 3, 4, 5
    X = torch.randn(b, l, h, p, dtype=torch.float64)
    A = -torch.rand(b, l, h, dtype=torch.float64) * 0.5
    B = torch.randn(b, l, h, n, dtype=torch.float64)
    C = torch.randn(b, l, h, n, dtype=torch.float64)
    assert torch.allclose(ssd_scan(X, A, B, C, chunk), ssd_reference(X, A, B, C), atol=1e-9)


@pytest.mark.parametrize("arch", ["transformer", "mamba"])
def test_lm_forward_backward_and_causality(arch):
    torch.manual_seed(0)
    c = ModelConfig(arch=arch, vocab_size=100, d_model=64, n_layers=2, n_heads=4, headdim=16,
                    d_state=8, chunk=16, max_seq_len=64)
    m = LM(c)
    x = torch.randint(0, 100, (2, 40))          # 40 is not a multiple of chunk: tests padding
    logits, loss = m(x, x)
    assert logits.shape == (2, 40, 100)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    # causality: changing a later token must not change earlier logits
    x2 = x.clone()
    x2[:, 30] = (x2[:, 30] + 1) % 100
    with torch.no_grad():
        a, b2 = m(x), m(x2)
    assert torch.allclose(a[:, :30], b2[:, :30], atol=1e-5)
    assert not torch.allclose(a[:, 30:], b2[:, 30:])


def test_default_sizes_are_comparable():
    t = LM(ModelConfig(arch="transformer", vocab_size=16001, n_layers=8)).param_counts()
    m = LM(ModelConfig(arch="mamba", vocab_size=16001, n_layers=16)).param_counts()
    assert 20e6 < t["non_embedding"] < 35e6
    assert 20e6 < m["non_embedding"] < 35e6
