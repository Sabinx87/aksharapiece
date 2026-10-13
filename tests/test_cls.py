import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finetune_cls import macro_f1, make_split  # noqa: E402


def test_macro_f1():
    assert macro_f1([0, 1, 2], [0, 1, 2], 3) == 1.0
    assert abs(macro_f1([0, 0, 1, 1], [0, 0, 0, 0], 2) - (2 / 3) / 2) < 1e-9


def test_split_is_balanced_and_disjoint():
    rows = [(f"नेपाली समाचार नम्बर {i} " * 5, "क" if i % 2 else "ख") for i in range(400)]
    tr, te, labels = make_split(rows, n_train=50, n_test=20, max_chars=600)
    assert labels == ["क", "ख"]
    assert len(tr) == 100 and len(te) == 40
    assert not {d for d, _ in tr} & {d for d, _ in te}
    assert make_split(rows, 50, 20, 600) == (tr, te, labels)   # deterministic


def test_generation_scored_with_prompt():
    from generate_eval import generated_part, text_metrics
    # continuation that extends the prompt's last word is valid
    ws, a, m = generated_part("लोग बसाइ", "लोग बसाइँसराइ गर्छन्")
    assert m == 0 and ws == ["बसाइँसराइ", "गर्छन्"]
    # a vowel sign after a space is broken
    ws, a, m = generated_part("लोग बसाइ", "लोग बसाइ ेदार")
    assert m == 1 and ws == ["ेदार"]
    r = text_metrics([("क ख", "क ख गा"), ("क ख", "क खाे")], None, 3)
    assert r["malformed"] == 1 and r["words"] == 2
