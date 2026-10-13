import unicodedata

import pytest

from aksharapiece.akshara import aksharas, akshara_boundaries, normalize

CASES = [
    # word, expected aksharas
    ("नमस्ते", ["न", "म", "स्ते"]),
    ("नेपाल", ["ने", "पा", "ल"]),
    ("नेपालको", ["ने", "पा", "ल", "को"]),
    ("क्षेत्र", ["क्षे", "त्र"]),
    ("विद्यालयहरूमा", ["वि", "द्या", "ल", "य", "ह", "रू", "मा"]),
    ("काठमाडौं", ["का", "ठ", "मा", "डौं"]),
    ("गाउँ", ["गा", "उँ"]),                      # independent vowel + chandrabindu
    ("छन्", ["छ", "न्"]),                         # word-final halant stays with its consonant
    ("स्त्री", ["स्त्री"]),                        # three-consonant conjunct
    ("अर्थ", ["अ", "र्थ"]),                        # reph
    ("दुःख", ["दुः", "ख"]),                        # visarga
    ("ॐ", ["ॐ"]),
    ("र्‍या", ["र्‍या"]),               # eyelash ra with ZWJ stays one unit
    ("क्‌ष", ["क्‌", "ष"]),             # ZWNJ forces a visible halant: two units
    ("१२३", ["१", "२", "३"]),
    ("AI र नेपाली", ["A", "I", " ", "र", " ", "ने", "पा", "ली"]),
]


@pytest.mark.parametrize("word,expected", CASES)
def test_aksharas(word, expected):
    assert aksharas(word) == expected


@pytest.mark.parametrize("word,_", CASES)
def test_roundtrip(word, _):
    assert "".join(aksharas(word)) == word


def test_nukta_is_decomposed_by_nfc():
    precomposed = "क़"  # क़ as one code point
    out = normalize(precomposed)
    assert out == "क़"
    assert aksharas(out) == ["क़"]


def test_orphan_vowel_sign_is_its_own_unit():
    # A vowel sign with no consonant (bad text) must not glue onto a space.
    assert aksharas(" ा") == [" ", "ा"]


def test_byte_boundaries():
    text = "नमस्ते"
    cuts = akshara_boundaries(text, unit="byte")
    # न(3 bytes) म(3) स्ते(4 code points = 12 bytes)
    assert cuts == {0, 3, 6, 18}


def test_malformed_detection():
    from aksharapiece.akshara import malformed_aksharas, script_validity
    assert malformed_aksharas("नेपालको राजधानी काठमाडौं हो") == []
    assert malformed_aksharas("गाउँ दुःख स्त्री छन् र्‍या") == []
    assert malformed_aksharas("क ा") == ["ा"]              # vowel sign after a space
    assert malformed_aksharas("काे") == ["काे"]             # two vowel signs on one syllable
    assert malformed_aksharas("क्ा") == ["ा"]              # vowel sign after a halant
    v = script_validity("राम ा घर")
    assert v["malformed"] == 1 and v["malformed_words"] == 1 and v["words"] == 3


def test_vowel_sign_typo_fixed():
    assert normalize("न्थ्याे") == "न्थ्यो"
    assert normalize("गयाै") == "गयौ"


def test_normalize_digits_and_spaces():
    assert normalize("२०८२  साल", devanagari_digits="to_ascii") == "2082 साल"
    assert normalize("abc­ def") == "abc def"


def test_every_devanagari_letter_roundtrips():
    block = "".join(chr(c) for c in range(0x0900, 0x0980) if unicodedata.category(chr(c)) != "Cn")
    assert "".join(aksharas(block)) == block
