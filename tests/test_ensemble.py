"""The agreement-gated ensemble ('bilstm+rawi')."""

import unicodedata

from text2tashkeel import Diacritizer

_HARAKAT = {chr(c) for c in range(0x64B, 0x653)}


def _mark_count(s: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFD", s) if c in _HARAKAT)


def test_ensemble_only_removes_value_marks():
    """Gating keeps rawi's mark *value* but only at gate-approved positions, so
    the ensemble can never have MORE marks than rawi alone."""
    rawi = Diacritizer("rawi")
    ens = Diacritizer("bilstm+rawi")
    for text in [
        "العلم نور والجهل ظلام",
        "بسم الله الرحمن الرحيم",
        "في التأني السلامة وفي العجلة الندامة",
        "هذا كتاب مفيد",
    ]:
        assert _mark_count(ens.diacritize(text)) <= _mark_count(rawi.diacritize(text))


def test_ensemble_differs_from_components():
    text = "العلم نور والجهل ظلام"
    out = Diacritizer("bilstm+rawi").diacritize(text)
    # It should actually do something, and differ from raw rawi here (rawi
    # over-marks this sentence, e.g. a stray mark on the space before والجهل).
    assert _mark_count(out) > 0
    assert out != Diacritizer("rawi").diacritize(text)


def test_ensemble_preserves_letters():
    text = "محمد رسول الله"
    out = Diacritizer("bilstm+rawi").diacritize(text)
    strip = lambda s: "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )
    assert strip(out).replace(" ", "") == strip(text).replace(" ", "")
