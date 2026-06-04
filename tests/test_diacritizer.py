"""Behavioural tests for the public Diacritizer API across all models."""

import unicodedata

import pytest

from text2tashkeel import Diacritizer, available_models, diacritize

MODELS = available_models()


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def _strip_marks(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")


def test_available_models():
    assert MODELS == [
        "bilstm", "bilstm-int8", "rawi", "libtashkeel",
        "bilstm+rawi", "libtashkeel+rawi", "ensemble",
    ]


@pytest.mark.parametrize("model", MODELS)
def test_adds_diacritics(model):
    d = Diacritizer(model)
    out = d.diacritize("بسم الله الرحمن الرحيم")
    # Output must contain combining marks that the input lacked.
    assert any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", out))


@pytest.mark.parametrize("model", MODELS)
def test_preserves_base_letters(model):
    """Diacritization must only ADD marks; the consonant skeleton is unchanged."""
    text = "محمد رسول الله"
    out = Diacritizer(model).diacritize(text)
    # Stripping marks from the output recovers the input letters (modulo NFC).
    assert _strip_marks(out).replace(" ", "") == _strip_marks(text).replace(" ", "")


@pytest.mark.parametrize("model", MODELS)
def test_empty_and_whitespace(model):
    d = Diacritizer(model)
    assert d.diacritize("") == ""
    # Whitespace-only input must not crash and must not invent letters.
    out = d.diacritize("   ")
    assert _strip_marks(out).strip() == ""


@pytest.mark.parametrize("model", MODELS)
def test_idempotent_on_already_diacritized(model):
    """Re-diacritizing our own output yields the same string (we strip first)."""
    d = Diacritizer(model)
    once = d.diacritize("هذا كتاب مفيد")
    twice = d.diacritize(once)
    assert _nfc(once) == _nfc(twice)


@pytest.mark.parametrize("model", MODELS)
def test_non_arabic_passthrough(model):
    """Latin / punctuation should survive (not crash, not vanish)."""
    out = Diacritizer(model).diacritize("OK? نعم.")
    assert "OK" in out or "?" in out  # at least the ASCII is not destroyed


def test_callable_alias():
    d = Diacritizer()
    assert d("نص") == d.diacritize("نص")


def test_module_convenience():
    assert diacritize("نص عربي") == Diacritizer("bilstm").diacritize("نص عربي")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        Diacritizer("does-not-exist")
