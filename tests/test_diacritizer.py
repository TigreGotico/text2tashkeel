"""Behavioural tests for the public Diacritizer API across all models."""

import unicodedata

import pytest

from text2tashkeel import Diacritizer, available_models, diacritize, DEFAULT_MODEL

MODELS = available_models()


def _nfc(s):
    return unicodedata.normalize("NFC", s)


def _strip_marks(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                    if unicodedata.category(c) != "Mn")


def test_available_models():
    assert MODELS == [
        "bilstm", "bilstm-int8", "rawi", "rawi-int8", "rawi-v2", "rawi-v2-int8", "rawi-v3", "rawi-v3-int8",
        "libtashkeel",
        "bilstm+rawi", "bilstm+rawi-int8",
        "libtashkeel+rawi", "libtashkeel+rawi-int8", "bilstm-int8+rawi-int8",
        "bilstm+libtashkeel+rawi", "bilstm+libtashkeel+rawi-int8",
        "rawi-v2+rawi", "rawi-v2+rawi-int8", "rawi-v2-int8+rawi-int8",
        "rawi-v2+rawi-v3", "rawi-v2-int8+rawi-v3-int8",
        "rawi-ensemble",
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
    assert diacritize("نص عربي") == Diacritizer(DEFAULT_MODEL).diacritize("نص عربي")


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        Diacritizer("does-not-exist")


# ── bundling / download / bring-your-own ─────────────────────────────────────
from text2tashkeel import BUNDLED, register_model            # noqa: E402
from text2tashkeel._models import _model_path, _MODELS_DIR   # noqa: E402


def test_default_is_bundled():
    # Diacritizer() must work fully offline → the default's weights ship in the wheel.
    assert DEFAULT_MODEL in BUNDLED


def test_bundled_only_subset():
    bundled = set(available_models(bundled_only=True))
    assert bundled == set(BUNDLED) <= set(available_models())
    # every bundled model's files are actually present in the package
    for name in bundled:
        Diacritizer(name).diacritize("نص")        # constructs + runs, no download


def test_model_path_unknown_file_errors():
    with pytest.raises(FileNotFoundError):
        _model_path("not_a_real_model.onnx")       # not bundled, no HF source


def test_register_model_byo():
    # point at our own bundled rawi-v2 files via the public API → must match rawi-v2-int8
    from text2tashkeel._models import _REGISTRY, _build
    try:
        register_model(
            "byo-test",
            _MODELS_DIR / "rawi_v2.int8.onnx",
            _MODELS_DIR / "rawi_v2.vocab.json",
            arch="rawi",
        )
        assert "byo-test" in available_models()
        txt = "بسم الله الرحمن الرحيم"
        assert Diacritizer("byo-test").diacritize(txt) == Diacritizer("rawi-v2-int8").diacritize(txt)
    finally:
        _REGISTRY.pop("byo-test", None)
        _build.cache_clear()


def test_register_model_bad_arch():
    with pytest.raises(ValueError):
        register_model("x", "a.onnx", "b.json", arch="nope")
