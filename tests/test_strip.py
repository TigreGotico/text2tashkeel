"""Diacritic stripping / normalization helpers."""

import unicodedata

from text2tashkeel._models import _is_arabic_letter, _strip


def test_strip_removes_all_harakat():
    diacritized = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
    bare = _strip(diacritized)
    # No combining marks (category Mn) survive.
    assert all(unicodedata.category(c) != "Mn" for c in bare)
    # Base letters and the space are preserved.
    assert "بسم" in bare and " " in bare


def test_strip_is_idempotent():
    s = "هذا كتاب مفيد"
    assert _strip(_strip(s)) == _strip(s)


def test_strip_leaves_bare_text_unchanged():
    bare = "محمد رسول الله"
    assert _strip(bare) == bare


def test_is_arabic_letter():
    assert _is_arabic_letter("ب")
    assert _is_arabic_letter("ي")
    assert not _is_arabic_letter(" ")
    assert not _is_arabic_letter(".")
    assert not _is_arabic_letter("A")
