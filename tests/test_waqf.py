"""The pausal (waqf) transform: drop the iʿrāb a diacritizer restores."""
import pytest

from text2tashkeel import Diacritizer, pausal


# ─── the rules ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("marked,spoken,rule", [
    # A final short vowel is the case/mood ending — it goes.
    ("كِتَابٌ", "كِتَاب", "final ḍamma (nominative) dropped"),
    ("فِي الْبَيْتِ", "فِي الْبَيْت", "final kasra (genitive) dropped"),
    ("بَيْتٌ", "بَيْت", "tanwīn ḍamm dropped with its /n/"),
    ("كِتَابٌ جَمِيلٌ", "كِتَاب جَمِيل", "every word, not just the last"),
    # Tanwīn al-fatḥ is the exception: it lengthens rather than vanishing.
    ("كِتَابًا", "كِتَابَا", "tanwīn fatḥ → fatḥa, the alif carries /aː/"),
    ("مَرْحَبًا", "مَرْحَبَا", "tanwīn fatḥ → fatḥa"),
    # Gemination survives — only the vowel riding on the shadda goes.
    ("الْحَقُّ", "الْحَقّ", "shadda kept, ḍamma dropped"),
    ("مُدَرِّسٌ", "مُدَرِّس", "internal shadda untouched"),
    # Tāʾ marbūṭa keeps its preceding vowel, loses its own ending.
    ("مَدْرَسَةٌ", "مَدْرَسَة", "tāʾ marbūṭa ending dropped"),
    # Sukūn and long vowels are not endings.
    ("بِكُمْ", "بِكُمْ", "sukūn is not a case ending"),
    ("فِي", "فِي", "a long vowel is not a case ending"),
])
def test_pausal_rules(marked, spoken, rule):
    assert pausal(marked) == spoken, rule


@pytest.mark.parametrize("word", ["هُوَ", "هِيَ", "نَحْنُ", "أَنْتَ", "الَّذِي", "حَيْثُ"])
def test_lexical_final_vowels_survive(word):
    """A pronoun's final vowel is part of the word, not a case ending."""
    assert pausal(word) == word


# ─── scope ──────────────────────────────────────────────────────────────

def test_non_arabic_passes_through():
    assert pausal("BMW X5 2024 — 100%") == "BMW X5 2024 — 100%"


def test_mixed_script_only_touches_the_arabic():
    assert pausal("سَيَّارَةُ BMW") == "سَيَّارَة BMW"


def test_undiacritized_text_is_unchanged():
    """Nothing to drop — the transform never adds or guesses."""
    assert pausal("كتاب جميل") == "كتاب جميل"


def test_idempotent():
    once = pausal("كِتَابٌ جَمِيلٌ جِدًّا")
    assert pausal(once) == once


# ─── phrase_final_only: literal classical waqf ──────────────────────────

def test_phrase_final_only_pauses_at_the_pause():
    """Literal waqf: only the word standing at the pause loses its ending."""
    assert pausal("كِتَابٌ جَمِيلٌ", phrase_final_only=True) == "كِتَابٌ جَمِيل"


def test_phrase_final_only_respects_sentence_punctuation():
    out = pausal("جَاءَ الْوَلَدُ. رَأَيْتُ الْبِنْتَ.", phrase_final_only=True)
    assert out == "جَاءَ الْوَلَد. رَأَيْتُ الْبِنْت."


# ─── the Diacritizer surface ────────────────────────────────────────────

def test_diacritizer_waqf_flag_suppresses_irab():
    """The model restores the full endings; waqf=True takes them back off."""
    full = Diacritizer("rawi-v2-int8")
    spoken = Diacritizer("rawi-v2-int8", waqf=True)
    text = "بسم الله الرحمن الرحيم"
    assert spoken.diacritize(text) == pausal(full.diacritize(text))


def test_diacritizer_defaults_to_the_full_form():
    d = Diacritizer("rawi-v2-int8")
    assert d.waqf is False
