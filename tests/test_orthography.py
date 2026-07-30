"""The model may restore the vowels. It may not rewrite the word.

These models normalize to NFD and treat the hamza as a *mark on a bare alif*, so
29 of their ~75 classes carry a hamza or a madda. That means they can change the
letter, and they do — and they can overwrite a mark a human wrote, and they do.
Both are the model answering a question the text had already answered.
"""
import pytest

from text2tashkeel import Diacritizer
from text2tashkeel.orthography import (
    MADDA, ORTHOGRAPHIC_MARKS, VOCALIC_MARKS, allowed_classes, pinned_class,
    project_class,
)

MODELS = ["rawi-ensemble", "rawi-v2-int8"]


def unconstrained(model):
    return Diacritizer(model, preserve_orthography=False, respect_existing=False)


# ─── the letter is not the model's to change ────────────────────────────

@pytest.mark.parametrize("model", MODELS)
def test_a_madda_survives(model):
    """⟨آ⟩ is /ʔaː/. Unconstrained the model rewrites it as ⟨أ⟩ and the length is
    simply gone — آبد /ʔaːbid/ comes back as أَبْد /ʔabd/."""
    assert Diacritizer(model).diacritize("آبد").startswith("آ")
    assert unconstrained(model).diacritize("آبد").startswith("أ")


@pytest.mark.parametrize("model", MODELS)
def test_a_madda_alif_takes_no_haraka(model):
    """It is already a long vowel. ⟨آَ⟩ is not a thing anyone writes."""
    out = Diacritizer(model).diacritize("آباء")
    assert not any(m in out[1:2] for m in VOCALIC_MARKS)


@pytest.mark.parametrize("model", MODELS)
def test_a_written_hamza_survives(model):
    assert "أ" in Diacritizer(model).diacritize("مأمور")


# ─── a human's marks are evidence, not a prediction ─────────────────────

@pytest.mark.parametrize("model", MODELS)
def test_existing_marks_are_not_overwritten(model):
    """كِتَاب carries a kasra. Unconstrained, it comes back كَتَاب."""
    assert Diacritizer(model).diacritize("كِتَاب").startswith("كِ")
    assert unconstrained(model).diacritize("كِتَاب").startswith("كَ")


# ─── but restoration is the job, not a bug ──────────────────────────────

@pytest.mark.parametrize("model", MODELS)
def test_a_hamza_is_still_restored_on_a_silent_letter(model):
    """A bare alif typed where أ was meant is a real misspelling these models fix.

    The rule is asymmetric on purpose: dropping a mark the writing carries is
    always wrong; adding one to a letter that carries none is the task.
    """
    assert Diacritizer(model).diacritize("امير").startswith("أ")


@pytest.mark.parametrize("model", MODELS)
def test_restoration_can_be_turned_off(model):
    d = Diacritizer(model, allow_restoration=False)
    assert not d.diacritize("امير").startswith("أ")


# ─── the class algebra ──────────────────────────────────────────────────

def test_a_madda_position_admits_exactly_one_class():
    classes = ["", "َ", MADDA, MADDA + "َ", "ٔ"]
    assert allowed_classes(classes, {MADDA}) == [2]


def test_a_written_hamza_constrains_the_class_but_not_the_vowel():
    classes = ["", "َ", "ٔ", "ٔ" + "َ", MADDA]
    allowed = allowed_classes(classes, {"ٔ"})
    assert set(allowed) == {2, 3}  # both hamza classes; the vowel stays open


def test_a_silent_letter_leaves_the_model_free():
    classes = ["", "َ", "ٔ", MADDA]
    assert allowed_classes(classes, set(), allow_restoration=True) == [0, 1, 2, 3]
    assert allowed_classes(classes, set(), allow_restoration=False) == [0, 1]


def test_projection_keeps_the_vowel_and_fixes_the_letter():
    """The half the model was asked about survives; the half it wasn't is corrected."""
    classes = ["", "َ", "ٔ", "ٔ" + "َ", MADDA + "َ", MADDA]
    # predicted "hamza + fatḥa" on a letter the source spells with a madda
    assert project_class(classes, 3, {MADDA}) == 5  # → madda alone, no ḥaraka


def test_pin_prefers_the_class_that_says_it_once():
    """rawi's vocabulary contains duplicate-mark classes (kasra+kasra+shadda).

    They carry the same mark SET as a well-formed class, so matching on the set
    alone pins a letter to a class that writes its kasra twice — and the result
    is not even idempotent.
    """
    classes = ["", "ِّ", "ِِّ"]
    assert pinned_class(classes, {"ِ", "ّ"}) == 1


def test_the_two_mark_classes_do_not_overlap():
    assert not (ORTHOGRAPHIC_MARKS & VOCALIC_MARKS)
