"""What the writing already decided — and what it left to the model.

These models do not predict "a vowel". They predict, per letter, one of ~75
**classes**, where a class is a whole combining-mark sequence. And 29 of those
classes carry a **hamza or a madda**, because rawi normalizes to NFD (⟨أ⟩ → alif
+ hamza) and therefore treats the hamza as a *mark on a bare alif* rather than as
a letter in its own right.

That is a defensible modelling choice. It also means **the model can change the
letter**, and it does:

    ⟨آ⟩ decomposes to alif + madda. Asked to classify that alif, the model often
    answers "hamza above". The word comes back as ⟨أ⟩ and the long /aː/ the madda
    stood for is simply gone — آبد /ʔaːbid/ becomes أَبْد /ʔabd/. On a 17.5k-word
    WikiPron sweep this happened to **6.5%** of words.

The same mechanism lets it **overwrite a human's tashkeel**: asked about a letter
that already carries a kasra, it answers with whatever class it likes, and كِتَاب
comes back as كَتَاب.

Neither is a hard problem. Both are the model being asked a question the text had
already answered.

## The fix, and why it is a mask and not a correction

Correcting the output afterwards works, but it is the wrong shape: the model was
allowed to make a choice that was never open. Instead we **remove the classes that
contradict the text before the argmax is taken**. A masked class cannot be chosen,
so the error is not repaired — it cannot occur.

The rule separates two kinds of mark, and the separation is exactly what an abjad
is:

* :data:`ORTHOGRAPHIC_MARKS` — hamza above, hamza below, madda. These are
  **written**. They are how the word is spelled. A reader does not restore them
  because they were never missing.
* :data:`VOCALIC_MARKS` — the ḥarakāt, shadda, tanwīn, dagger alef. These are
  **omitted**. They are what a reader supplies, and they are the entire reason a
  diacritizer exists.

Mask the first. Leave the second free.

## The rule is asymmetric, deliberately

Destroying written information is always wrong, so a class that drops or swaps a
mark the source carries is removed unconditionally.

*Adding* a mark to a letter that carries none is a different act — a bare alif
typed where ⟨أ⟩ was meant is a real and common misspelling, and restoring the
hamza is a documented capability of these models, not a bug. So where the source
is silent, the model stays free. Forbidding that would fix one bug by throwing
away a feature.
"""

from __future__ import annotations

import unicodedata
from typing import List, Sequence, Set, Tuple

__all__ = [
    "ORTHOGRAPHIC_MARKS",
    "VOCALIC_MARKS",
    "source_marks",
    "allowed_classes",
    "pinned_class",
    "project_class",
    "MADDA",
]

#: Marks that are part of the SPELLING: written, never omitted, never restored.
#: A model that changes one of these has changed the word, not marked it.
ORTHOGRAPHIC_MARKS = frozenset({
    "ٔ",  # hamza above  — أ = ا + this, ئ = ي + this
    "ٕ",  # hamza below  — إ
    "ٓ",  # madda above  — آ
})

#: The madda ⟨آ⟩ is vowel-complete: it already *is* /ʔaː/, so the letter carrying
#: it takes no ḥaraka of its own. A class that puts a fatḥa on a madda alif is not
#: a possible reading of Arabic, so it is not offered — otherwise projecting a
#: predicted "hamza + fatḥa" onto a madda would yield ⟨آَ⟩, which nobody writes.
MADDA = "ٓ"

#: Marks the writing OMITS: what the reader supplies, and what a diacritizer is for.
VOCALIC_MARKS = frozenset({
    "ً", "ٌ", "ٍ",  # tanwīn: fatḥatān, ḍammatān, kasratān
    "َ", "ُ", "ِ",  # fatḥa, ḍamma, kasra
    "ّ",                      # shadda
    "ْ",                      # sukūn
    "ٰ",                      # dagger alef
})


def _orthographic(marks: Sequence[str]) -> Set[str]:
    return {m for m in marks if m in ORTHOGRAPHIC_MARKS}


def source_marks(text: str, normalize) -> Tuple[str, List[Set[str]]]:
    """Split *text* into base characters and the marks sitting on each.

    *normalize* is the backend's own normalizer, so the decomposition here is the
    same one the model sees and the two index-align — which is the only reason a
    mask built from the source can be applied to the model's output at all.
    """
    bases: List[str] = []
    marks: List[Set[str]] = []
    for ch in normalize(text):
        if unicodedata.category(ch) == "Mn":
            if marks:
                marks[-1].add(ch)
        else:
            bases.append(ch)
            marks.append(set())
    return "".join(bases), marks


def allowed_classes(
    classes: Sequence[str],
    marks_here: Set[str],
    allow_restoration: bool = True,
) -> List[int]:
    """Which class ids may be chosen for a letter carrying *marks_here*.

    * The source spells an orthographic mark → only classes carrying **exactly**
      that set. The model may still choose any vowel it likes on top.
    * The source carries none → the model is free, and may restore one
      (``allow_restoration``), because a defectively-spelled alif is a real thing
      these models are built to fix.
    """
    here = _orthographic(marks_here)
    if MADDA in here:
        # A madda alif is already /ʔaː/. It takes no ḥaraka, so exactly one class
        # is possible and the model is not asked.
        return [i for i, c in enumerate(classes) if set(c) == {MADDA}]
    if here:
        return [i for i, c in enumerate(classes) if _orthographic(c) == here]
    if allow_restoration:
        return list(range(len(classes)))
    return [i for i, c in enumerate(classes) if not _orthographic(c)]


def pinned_class(classes: Sequence[str], marks_here: Set[str]) -> int:
    """The class that spells *marks_here* exactly, or ``-1`` if none does.

    Used where the source already carries vocalic marks: a human's tashkeel is
    the answer, not a prediction to be made, so the position is pinned to what is
    written and the model is not consulted.

    The vocabulary contains **duplicate-mark classes** — one of rawi's 73 spells
    kasra + kasra + shadda — which carry the same mark *set* as a well-formed
    class. Matching on the set alone would pin a letter to a class that writes its
    kasra twice, and the result would not even be idempotent. So among the classes
    with the right set, take the shortest: the one that says it once.
    """
    candidates = [i for i, c in enumerate(classes) if set(c) == marks_here]
    if not candidates:
        return -1
    return min(candidates, key=lambda i: len(classes[i]))


def project_class(
    classes: Sequence[str],
    predicted: int,
    marks_here: Set[str],
    allow_restoration: bool = True,
) -> int:
    """Move a predicted class into the subspace the writing allows.

    For a backend that hands back a **decision** rather than a distribution — the
    stitched ensemble emits one class id per letter, with the gating already
    folded into the graph — there are no logits left to mask. But the decision is
    still a *class*, and a class factors cleanly into two independent parts: the
    orthographic marks (which the writing fixed) and the vocalic ones (which the
    model was actually asked about).

    So keep the model's vocalic answer and swap its orthographic marks for the
    source's. The vowel decision survives intact; only the part it was never
    entitled to decide is overwritten. Where no class spells that combination,
    the prediction stands — inventing one would be worse than the error.
    """
    here = _orthographic(marks_here)
    predicted_marks = set(classes[predicted])
    predicted_ortho = _orthographic(predicted_marks)
    if MADDA in here:
        want = {MADDA}  # vowel-complete: no ḥaraka rides on a madda alif
    elif predicted_ortho == here:
        return predicted
    elif not here and allow_restoration:
        return predicted  # a silent letter: restoring a hamza is the model's job
    else:
        want = (predicted_marks - predicted_ortho) | here
    projected = pinned_class(classes, want)
    return projected if projected >= 0 else predicted
