"""Waqf (pausal form) — drop the iʿrāb a diacritizer restores.

A diacritizer trained on Tashkeela-class classical/MSA text restores the **full**
case and mood endings (iʿrāb): every word gets its ``-u`` / ``-a`` / ``-i`` or
its tanwīn. That is correct for a Qurʾānic or pedagogical text and wrong for
speech. Read aloud, Arabic **pauses in waqf form**: the final short vowel of a
word is not pronounced. Reading the endings out is the "stilted", over-formal
register that a native ear rejects immediately, and it is the single loudest
difference between text a model diacritized and text a person would say.

So this module is a deterministic transform from the fully-marked form to the
pausal one — no model, no data, no training::

    >>> pausal("كِتَابٌ جَمِيلٌ")
    'كِتَاب جَمِيل'
    >>> pausal("رَأَيْتُ كِتَابًا")          # tanwīn al-fatḥ survives as /aː/
    'رَأَيْتُ كِتَابَا'

## The rules

Word-finally, at a pause (Wright, *A Grammar of the Arabic Language*, 3rd ed.,
I §§ 372–374; Ryding, *A Reference Grammar of MSA*, CUP 2005, § 2.4):

1. A final short vowel — fatḥa, ḍamma, kasra — is **dropped**. This is the case
   or mood ending.
2. **Tanwīn ḍamm** (ٌ) and **tanwīn kasr** (ٍ) are **dropped** with their /n/.
3. **Tanwīn fatḥ** (ً) is the exception: it does **not** vanish, it lengthens —
   ``kitāban`` → ``kitābā``. The orthographic alif that carries it is already
   written, so the mark becomes a plain fatḥa and the alif does the rest.
4. **Long vowels, sukūn and shadda are untouched.** A shadda keeps its
   gemination; only the vowel riding on it goes (``ħaqqun`` → ``ħaqq``).
5. **Tāʾ marbūṭa** (ة) keeps its preceding vowel and loses its own ending.

## What this cannot know

Not every word-final short vowel is iʿrāb. A closed class of function words —
the pronouns ``هُوَ`` / ``هِيَ`` / ``أَنْتَ``, ``نَحْنُ``, the relative
``الَّذِي`` — carries a **lexical** final vowel that is part of the word, not a
case ending, and survives the pause. Telling the two apart in general needs
morphology this module does not have, so the exceptions are enumerated in
:data:`LEXICAL_FINAL_VOWEL` rather than inferred. The list is deliberately a
closed class: adding to it is safe, and guessing is not.

The transform is otherwise **purely orthographic and reversible-in-spirit** — it
only ever removes a mark or rewrites tanwīn fatḥ to fatḥa; it never adds a
vowel, and it never changes a letter.
"""

from __future__ import annotations

import re

__all__ = ["pausal", "LEXICAL_FINAL_VOWEL"]

FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SUKUN = "ْ"
SHADDA = "ّ"
TANWIN_FATH = "ً"
TANWIN_DAMM = "ٌ"
TANWIN_KASR = "ٍ"

#: The short vowels that a pause deletes word-finally (the case/mood endings).
SHORT_VOWELS = (FATHA, DAMMA, KASRA)

#: The tanwīn that a pause deletes outright, /n/ and all.
DROPPED_TANWIN = (TANWIN_DAMM, TANWIN_KASR)

#: Marks that survive a pause untouched.
KEPT = (SUKUN, SHADDA)

#: Words whose final short vowel is **lexical**, not a case/mood ending, and so
#: survives the pause. A closed class — pronouns, demonstratives, relatives and
#: a few particles. Keyed by the word *with* its diacritics stripped of the
#: final mark, so a match is exact rather than a prefix guess.
LEXICAL_FINAL_VOWEL = frozenset({
    "هُوَ", "هِيَ", "هُمَا", "هُمْ", "هُنَّ",          # 3rd-person pronouns
    "أَنْتَ", "أَنْتِ", "أَنْتُمَا", "أَنْتُمْ", "أَنْتُنَّ",  # 2nd-person
    "أَنَا", "نَحْنُ",                                  # 1st-person
    "الَّذِي", "الَّتِي", "الَّذِينَ", "اللَّاتِي",           # relatives
    "هَذَا", "هَذِهِ", "ذَلِكَ", "تِلْكَ",                  # demonstratives
    "مُنْذُ", "حَيْثُ", "قَبْلُ", "بَعْدُ",                 # adverbial particles
})

#: A word is the maximal run of Arabic letters and marks.
_WORD = re.compile(r"[ء-ٰٟٱ-ۓ]+")

#: Trailing marks: the diacritics sitting at the very end of a word.
_TRAILING_MARKS = re.compile(r"[ً-ْٰ]+$")


def _pausal_word(word: str) -> str:
    """Apply the pausal rules to one word.

    Works on the *set* of trailing marks rather than the last character: a
    geminate ending is written both ways in the wild — C + shadda + harakah and
    C + harakah + shadda — and only one of those puts the vowel last.
    """
    # Tanwīn al-fatḥ lengthens rather than vanishing: the carrying alif is
    # already written, so the mark degrades to a plain fatḥa (kitāban →
    # kitābā). It is handled first and by a whole-word search, because it sits
    # *before* the alif and so is not among the word's trailing marks at all.
    # Tanwīn only ever occurs word-finally, so this cannot fire mid-word.
    if TANWIN_FATH in word:
        return word.replace(TANWIN_FATH, FATHA)

    marks = _TRAILING_MARKS.search(word)
    if not marks:
        return word  # ends in a bare letter or a long vowel — nothing to drop

    tail = marks.group()
    stem = word[: marks.start()]

    # A lexical final vowel is part of the word, not an ending — leave it.
    if word in LEXICAL_FINAL_VOWEL:
        return word

    # The other two tanwīn go entirely, along with their /n/; and a final short
    # vowel is the case or mood ending, so it goes too. Either way the marks
    # that survive a pause — shadda's gemination, a sukūn — are kept, in the
    # order they were written: ḥaqqun → ḥaqq keeps its gemination.
    if any(t in tail for t in DROPPED_TANWIN) or any(v in tail for v in SHORT_VOWELS):
        return stem + "".join(ch for ch in tail if ch in KEPT)

    return word


#: A phrase ends at sentence punctuation or at the end of the string.
_PHRASE = re.compile(r"[^.!?،؛؟…\n]+|[.!?،؛؟…\n]+")

#: Sentence punctuation, Arabic and Latin.
_PUNCT = frozenset(".!?،؛؟…\n")


def pausal(text: str, phrase_final_only: bool = False) -> str:
    """Rewrite fully-diacritized *text* into its pausal (waqf) form.

    Drops the word-final case and mood endings a diacritizer restores, so the
    text reads the way it is *spoken* rather than the way it is parsed.
    Anything that is not an Arabic word — Latin, digits, punctuation,
    whitespace — passes through untouched.

    Args:
        phrase_final_only: apply the pause only where a pause actually falls —
            at the end of a phrase — which is *literal* waqf: mid-phrase words
            keep their endings and are read in connected form (waṣl). This is
            the classical recitation register.

            The default (``False``) drops the endings **throughout**, which is
            what the spoken language does: the modern dialects have no iʿrāb at
            all, and a leveled register reads a whole sentence without case
            endings, not just its last word. That — not classical waqf — is the
            register a TTS voice should speak, so it is the default.
    """
    if not phrase_final_only:
        return _WORD.sub(lambda m: _pausal_word(m.group()), text)

    out = []
    for chunk in _PHRASE.findall(text):
        if chunk and chunk[0] in _PUNCT:
            out.append(chunk)
            continue
        # Only the last Arabic word of the phrase stands at the pause.
        words = list(_WORD.finditer(chunk))
        if not words:
            out.append(chunk)
            continue
        last = words[-1]
        out.append(
            chunk[: last.start()]
            + _pausal_word(last.group())
            + chunk[last.end():]
        )
    return "".join(out)
