"""04 — Hear the vowels (romanization for non-Arabic readers).

The whole point of diacritization is to make the short vowels explicit. If you
don't read Arabic, that's invisible — so this script romanizes the output into
Latin letters, showing the vowels the model added.

The transliteration is APPROXIMATE (a rough ALA-LC-ish scheme), just enough to
make the added vowels audible in your head. It is not a precise standard.

Run:  python examples/04_transliteration.py
"""

import unicodedata

from text2tashkeel import Diacritizer

# Base consonants (rough romanization).
CONSONANTS = {
    "ا": "ā", "أ": "a", "إ": "i", "آ": "ʾā", "ء": "ʾ", "ؤ": "ʾ", "ئ": "ʾ",
    "ب": "b", "ت": "t", "ة": "h", "ث": "th", "ج": "j", "ح": "ḥ", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "ṣ",
    "ض": "ḍ", "ط": "ṭ", "ظ": "ẓ", "ع": "ʿ", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h", "و": "w", "ي": "y", "ى": "ā",
}
# Diacritics (the vowels we care about).
VOWELS = {
    "َ": "a", "ُ": "u", "ِ": "i", "ً": "an", "ٌ": "un", "ٍ": "in",
    "ْ": "", "ّ": "·",  # sukun = no vowel; shadda shown as a gemination dot
}


def romanize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in CONSONANTS:
            out.append(CONSONANTS[ch])
        elif ch in VOWELS:
            # shadda doubles the previous consonant
            if ch == "ّ" and out:
                out.append(out[-1])
            else:
                out.append(VOWELS[ch])
        elif ch == " ":
            out.append(" ")
        # ignore anything else
    return "".join(out)


d = Diacritizer()
for s in [
    "بسم الله الرحمن الرحيم",
    "محمد رسول الله",
    "العلم نور",
]:
    diac = d.diacritize(s)
    print("bare        :", s)
    print("diacritized :", diac)
    print("romanized   :", romanize(diac))
    print()
