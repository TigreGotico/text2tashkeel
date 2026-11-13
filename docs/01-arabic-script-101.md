# 1. Arabic script 101 (for people who read zero Arabic)

You do not need to learn Arabic to understand this library. You only need a
few facts about how the writing system works. This page gives you those
facts using Latin-letter analogies.

## 1.1 Arabic is an abjad: consonants are written, short vowels usually are not

English is an alphabet: writers write every vowel and consonant. "cat",
"cot", and "cut" each spell their vowel out.

Arabic is an abjad. In an abjad, the letters are mostly consonants, and the
short vowels are normally omitted. The reader fills them in from knowledge
of the language and the context, the same way an English reader instantly
reads "bld" as "build", "bold", "bald", "bled", or "blood" because the
sentence tells them which one fits.

Imagine English written as `Th ct st n th mt`, and you, the reader, must
recover "The cat sat on the mat." That is what reading everyday Arabic is
like. Skilled readers do it without effort.

Here is a worked example. The three Arabic letters ك ت ب (read
right-to-left, romanized k, t, b) form a consonant skeleton. Depending on
which vowels a reader mentally inserts, the same three letters can be read
as:

| Vowels you add | Romanized | Meaning |
|----------------|-----------|---------|
| a–a–a | kataba | "he wrote" |
| u–i–a | kutiba | "it was written" |
| a–a–a (doubled t) | kattaba | "he made (someone) write" |
| u–u (plural) | kutub | "books" |

Four different words share one written form. The text on the page, `كتب`,
is identical in all four cases. Context tells a human reader which word is
meant.

Arabic reads right-to-left. In `كتب`, the first (rightmost) letter is k,
then t, then b. Your terminal handles the direction automatically. You
never have to reverse anything yourself.

## 1.2 The vowel marks (tashkeel / harakat)

When Arabic does need to be unambiguous, such as in the Quran, in poetry, in
dictionaries, in children's books, and for language learners, small marks
are added above and below the consonants to write the vowels in. These marks
are called tashkeel (تشكيل, "forming" or "shaping") or harakat (حركات,
"movements"). Adding them is the task this library automates.

Here are the marks, with the base letter ب (b) shown carrying each one. You
do not need to memorize them. Notice only that each is a small stroke
attached to a normal letter.

| Mark | Name | Sound it adds | On letter b | Result |
|------|------|---------------|---------------|--------|
| ` ◌َ ` | fatha | short a | بَ | "ba" |
| ` ◌ُ ` | damma | short u | بُ | "bu" |
| ` ◌ِ ` | kasra | short i | بِ | "bi" |
| ` ◌ْ ` | sukun | (no vowel) | بْ | "b" (closes a syllable) |
| ` ◌ّ ` | shadda | (doubles the consonant) | بّ | "bb" |

Two things to note:

- Sukun means "there is no vowel here." It marks a consonant that closes a
  syllable, like the first s in the English word "as-tronaut".
- Shadda means "this consonant is doubled (held twice as long)." Compare
  Italian "pizza" with "piza", or English "un-natural" with "unatural": the
  doubled consonant is a real, meaning-changing difference in Arabic.

## 1.3 Nunation (tanween): the marks that add an n

Three more marks are doubled versions of fatha, damma, and kasra. They add
a final -n sound and signal that a noun is grammatically indefinite ("a
book" rather than "the book"). This is called nunation or tanween.

| Mark | Name | Sound | Meaning signal |
|------|------|-------|----------------|
| ` ◌ً ` | fathatan | -an | indefinite, accusative |
| ` ◌ٌ ` | dammatan | -un | indefinite, nominative |
| ` ◌ٍ ` | kasratan | -in | indefinite, genitive |

So kitāb plus dammatan becomes kitābun, "a book (as a subject)". A theme
already appears here: the vowel on the last letter of a word often encodes
grammar (subject, object, or possessive). This is exactly why diacritization
needs to understand the whole sentence, the subject of
[the next page](02-the-diacritization-problem.md).

## 1.4 Shadda combines with a vowel

A doubled consonant still needs a vowel. So shadda almost always appears
together with one of the vowel marks: shadda and fatha, shadda and damma,
and so on, stacked on the same letter.

Example: مُحَمَّد ("Muhammad"). The middle letter م (m) carries shadda plus
fatha, which is why the name has a doubled "mm": mu-ḥam-mad.

This is why the model (described on the next pages) uses 15 categories
instead of 8: the 8 single marks, plus the 6 common shadda-plus-vowel
combinations, plus one category for "no mark at all." Treating "shadda plus
fatha" as a single decision matches how the marks actually attach to a
letter. The full list is in the
[architecture page](03-architecture.md#the-15-classes).

## 1.5 What "diacritization" means, precisely

Putting it together, diacritization is this task:

> Given Arabic text with the vowel marks missing, or partially missing,
> predict the correct mark for every base letter, then attach those marks.

The base letters never change. Diacritization only adds marks on top of
them. That one-letter-in, one-decision-out structure is what makes a
character-level model the natural fit, as the architecture page shows.

```
Base letters (unchanged):   ك  ت  ب
Predicted marks:            َ   َ   َ      (fatha, fatha, fatha)
Combined output:           كَ  تَ  بَ   →  كَتَبَ  ("kataba", "he wrote")
```

---
[Home](index.md) · [Next →](02-the-diacritization-problem.md)
