# 2. The diacritization problem (why it is hard)

If diacritization were just "look each word up in a dictionary and add its
vowels," a neural network would not be necessary. This page shows why the
task genuinely requires understanding the whole sentence.

## 2.1 The same spelling, many readings

Recall from [page 1](01-arabic-script-101.md) that the bare consonants كتب
(k-t-b) can be kataba ("he wrote"), kutiba ("it was written"), kutub
("books"), and more. The writing is identical. Only the vowels differ. So a
diacritizer must choose the reading, and the only way to choose correctly is
to look at the surrounding words.

```
... الكاتب كتب الرسالة     →  kataba    ("the writer WROTE the letter")  — a verb
... قرأت كتب التاريخ        →  kutub     ("I read history BOOKS")          — a noun
```

The same three letters take opposite parts of speech, decided entirely by
context. This is lexical ambiguity, and it is everywhere in Arabic.

## 2.2 Grammar lives in the last vowel (i'rab)

Arabic marks a word's grammatical role, such as subject, object, or
possessor, largely through the vowel on its final letter. This system is
called i'rab (إعراب, "case inflection"). The base letters of a noun stay the
same. Only the final mark changes:

| Final mark | Romanized ending | Grammatical role |
|------------|------------------|------------------|
| damma ` ُ ` | -u | subject (nominative) |
| fatha ` َ ` | -a | object (accusative) |
| kasra ` ِ ` | -i | possessive or after a preposition (genitive) |

Take the word al-walad ("the boy"):

```
الولدُ ذهب     →  al-waladU dhahaba    "the boy (SUBJECT) left"
رأيت الولدَ     →  ra'aytu al-waladA    "I saw the boy (OBJECT)"
بيت الولدِ      →  baytu al-waladI      "the house of the boy (POSSESSIVE)"
```

To pick the final mark, the model must figure out the word's syntactic job
in the sentence. That can depend on a verb five words earlier, or a
preposition right before it. There is no local rule. It is a parsing
problem.

## 2.3 Why this forces whole-sentence context

Put 2.1 and 2.2 together and the core design requirement follows:

> The correct diacritic for a character can depend on words anywhere in the
> sentence, to its left and to its right.

- To its right (later words): a verb appearing after a noun can reveal that
  the noun was its subject.
- To its left (earlier words): a preposition or an earlier verb sets up the
  grammatical case of what follows.

A model that reads left-to-right only, or that looks at each word in
isolation, systematically misses these cases. This is why the architecture
(next page) is built on a bidirectional network, reading context from both
sides so a letter's mark can depend on the words before and after it. One
bundled model, `bilstm`, adds an attention mechanism on top for direct
long-range links. The default rawi models rely on the bidirectional
recurrence alone.

## 2.4 Why character-level, not word-level

Modern NLP models often split text into subword pieces (for example,
"diacritization" becomes `dia`, `crit`, `ization`). That approach fails this
task, because diacritization needs one clean decision per base letter: each
Arabic letter gets exactly one mark, or none. Chopping a word into chunks
breaks the tidy alignment between a letter and its mark.

So this model works at the character level: one input symbol per base
letter, one output decision per base letter. The mapping is exact and never
needs to be guessed:

```
letters:    ك    ت    ب
decisions:  [?]  [?]  [?]      ← exactly one per letter, by construction
```

## 2.5 What "good" looks like: the metrics

Two numbers, both "lower is better," measure quality. They are defined
precisely in [the inference page](04-inference-pipeline.md#metrics) and
implemented in [`benchmarks/benchmark.py`](../benchmarks/benchmark.py):

- DER, Diacritic Error Rate. Of all the base characters, what fraction got
  the wrong mark? This is the fine-grained score.
- WER, Word Error Rate. Of all the words, what fraction had at least one
  character wrong? This score is stricter, because one slip anywhere ruins
  the whole word.

A perfect diacritizer scores 0% on both. For reference, even strong systems
score in the single-digit to low-double-digit DER range on hard test sets,
and a general-purpose LLM scores worse than a small purpose-built model like
this one, because, as the previous sections showed, this is a structured
linguistic task, not a free-form text rewrite.

---
[← Arabic script 101](01-arabic-script-101.md) · [Home](index.md) · [Next →](03-architecture.md)
