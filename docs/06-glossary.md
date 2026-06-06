# 6. Glossary

Quick definitions of every Arabic-linguistics and machine-learning term used in
these docs. Arabic terms are romanized.

## Arabic & linguistics

**Abjad** — a writing system that records mostly consonants and leaves short
vowels to the reader. Arabic and Hebrew are abjads. Contrast with an *alphabet*
(records all vowels, like Latin).

**Diacritic** — a small mark added above or below a letter. In Arabic, the
diacritics write the short vowels and a few other features. Collectively
*tashkeel* / *harakat*.

**Diacritization** — the task of adding the correct diacritics to undiacritized
text. The whole point of this library. Also called *tashkeel*.

**Tashkeel** (تشكيل) — Arabic for "forming/shaping"; the diacritic marks, and the
act of adding them.

**Harakat** (حركات) — "movements"; another name for the short-vowel diacritics
(fatha, damma, kasra).

**Fatha** ` َ ` — diacritic for short *a*, written above the letter.

**Damma** ` ُ ` — diacritic for short *u*, written above the letter.

**Kasra** ` ِ ` — diacritic for short *i*, written below the letter.

**Sukun** ` ْ ` — diacritic meaning "no vowel here"; marks a consonant that closes
a syllable.

**Shadda** ` ّ ` — diacritic meaning "this consonant is doubled" (held twice as
long). Almost always combined with a vowel mark.

**Tanween / nunation** — the doubled vowel marks (*fathatan* ` ً `, *dammatan*
` ٌ `, *kasratan* ` ٍ `) that add a final *-n* sound and signal an indefinite noun.

**I'rab** (إعراب) — Arabic case inflection: a word's grammatical role (subject /
object / possessor) shown mainly by the vowel on its **final** letter. The reason
diacritization needs whole-sentence syntax.

**Rasm** — the bare consonantal skeleton of a word, with no dots or vowels. Loosely,
the "shape" the model receives as input after stripping.

**MSA (Modern Standard Arabic)** — the formal, pan-Arab written register (news,
books, official speech), as opposed to regional spoken dialects. This model targets
MSA / classical Arabic.

**NFC / NFD normalization** — Unicode canonical forms. **NFC** composes a letter and
its marks into one code point; **NFD** decomposes them (e.g. أ → ا + hamza). The rawi
models normalize input with NFD (so they can treat every combining mark as a target)
and emit NFC; `bilstm` works in NFC. Used so that two visually identical strings
compare as equal.

## Machine learning

**Token / tokenizer** — a token is one unit fed to the model; the tokenizer turns
text into tokens. Here tokens are **characters**, one per Arabic base letter.

**Character-level** — a model whose tokens are individual characters (rather than
words or subword pieces). Chosen so each base letter maps to exactly one diacritic
decision.

**Embedding** — a learned vector of numbers that represents a token, placing
similar tokens near each other in a continuous space.

**LSTM (Long Short-Term Memory)** — a recurrent network that reads a sequence step
by step while maintaining a memory, letting it model context.

**BiLSTM (bidirectional LSTM)** — two LSTMs reading the sequence in opposite
directions; their outputs are concatenated so each position sees both past and
future context.

**Attention (Bahdanau / additive)** — a mechanism that lets each position build a
weighted blend of *all* positions, giving every letter direct access to the whole
sentence. "Bahdanau" / "additive" refers to the specific scoring formula
(`v·tanh(W·query + U·key)`).

**Logits** — the raw, unnormalized scores a model outputs before picking an answer.
One logit per diacritic class per letter.

**Argmax** — "the index of the maximum value." Applied to a letter's logits, it
selects the highest-scoring diacritic class.

**Class** — one of the diacritic categories a model can predict (0 = no mark). The
count depends on the model: **15** for `bilstm`/`libtashkeel` (the standard tashkeel
marks), **73/75** for the rawi family (an NFD scheme that also restores hamza and the
dagger alef).

**ONNX (Open Neural Network Exchange)** — a portable file format for neural
networks. Lets the model run via `onnxruntime` in many languages without PyTorch.

**onnxruntime (ORT)** — the runtime that executes ONNX models efficiently on CPU
(or GPU, with the right provider).

**Quantization (dynamic INT8)** — storing weights as 8-bit integers instead of
32-bit floats to shrink the model (~4×). Whether it costs accuracy is
architecture-dependent: **lossless** for the attention-free rawi models (so the
default ships as INT8), but lossy for `bilstm`'s attention (see
[§4.4](04-inference-pipeline.md#int8)).

**Padding / `<PAD>`** — filler tokens used to make sequences equal length for
batching. Harmful here because the encoders are unmasked (the bidirectional LSTM,
and `bilstm`'s attention) — see [batching](04-inference-pipeline.md#batching).

**DER (Diacritic Error Rate)** — fraction of base characters given the wrong mark.
Lower is better.

**WER (Word Error Rate)** — fraction of words with at least one wrong mark. Lower
is better; stricter than DER.

**Inference** — running a trained model to get predictions (as opposed to
*training*, which produces the model). This library only does inference.

**Next:** [Credits & license →](07-credits-and-license.md)
