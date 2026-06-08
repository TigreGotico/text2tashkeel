# 12. Alternative architectures — how rawi compares

`text2tashkeel`'s models are deliberately plain: an embedding, a stacked BiLSTM, a
per-letter classifier ([§3](03-architecture.md)). Arabic diacritization has been
attacked with a whole range of architectures — rule engines, attention RNNs,
character transformers. This page places rawi against that wider field **under a
deployment budget**, because for a voice pipeline the question is not "what is the
most accurate model" but "what is the most accurate model that *fits*."

> **The embeddable tier.** A diacritizer is **embeddable** if it is **≤ 25 MB on
> disk** *and* runs in **≤ 20 ms/sentence** single-thread. That is the budget a
> real-time or on-device (or in-browser) voice pipeline can spend on diacritization
> *before* the TTS step without the delay being heard. Models outside the tier are
> accurate-but-impractical for that use; the question this page asks is **which model
> is most accurate _within_ the tier.**

> Read [§8.2](08-models-and-benchmarks.md#82--the-contamination-caveat--read-this-before-the-numbers)
> first: absolute DER is optimistic (the corpus overlaps training data). The robust
> result here is the **relative** picture — the ranking within the tier.

## 12.1 The contenders, by architecture

| Model | Architecture | Params | Classes |
|-------|--------------|-------:|--------:|
| **rawi-v2** | embedding → 2× BiLSTM-256 → linear (no attention) | 2.4 M | 75 |
| **rawi-ensemble** | rawi-v2 gates *where*, rawi-v3's value head supplies *which* | 4.8 M | 75 |
| `bilstm` | BiLSTM **+ Bahdanau attention** | 5.2 M | 15 |
| Shakkala v3 | embedding → **3× BiLSTM** + BatchNorm, **fixed 315-length** | 2.5 M | 28 |
| Shakkelha (RNN) | embedding → 2× BiLSTM-256 (variable length) | 2.7 M | 19 |
| CATT | **character transformer** (encoder, attention) | ~12 M | — |
| Mishkal | **rule-based** morphology + dictionary (no network) | — | — |

rawi-v2 and Shakkelha's RNN are almost the *same network* — two BiLSTM-256 layers
over a character embedding. Keep that in mind for §12.4.

## 12.2 The constrained leaderboard

Scored with the **same DER metric** on the **held-out test split**. Bundled models
(`rawi`, `bilstm`, `libtashkeel`) use the full 817k-sentence run from
[§10](10-benchmark-report.md); Shakkelha and Shakkala were run over the 813,993
sentences under 314 characters (Shakkala's fixed-length limit); Mishkal, rule-based, is
scored on a 30k sample over the ~52 % of sentences where it preserves the consonant
skeleton. Latency is single-thread ms/sentence on ~58-char input.

**In-tier rows are sorted by accuracy.** "In tier?" applies the ≤ 25 MB **and**
≤ 20 ms rule.

| Model | DER ↓ | DER\* ↓ | WER ↓ | latency ↓ | size | in tier? |
|-------|------:|-------:|------:|----------:|-----:|:--------:|
| **`rawi-ensemble`** 🏆 | **2.04%** | 2.94% | 7.51% | ~2 ms | 4.9 MB | ✅ |
| **`rawi-v2-int8`** ⭐ | 2.30% | 3.39% | 8.36% | **~1 ms** | **2.5 MB** | ✅ |
| CATT | 4.74%² | 2.99% | 28.45%⁴ | ~14–20 ms | 21.6 MB | ✅ (edge) |
| Shakkelha (RNN, big/avg20) | 4.75% | 4.51% | 17.07% | ~1 ms¹ | ~2.5 MB¹ | ✅ |
| `bilstm` | 4.95% | 5.08% | 18.02% | 6.5 ms | 17.9 MB | ✅ |
| `libtashkeel` | 6.89% | 7.80% | 24.56% | 3.3 ms | 4.8 MB | ✅ |
| Mishkal | 19.08%³ | 26.00% | 58.86% | 22 ms | — | ✅ (rule) |
| Shakkala v3 | 5.90% | 6.51% | 21.22% | **254 ms** | 10.2 MB | ✗ (latency) |

¹ Shakkelha's RNN is architecturally rawi-v2's twin (2× BiLSTM-256); its
accuracy is measured from the reference model, and an int8 ONNX of that network
sizes/times like rawi-v2. ² CATT on this broad test; it scores far lower on its own
narrow benchmark — distribution dominates ([§8.2](08-models-and-benchmarks.md)).
³ Mishkal rewrites the consonant skeleton on ~half of sentences (so those can't be
char-aligned for scoring); the figure is the alignable subset, and the raw,
unaligned number is far worse. ⁴ CATT normalizes its output (drops the dagger-alef and
non-Arabic), so its **WER is inflated** under a strict word metric; all rows are scored
on Arabic letters only, where its DER\*/DER are competitive.

**The result:** **everything except Shakkala fits the embeddable tier** — so the tier
isn't a trick that excludes the competition. Within it, rawi is **~2× more accurate
than the next-best model** (`rawi-ensemble` 2.04% vs CATT 4.74%), and `rawi-v2-int8`
delivers near-that accuracy at **~1 ms / 2.5 MB**. The only thing close to rawi on
accuracy (CATT) sits at the tier's size/latency *edge* and slips further on broad
data ([§8.2](08-models-and-benchmarks.md)). Shakkala is excluded purely on latency
(254 ms — see §12.3). So rawi is not just "fast and small"; it is **the most accurate
diacritizer that an on-device voice pipeline can actually run.**

## 12.3 It's the inference *shape*, not the parameter count

Shakkala v3 and rawi-v2 have **almost identical parameter counts** (2.5 M vs 2.4 M),
yet Shakkala is **~250× slower** (254 ms vs ~1 ms). Latency is set by the **shape of
the computation**, not the weight count:

| | Shakkala v3 | rawi-v2 |
|---|---|---|
| timesteps per call | **fixed 315** (pads every sentence) | variable (~58 avg) |
| recurrent layers | 3 stacked BiLSTM + BatchNorm | 2 BiLSTM |
| precision | fp32 | **int8** (lossless — no attention) |

The dominant cost is the **fixed 315-length padding**: a 58-character sentence still
runs the full 315 timesteps, and LSTMs are sequential (no parallelism across time),
so that waste multiplies by the layer count. rawi runs only the actual length, in
two layers, in int8. The "boring" choices — variable length, fewer layers, no
attention so int8 is lossless ([§11](11-what-makes-rawi-different.md)) — are exactly
what buy the millisecond.

## 12.4 The architectural twin: data and task, not the network

Shakkelha's RNN is the most informative comparison because it is **the same network
as rawi-v2** — two BiLSTM-256 layers over a character embedding — trained by other
authors on other data. On the same test it scores **4.75% DER; rawi-v2 scores
2.30%**. Same architecture, ~2× the accuracy.

That isolates *where rawi's quality comes from*: not a cleverer network, but the
**corpus and the task framing** — the NFD label scheme that also restores hamza and
the dagger alef, and the where/which factorization of the ensemble
([§11](11-what-makes-rawi-different.md), [§9](09-combining-models.md)). The network
is intentionally ordinary; the data and the decomposition are the work.

## 12.5 Takeaway

Bigger or fancier architectures (attention RNNs, transformers) can win on absolute
accuracy when latency and size are free. But the **embeddable tier** (≤ 25 MB,
≤ 20 ms) is what real-time and on-device voice pipelines actually have to work with,
and *most* of the field fits it — so the tier is a fair bar, not a loophole. Within
it, a plain BiLSTM trained on the right corpus with the right label scheme is the
most accurate option measured here, by roughly 2×. That is the niche
`text2tashkeel` targets: the **most accurate embeddable diacritizer**, not the best
diacritizer at any cost.

## 12.6 Out of tier, for completeness

**Autoregressive** systems decode left-to-right with search — accurate, but far slower
than a single-pass tagger, so they fall outside the embeddable tier and aren't
benchmarked above. CATT's encoder-*decoder* variant is one; AppTek's **2SDiac**
([Bahar et al., *Take the Hint*, Interspeech 2023](https://arxiv.org/abs/2306.03557))
is another — it adds *hint-based* diacritization, reading partially-diacritized input
to improve its output, an idea close to the where/which split in
[§9](09-combining-models.md). These set the accuracy ceiling when latency and size are
unconstrained; the tier question is what survives the ~20 ms / 25 MB budget, which they
do not.

---

*External models referenced for comparison, with credit to their authors:*
**Shakkala** (Ahmad Barqawi, MIT), **Shakkelha** (Ali Hamdi Ali Fadel et al., MIT),
**CATT** (character-transformer diacritizer), **Mishkal** (Taha Zerrouki, GPLv3 —
referenced for comparison only, not bundled). `bilstm` (Zain Mahmood, MIT) and
`libtashkeel` (Musharraf Omer, MIT) are bundled — see
[Credits & license](07-credits-and-license.md).
