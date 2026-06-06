# Training notebooks — the `rawi` line

The `rawi` model's training notebooks, kept here for the record and for future
work. They are **authored, not executed** (training runs on the
`TigreGotico/arabic_diacritized_text` corpus on a GPU — see each notebook's config).

| Notebook | What it is |
|----------|------------|
| [`rawi_v1_train.ipynb`](rawi_v1_train.ipynb) | **The released V1, unchanged** — with markdown notes flagging the training bug. |
| [`rawi_v2_train.ipynb`](rawi_v2_train.ipynb) | **Bug fixed.** Padding uses `-100` (not `0`), so the loss stops ignoring real "no-mark" positions and the model learns to abstain. |
| [`rawi_v3_train.ipynb`](rawi_v3_train.ipynb) | **Two-stage gated model.** A shared encoder with a *presence* head (where) + a *value* head (which) — the gating that powers the ensemble, internalised into one model. |

## The bug (V1) in one line

In V1, the **"no-diacritic" class and the padding fill value share index `0`**, and
the loss is `CrossEntropyLoss(ignore_index=0)`. So padding *and every genuinely-bare
letter* are excluded from the loss — the model is **never trained to output
"nothing"**, and it over-marks. Result: great marked-position accuracy
(DER\* ≈ 3%), poor overall DER (≈ 18%). Full analysis in
[`../docs/09-combining-models.md`](../docs/09-combining-models.md) §9.5.

## V2 — the fix

Three one-line changes (pad targets with `-100`; `CrossEntropyLoss()` default
`ignore_index=-100`; metric counts all real positions). Class 0 now gets gradient,
so the model learns when **not** to mark. Expected to collapse the DER(all) gap in
a single model — cheaper than the inference-time ensemble.

## V3 — two-stage / two-head

Factorizes the task the way the ensemble does, but trained end to end:

- **presence head** — binary, trained on *all* real positions → decides **where**.
- **value head** — trained on *marked positions only* → decides **which** (rawi's
  existing strength).

At inference the presence head gates and the value head fills the approved
positions. This is the trained analogue of `text2tashkeel`'s gate(where)+value(which)
ensemble (docs §9), aiming for ensemble-level accuracy in one model.

> When a `rawi-v2` / `rawi-v3` checkpoint is trained, export it to ONNX and drop it
> into `text2tashkeel`'s model registry — the library is built to absorb new models.
