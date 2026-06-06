# 7. Credits & license

`text2tashkeel` bundles models from three projects. The **rawi** family is original
research by this project; **bilstm** and **libtashkeel** are independent third-party
models, re-exported here with credit. Please cite the relevant authors below.

## Original to this project (TigreGotico)

- **The rawi models** — `rawi` (V1), `rawi-v2`, and the two-head `rawi-v3` — by
  **TigreGotico**: an NFD-based BiLSTM diacritizer that restores hamza and the dagger
  alef alongside the standard marks, trained on a corpus aggregated for this work.
  (Training compute for V2/V3 was provided by **Mike Hansen**.)
- **The gated-ensemble method and the flagship** — using one model to decide *where*
  a mark goes and another to decide *which*, and the `rawi-ensemble` that fuses
  rawi-v2 (where) and rawi-v3 (which) into a **single ONNX graph**
  ([§9](09-combining-models.md)).
- **The analysis** — the diagnosis of rawi V1's training behaviour and its fix
  ([§9.3](09-combining-models.md#93-why-rawi-v1-over-marks)), and the cross-corpus
  contamination study against an external transformer baseline
  ([§10.5](10-benchmark-report.md#105-external-comparison--catt-and-why-cross-corpus-der-is-slippery)).
- **The toolkit** — the unified, dependency-light (numpy + onnxruntime) multi-model
  API, the INT8 quantizations, the documentation, and the benchmark harness.

## The models

### `bilstm`

- **Author:** Zain Mahmood
- **Source:** [`Z-Mahmood/arabic-diacritizer-public-release`](https://github.com/Z-Mahmood/arabic-diacritizer-public-release)
- **License:** MIT
- **What's original:** the BiLSTM + Bahdanau-attention architecture, the 15-class
  label scheme, the character tokenizer, the trained weights, and the
  DER/WER metric definitions reused in our benchmark.
- **What we did:** exported the released PyTorch checkpoint to ONNX (`bilstm.onnx`)
  and produced an INT8 quantization (`bilstm.int8.onnx`), mirrored under MIT at
  [`TigreGotico/bilstm-diacritizer`](https://huggingface.co/TigreGotico/bilstm-diacritizer).
  We do **not** ship the upstream sentence cache, so output may differ from the
  upstream demo.

### `rawi`

- **Author:** TigreGotico
- **Source:** [`TigreGotico/rawi`](https://huggingface.co/TigreGotico/rawi) (model card + training notebook)
- **Acknowledgement:** TigreGotico built the corpus, the model, and the training
  notebook; **Mike Hansen** ran the V2/V3 training on his GPUs. `rawi` (V1) is an
  earlier research checkpoint.
- **What's original:** a BiLSTM with an NFD-based diacritic scheme (73 classes in
  V1, 75 in V2; notably it restores hamzas and other marks), the trained weights,
  and the vocabulary.
- **What we did:** exported the checkpoints to ONNX and reproduced the notebook's exact
  NFD normalization and letter-only decode in pure Python.

### `libtashkeel`

- **Author:** Musharraf Omer
- **Source:** [`mush42/libtashkeel`](https://github.com/mush42/libtashkeel)
  (models trained via [`mush42/hareef`](https://github.com/mush42/hareef))
- **License:** MIT
- **What's original:** the character+hint encoder model, the 15-target scheme
  with a length input, the vocab/hint/target maps, and the Rust reference
  implementation.
- **What we did:** exported the project's model to ONNX (`libtashkeel.onnx`) with
  its three JSON maps, mirrored under MIT at
  [`TigreGotico/libtashkeel-diacritizer`](https://huggingface.co/TigreGotico/libtashkeel-diacritizer),
  and ported the pure-prediction path (no hints, no taskeen, no sentence
  segmentation) from the Rust `crates/core/src/lib.rs` to Python.

## The benchmark corpus

- [`TigreGotico/arabic_diacritized_text`](https://huggingface.co/datasets/TigreGotico/arabic_diacritized_text)
  — an aggregate of many public diacritized-Arabic sources, with train/test/val
  splits. **See [§8](08-models-and-benchmarks.md) for the important caveat that
  all models likely overlap this data.**

## This library's license

The wrapper code (everything under `text2tashkeel/` that isn't a bundled model
file, plus `docs/`, `examples/`, `tests/`, `benchmarks/`) is offered under the
**Apache-2.0** license. The bundled model files remain under their respective
upstream licenses and authors as listed above (the `bilstm` and `libtashkeel`
weights are MIT, both Apache-2.0-compatible) — see each source for details before
redistribution.

If you use this in research or a product, credit the **model authors above** —
TigreGotico for the rawi family, Zain Mahmood for `bilstm`, and Musharraf Omer for
`libtashkeel`.

**Next:** [Models & benchmarks →](08-models-and-benchmarks.md)
