# 7. Credits & license

This library is **a repackaging, not original research.** All the model weights
and the architectures are other people's work. `text2tashkeel` contributes only:
ONNX exports/repackaging, a single dependency-light Python interface that unifies
them, the documentation you're reading, and the benchmark harness.

Please cite and credit the original authors below.

## The models

### `bilstm` (the default)

- **Author:** Zain Mahmood
- **Source:** [`Z-Mahmood/arabic-diacritizer-public-release`](https://github.com/Z-Mahmood/arabic-diacritizer-public-release)
- **License:** MIT
- **What's original:** the BiLSTM + Bahdanau-attention architecture, the 15-class
  label scheme, the character tokenizer, the trained weights, and the
  DER/WER metric definitions reused in our benchmark.
- **What we did:** exported the released PyTorch checkpoint to ONNX (`bilstm.onnx`)
  and produced an INT8 quantization (`bilstm.int8.onnx`). We do **not** ship the
  upstream sentence cache, so output may differ from the upstream demo.

### `rawi`

- **Author:** TigreGotico
- **Source:** [`TigreGotico/rawi`](https://huggingface.co/TigreGotico/rawi) (model card + training notebook)
- **What's original:** a BiLSTM with a 73-class diacritic scheme (notably,
  it restores hamzas and other marks via NFD-based labeling), the trained
  weights, and the vocabulary.
- **What we did:** bundled the released `diacritization_model_lstm.onnx` and its
  `vocab.json`, and reproduced the notebook's exact NFD normalization and
  letter-only decode in pure Python.

### `libtashkeel`

- **Author:** Musharraf Omer
- **Source:** [`mush42/libtashkeel`](https://github.com/mush42/libtashkeel)
  (models trained via [`mush42/hareef`](https://github.com/mush42/hareef))
- **License:** MIT
- **What's original:** the character+hint encoder model, the 15-target scheme
  with a length input, the vocab/hint/target maps, and the Rust reference
  implementation.
- **What we did:** bundled the project's `model.onnx` and its three JSON maps,
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
**MIT** license, matching the upstream projects. The bundled model files remain
under their respective upstream licenses and authors as listed above — see each
source for details before redistribution.

If you use this in research or a product, credit the **model authors above**, not
just this repackaging.

**Next:** [Models & benchmarks →](08-models-and-benchmarks.md)
