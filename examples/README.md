# Examples

Runnable, commented scripts. They print intermediate steps so you can *see* what
the model does — no Arabic knowledge required (output is romanized where helpful).

Run any of them after `pip install -e .` from the repo root:

```bash
python examples/01_quickstart.py
```

| Script | Shows |
|--------|-------|
| [`01_quickstart.py`](01_quickstart.py) | The three-line basic usage. |
| [`02_choose_a_model.py`](02_choose_a_model.py) | The bundled models side by side on the same sentence. |
| [`03_understand_output.py`](03_understand_output.py) | Strip → predict → per-letter class → re-apply, step by step. |
| [`04_transliteration.py`](04_transliteration.py) | Romanize the result so non-Arabic readers can hear the vowels the model added. |
| [`05_batch_safely.py`](05_batch_safely.py) | Process many sentences correctly (and why padded batching is wrong). |
| [`06_raw_onnx.py`](06_raw_onnx.py) | Drive the ONNX session directly — input IDs in, logits out. |
| [`07_file_pipeline.py`](07_file_pipeline.py) | Diacritize a text file line by line. |
| [`sample.txt`](sample.txt) | A few undiacritized lines for `07`. |

New here? Read [`docs/index.md`](../docs/index.md) first — it explains the
writing system and the model from scratch.
