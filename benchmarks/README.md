# Benchmarks

Accuracy measurements and the model-combination research, on the
[`TigreGotico/arabic_diacritized_text`](https://huggingface.co/datasets/TigreGotico/arabic_diacritized_text)
corpus (train/test/val splits).

> ⚠️ **Contamination caveat.** This corpus aggregates many public sources that the
> models were likely trained on, so absolute numbers are optimistic and small
> gaps are not a clean ranking. See
> [`../docs/08-models-and-benchmarks.md`](../docs/08-models-and-benchmarks.md#82--the-contamination-caveat--read-this-before-the-numbers).

## Two metrics

- **DER (all)** — wrong-mark fraction over *all* base characters (strict; the
  standard definition).
- **DER\* (marked)** — over only the characters marked in the gold (mirrors rawi's
  training metric). The gap between the two fingerprints over- vs under-marking —
  the core of the [§9](../docs/09-combining-models.md) research.

(WER — fraction of words with any wrong mark — is also reported.)

## Scripts

| File | Purpose |
|------|---------|
| `benchmark.py` | Stream a split, score all models (DER / DER\* / WER). `--limit N`, or `-1` for full. |
| `parallel_benchmark.py` | Full-file scoring parallelized across cores (for the 820k-line splits). Pin BLAS threads — see its header. |
| `quantize.py` | Rebuild `bilstm.int8.onnx` from `bilstm.onnx`. |
| `rawi_confidence_filter.py` | §9.3 — confidence-threshold filter on rawi (attempt A). |
| `rawi_agreement_gating.py` | §9.4 — agreement gating, bilstm gate (attempt B). |
| `rawi_gating_sweep.py` | §9.4 — gate (bilstm / libtashkeel / AND / OR) × threshold grid. |
| `results_full_test.txt` | Saved single-model run on the full 817k-line test split. |
| `results_ensemble.txt` | Saved ensemble run on the full test split. |

## Results — full test split (817,035 sentences, 49.4M chars)

| Model | DER (all) ↓ | DER\* (marked) ↓ | WER ↓ |
|-------|-------------|------------------|-------|
| **`ensemble`** (gated) | **3.99%** | 3.13% | **15.79%** |
| `libtashkeel+rawi` (gated) | 4.12% | 3.59% | 16.22% |
| `bilstm+rawi` (gated) | 4.38% | 4.17% | 16.72% |
| bilstm | 4.95% | 5.08% | 18.02% |
| libtashkeel | 6.89% | 7.80% | 24.56% |
| bilstm-int8 | 12.89% | 18.42% | 37.54% |
| rawi | 18.49% | **3.07%** | 55.48% |

(External SOTA reference: [CATT](https://github.com/abjadai/catt)-EO, exported to
ONNX, scored ~4.06% DER on a 2k sample with a stricter per-Arabic-letter metric —
beating our ensemble. See `docs/08`.)

The gated **`ensemble`** (rawi's marks, gated by where bilstm *or* libtashkeel
mark) wins on every metric — see [`../docs/09-combining-models.md`](../docs/09-combining-models.md)
for the full story, including *why* rawi over-marks (a training bug) and what to
do about it.

## Reproduce

```bash
pip install -e ".[bench]"
python benchmarks/quantize.py                       # rebuild the int8 model
python benchmarks/benchmark.py --limit 15000        # quick sample, all models
python benchmarks/benchmark.py --limit -1 --splits test   # full split (slow on one core)
# many cores (set BLAS threads to 1 — see script header):
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python benchmarks/parallel_benchmark.py --file test.txt --workers 12
```
