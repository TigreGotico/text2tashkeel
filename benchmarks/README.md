# Benchmarks

Accuracy measurements and model-combination analysis, on the
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
| `rawi_confidence_filter.py` | §9.3 — confidence-threshold filter on rawi. |
| `rawi_agreement_gating.py` | §9.4 — agreement gating, bilstm gate. |
| `rawi_gating_sweep.py` | §9.4 — gate (bilstm / libtashkeel / AND / OR) × threshold grid. |
| `results_full_test.txt` | Saved single-model run on the full 817k-line test split. |
| `results_ensemble.txt` | Saved ensemble run on the full test split. |

## Results — full test split (817,035 sentences, 49.4M chars)

**The full 12-model table** (accuracy, latency, size) with Pareto plots is the
[**benchmark report**](../docs/10-benchmark-report.md). Headlines:

| Pick | Model | DER | latency | size |
|------|-------|----:|--------:|-----:|
| best accuracy | `bilstm+libtashkeel+rawi` | 3.99% | 21 ms | 32 MB |
| voice sweet spot | `libtashkeel+rawi-int8` | 4.13% | 7 ms | 7.3 MB |
| best single | `bilstm` | 4.95% | 6.5 ms | 18 MB |

`run_all_combos.py` writes `results_all_combos.txt`; `generate_plots.py` renders
the plots into `docs/images/`. The gating strategy and rawi's over-marking property
(high DER\*, best DER*) are covered in
[`../docs/09-combining-models.md`](../docs/09-combining-models.md).

(External SOTA reference: [CATT](https://github.com/abjadai/catt), exported to
ONNX — its EO variant scores ~4.1% DER, edging our best ensemble. See `docs/10` §10.5.)

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
