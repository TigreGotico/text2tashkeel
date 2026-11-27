# Run rawi-ensemble in loom

<!-- agent-hint: tools/loom holds the exporter, the host harness and the parity check. tests/test_export_rawi_loom.py skips when loom-exporter is not installed. The one-graph BiLSTM binding is loom-ai-org/loom.cpp#144; loom-ai-org/loom-exporter#98 makes the exporter emit it. -->

[loom.cpp](https://github.com/loom-ai-org/loom.cpp) runs a model from one GGUF file that holds the graphs, the weights and a Lua driver. `tools/loom/` exports `rawi-ensemble` (rawi-v2 decides where a mark goes, the rawi-v3 value head decides which mark) to that format, builds a small host program, and checks the result against the ONNX models.

## What the GGUF holds

- 12 graphs. For each of rawi-v2 and rawi-v3: the embedding, two BiLSTM layers with a forward and a backward cell each, and the head (`v2_head`, `v3_value`).
- A driver. `infer({tokens = ids})` returns `gated_logits`: T × 75 numbers, row-major. Where rawi-v2 predicts no mark, class 0 gets the highest value. The argmax per character is the `rawi-ensemble` class.
- The driver runs each BiLSTM layer with `loom.run_bi_lstm_and_retain`, which computes the whole layer as one graph. An engine without this binding stops at the first call.

The input is the rawi tokenization: NFD, then remove symbols (category So) and combining marks (category Mn), then map each character with `char_to_idx` from `TigreGotico/rawi-v2` `rawi_v2.vocab.json`. The host program keeps text processing out of loom: it reads ids and writes logits.

## Export

1. Install the extra and loom-exporter:

   ```bash
   pip install --pre "text2tashkeel[loom]"
   pip install -e <path to a loom-exporter checkout>
   ```

2. Export. The script downloads the two PyTorch checkpoints from the Hub.

   ```bash
   python tools/loom/export_rawi_loom.py rawi_ensemble.f32.gguf --quantize F32
   ```

   `--quantize` takes `F32`, `F16` or `Q8_0_PC`. The script prints how many BiLSTM calls it changed to `loom.run_bi_lstm_and_retain`. The count is 0 when loom-exporter already emits that call, and 4 when it does not.

## Check parity and speed

1. Build the host program against a loom.cpp checkout that has `loom.run_bi_lstm_and_retain`:

   ```bash
   cmake -B build -S tools/loom/host -DLOOM_CPP=<path to loom.cpp> -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ```

2. Compare with the fp32 ONNX models on the test split of `TigreGotico/arabic_diacritized_text`:

   ```bash
   python tools/loom/parity.py rawi_ensemble.f32.gguf --host build/rawi_run --limit 1000
   ```

   The script prints the number of characters whose class differs and the time per sentence. It exits with 1 when a character differs.

## Results

1,000 sentences of the test split, 60,393 characters, CPU, one laptop:

| GGUF weights | characters that differ from fp32 ONNX |
| :--- | ---: |
| F32 | 0 |
| F16 | 1 |

The F16 difference comes from the F16 weights: the per-step binding gives 2 differences on the same F16 file.

Median time per sentence on the 934 sentences of arbtok's gold set, one binary, each driver run twice in turn:

| | per-step binding | `run_bi_lstm_and_retain` | onnxruntime, 1 thread |
| :--- | ---: | ---: | ---: |
| F32 / fp32 | 17.6 ms | 10.7 ms | 5.0 ms |
| F16 | 14.3 ms | 11.1 ms | |
| int8 | | | 1.4 ms |

loom used its default thread count. Other jobs ran on the laptop, so compare the columns, not the absolute times.

---

[← Install, models and CLI](12-install-models-cli.md) · [Home](index.md)
