"""Produce the INT8 models from the fp32 ones via dynamic quantization.

Dynamic quantization stores the weights as 8-bit integers and dequantizes them
on the fly at inference — no calibration data needed.

Quant sensitivity is architecture-dependent (see ../docs/08):
  * `rawi` (plain BiLSTM, no attention)  → **essentially lossless** (18.34→18.33 DER)
  * `bilstm` (BiLSTM + attention)        → **lossy** (4.95→12.89 DER) — attention
    matmuls are quant-sensitive

    python benchmarks/quantize.py
"""

from __future__ import annotations

import os

from onnxruntime.quantization import QuantType, quantize_dynamic

MODELS = [
    ("text2tashkeel/models/bilstm.onnx", "text2tashkeel/models/bilstm.int8.onnx"),
    ("text2tashkeel/models/rawi.onnx", "text2tashkeel/models/rawi.int8.onnx"),
]


def main() -> None:
    for src, dst in MODELS:
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
        print(f"{os.path.basename(src)}: {os.path.getsize(src) / 1e6:.1f} MB"
              f"  ->  {os.path.basename(dst)}: {os.path.getsize(dst) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
