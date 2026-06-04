"""Produce the INT8 model from the fp32 model via dynamic quantization.

Dynamic quantization stores the weights as 8-bit integers and dequantizes them
on the fly at inference. It needs no calibration data, so it's a one-liner — but
LSTMs lose accuracy under it (see ../docs/04-inference-pipeline.md#int8 and the
measured cost in results.txt).

    python benchmarks/quantize.py
"""

from __future__ import annotations

import os

from onnxruntime.quantization import QuantType, quantize_dynamic

SRC = "text2tashkeel/models/bilstm.onnx"
DST = "text2tashkeel/models/bilstm.int8.onnx"


def main() -> None:
    quantize_dynamic(SRC, DST, weight_type=QuantType.QInt8)
    print(f"{SRC}: {os.path.getsize(SRC) / 1e6:.1f} MB")
    print(f"{DST}: {os.path.getsize(DST) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
