"""Check a rawi-ensemble loom GGUF against the fp32 ONNX members on the benchmark test split.

    python tools/loom/parity.py rawi_ensemble.f16.gguf --host build/rawi_run --limit 1000

The reference class per character is the rawi-ensemble rule on the fp32 ONNX files: 0 where
rawi-v2 predicts no mark, else the argmax of rawi-v3's value head. The loom class is the argmax of the
GGUF's `gated_logits`. Prints the characters that differ and the time per sentence; exits 1 on a
difference.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.benchmark import split_lines

N_CLASSES = 75


def bare_ids(text: str, c2i: dict, unk: int) -> list:
    nfd = (c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) not in ("So", "Mn"))
    return [c2i.get(c, unk) for c in nfd]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("--host", required=True, help="rawi_run, built from tools/loom/host")
    ap.add_argument("--limit", type=int, default=1000, help="sentences from the test split")
    ap.add_argument("--max-len", type=int, default=400, help="skip longer sentences (characters)")
    args = ap.parse_args()

    vocab = json.loads(Path(hf_hub_download("TigreGotico/rawi-v2", "rawi_v2.vocab.json")).read_text("utf-8"))
    c2i = dict(vocab["char_to_idx"])
    unk = c2i.get("<UNK>", 1)
    sents = []
    for line in split_lines("test"):
        ids = bare_ids(line, c2i, unk)
        if 0 < len(ids) <= args.max_len:
            sents.append(ids)
        if len(sents) == args.limit:
            break

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    v2 = ort.InferenceSession(hf_hub_download("TigreGotico/rawi-v2", "rawi_v2.onnx"), opts)
    v3 = ort.InferenceSession(hf_hub_download("TigreGotico/rawi-v3", "rawi_v3.onnx"), opts)
    ref = []
    for ids in sents:
        x = np.array([ids], dtype=np.int64)
        gate = v2.run(["output"], {"input": x})[0][0].argmax(-1)
        value = v3.run(["value"], {"input": x})[0][0].argmax(-1)
        ref.append(np.where(gate == 0, 0, value))
    ref = np.concatenate(ref)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ids.txt").write_text("\n".join(" ".join(map(str, s)) for s in sents) + "\n")
        subprocess.run([args.host, args.gguf, tmp / "ids.txt", tmp / "logits.f32", tmp / "times.txt"], check=True)
        got = np.fromfile(tmp / "logits.f32", dtype=np.float32).reshape(-1, N_CLASSES).argmax(-1)
        times_ms = np.loadtxt(tmp / "times.txt") / 1e3

    diff = int((got != ref).sum())
    print(f"sentences {len(sents)}  characters {len(ref)}  differ {diff}")
    print(f"median {np.median(times_ms):.1f} ms  p90 {np.percentile(times_ms, 90):.1f} ms per sentence")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
