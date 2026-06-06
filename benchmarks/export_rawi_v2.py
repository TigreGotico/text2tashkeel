"""Reconstruct the rawi V2 vocab from the corpus and export the checkpoint to ONNX.

The V2 training notebook only emitted the .pth state_dict (the in-notebook ONNX
export failed). The vocab is not shipped, but it is *deterministic*: it is the
sorted set of base chars / diacritic-combos seen across train+val+test of
TigreGotico/arabic_diacritized_text, with the exact normalization the notebook
used. We rebuild it here, verify the sizes against the checkpoint shapes
(236 chars, 75 diacritic classes), then export a dynamic-length ONNX graph.
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CKPT = Path("/home/miro/Transferências/text2tashkeel/diacritization_model_lstm_2.pth")
SNAP = Path.home() / (
    ".cache/huggingface/hub/datasets--TigreGotico--arabic_diacritized_text/"
    "snapshots/b9ac17960a5d01bd110267a0b736ab844ad30f71"
)
OUT_DIR = Path(__file__).resolve().parent.parent / "text2tashkeel" / "models"


# ── exact notebook normalization ────────────────────────────────────────────
def normalize_text(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "So"
    )


def extract_diacritics(text: str):
    text = normalize_text(text)
    result, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if unicodedata.category(ch) == "Mn":
            i += 1
            continue
        j = i + 1
        diac = []
        while j < n and unicodedata.category(text[j]) == "Mn":
            diac.append(text[j])
            j += 1
        result.append((ch, "".join(diac)))
        i = j
    return result


def build_vocab(files):
    chars, diacs = set(), set()
    for fp in files:
        print(f"  scanning {fp.name} ...", flush=True)
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for ch, d in extract_diacritics(line):
                    chars.add(ch)
                    diacs.add(d)
    char_to_idx = {"<PAD>": 0, "<UNK>": 1}
    for c in sorted(chars):
        char_to_idx[c] = len(char_to_idx)
    diac_to_idx = {"": 0}
    for d in sorted(diacs):
        if d:
            diac_to_idx[d] = len(diac_to_idx)
    return char_to_idx, diac_to_idx


# ── model (verbatim from the notebook) ──────────────────────────────────────
class LSTMDiacritizationModel(nn.Module):
    def __init__(self, vocab_size, diac_size, embedding_dim, hidden_dim,
                 num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, diac_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    vocab_size, emb_dim = sd["embedding.weight"].shape       # (236, 128)
    diac_size = sd["fc.weight"].shape[0]                      # 75
    hidden_dim = sd["lstm.weight_hh_l0"].shape[1]             # 256
    num_layers = 1 + max(
        int(k.split("_l")[1].split("_")[0])
        for k in sd if k.startswith("lstm.weight_hh_l")
    )
    print(f"checkpoint: vocab={vocab_size} diac={diac_size} "
          f"emb={emb_dim} hidden={hidden_dim} layers={num_layers}")

    files = [SNAP / "train.txt", SNAP / "val.txt", SNAP / "test.txt"]
    print("reconstructing vocab from train+val+test ...")
    c2i, d2i = build_vocab(files)
    print(f"reconstructed: chars={len(c2i)} diac={len(d2i)}")
    assert len(c2i) == vocab_size, f"char vocab {len(c2i)} != {vocab_size}"
    assert len(d2i) == diac_size, f"diac vocab {len(d2i)} != {diac_size}"
    print("  ✓ sizes match checkpoint exactly")

    model = LSTMDiacritizationModel(vocab_size, diac_size, emb_dim, hidden_dim,
                                    num_layers, dropout=0.3)
    model.load_state_dict(sd)
    model.eval()

    OUT_DIR.mkdir(exist_ok=True)
    onnx_path = OUT_DIR / "rawi_v2.onnx"
    dummy = torch.randint(2, vocab_size, (1, 32), dtype=torch.long)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 1: "seq"},
                      "output": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    vocab_path = OUT_DIR / "rawi_v2.vocab.json"
    vocab_path.write_text(
        json.dumps({"char_to_idx": c2i, "diac_to_idx": d2i}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    print(f"wrote {vocab_path}")

    # parity check: torch vs onnxruntime on the dummy
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(["output"], {"input": dummy.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"torch↔onnx max abs diff: {diff:.2e}  ({'OK' if diff < 1e-3 else 'FAIL'})")


if __name__ == "__main__":
    sys.exit(main())
