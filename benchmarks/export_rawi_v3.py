"""Export rawi V3 (two-head gated model) to ONNX.

V3 shares a BiLSTM encoder between a *presence head* (1 logit: does this letter
carry a mark? — the WHERE gate) and a *value head* (75 logits: which mark? — the
WHICH). It internalizes the gating idea from docs/09 into a single network. The
checkpoint is a bare state_dict; the vocab is identical to V2 (same corpus, same
deterministic build), so we reuse rawi_v2.vocab.json after verifying the shapes.
"""
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CKPT = Path("/home/miro/Transferências/diacritization_model_lstm_3.pth")
MODELS = Path(__file__).resolve().parent.parent / "text2tashkeel" / "models"
VOCAB = MODELS / "rawi_v2.vocab.json"   # identical 236/75 vocab


class TwoStageDiacritizationModel(nn.Module):
    def __init__(self, vocab_size, diac_size, embedding_dim, hidden_dim,
                 num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.presence_head = nn.Linear(hidden_dim * 2, 1)
        self.value_head = nn.Linear(hidden_dim * 2, diac_size)

    def forward(self, x):
        h, _ = self.lstm(self.embedding(x))
        h = self.dropout(h)
        presence = self.presence_head(h).squeeze(-1)   # (B, T)
        value = self.value_head(h)                     # (B, T, diac_size)
        return presence, value


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    vocab_size, emb = sd["embedding.weight"].shape          # (236, 128)
    diac = sd["value_head.weight"].shape[0]                 # 75
    hidden = sd["lstm.weight_hh_l0"].shape[1]               # 256
    layers = 1 + max(int(k.split("_l")[1].split("_")[0])
                     for k in sd if k.startswith("lstm.weight_hh_l"))
    assert sd["presence_head.weight"].shape == (1, hidden * 2)
    print(f"checkpoint: vocab={vocab_size} diac={diac} emb={emb} "
          f"hidden={hidden} layers={layers}")

    v = json.loads(VOCAB.read_text(encoding="utf-8"))
    assert len(v["char_to_idx"]) == vocab_size, "vocab/char mismatch"
    assert len(v["diac_to_idx"]) == diac, "vocab/diac mismatch"
    print(f"reusing V2 vocab: chars={len(v['char_to_idx'])} diac={len(v['diac_to_idx'])} ✓")

    model = TwoStageDiacritizationModel(vocab_size, diac, emb, hidden, layers, 0.3)
    model.load_state_dict(sd)
    model.eval()

    onnx_path = MODELS / "rawi_v3.onnx"
    dummy = torch.randint(2, vocab_size, (1, 32), dtype=torch.long)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["presence", "value"],
        dynamic_axes={"input": {0: "batch", 1: "seq"},
                      "presence": {0: "batch", 1: "seq"},
                      "value": {0: "batch", 1: "seq"}},
        opset_version=17, dynamo=False,
    )
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        rp, rv = (t.numpy() for t in model(dummy))
    gp, gv = sess.run(["presence", "value"], {"input": dummy.numpy()})
    dp, dv = float(np.abs(rp - gp).max()), float(np.abs(rv - gv).max())
    print(f"torch↔onnx parity: presence {dp:.2e}, value {dv:.2e} "
          f"({'OK' if max(dp, dv) < 1e-3 else 'FAIL'})")

    # int8
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q = MODELS / "rawi_v3.int8.onnx"
    quantize_dynamic(str(onnx_path), str(q), weight_type=QuantType.QInt8)
    print(f"int8: {q.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
