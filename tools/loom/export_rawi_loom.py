"""Export the rawi-ensemble (rawi-v2 gate + rawi-v3 value head) to one loom GGUF.

    python tools/loom/export_rawi_loom.py rawi_ensemble.f16.gguf --quantize F16

The PyTorch checkpoints come from TigreGotico/rawi-v2 and TigreGotico/rawi-v3. The GGUF holds 12 graphs
(per network: embedding, 2 BiLSTM layers x 2 directions, head) and a Lua driver whose
`infer({tokens = ids})` returns T x 75 `gated_logits`, row-major. argmax per character is the
rawi-ensemble class. Needs `pip install text2tashkeel[loom]` and loom-exporter.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from loom_exporter.multi_phase_export import BaseMultiPhaseModelExportConfig, ExportPhase, RecurrentPhase

HERE = Path(__file__).parent
MAX_LEN = 4096
STEPPED = "loom.run_bi_recurrent_and_retain"
FUSED = "loom.run_bi_lstm_and_retain"


class _Rawi(nn.Module):
    """The training notebooks' module shape: embedding 128, 2-layer BiLSTM 256, linear heads."""

    def __init__(self, n_chars: int, n_classes: int, heads: dict):
        super().__init__()
        self.embedding = nn.Embedding(n_chars, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)
        for name, width in heads.items():
            setattr(self, name, nn.Linear(512, n_classes if width is None else width))


def load_members() -> dict:
    vocab = json.loads(Path(hf_hub_download("TigreGotico/rawi-v2", "rawi_v2.vocab.json")).read_text("utf-8"))
    n_chars, n_classes = len(vocab["char_to_idx"]), len(vocab["diac_to_idx"])
    members = {}
    for key, repo, ckpt, heads in (
        ("v2", "TigreGotico/rawi-v2", "diacritization_model_lstm_2.pth", {"fc": None}),
        ("v3", "TigreGotico/rawi-v3", "diacritization_model_lstm_3.pth", {"presence_head": 1, "value_head": None}),
    ):
        sd = torch.load(hf_hub_download(repo, ckpt), map_location="cpu", weights_only=False)
        model = _Rawi(n_chars, n_classes, heads)
        model.load_state_dict(sd.get("model_state_dict", sd))
        members[key] = model.eval()
    return members


class _Embed(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.embedding = m.embedding

    def forward(self, tokens):
        return self.embedding(tokens)


class _Head(nn.Module):
    def __init__(self, fc):
        super().__init__()
        self.fc = fc

    def forward(self, lstm_out):
        return self.fc(lstm_out)


@dataclass(kw_only=True)
class RawiEnsembleExportConfig(BaseMultiPhaseModelExportConfig):
    architecture: Optional[str] = "rawi-ensemble"
    driver_script_path: Path = HERE
    _models: Optional[dict] = field(default=None, init=False, repr=False)

    def load_model(self):
        self._models = load_members()
        return self._models

    def export_architecture(self):
        return self.architecture

    def _heads(self):
        ms = self._models if self._models is not None else self.load_model()
        return (("v2", ms["v2"], ms["v2"].fc, "head", "v2_logits"),
                ("v3", ms["v3"], ms["v3"].value_head, "value", "v3_value"))

    def phases(self):
        t = 16
        out = []
        for p, m, head, hname, _ in self._heads():
            out += [
                ExportPhase(name=f"{p}_embed", wrapper=_Embed(m).eval(),
                            dummy_inputs=(torch.randint(2, 200, (1, t), dtype=torch.int32),),
                            mil_inputs=[ct.TensorType(name="tokens", dtype=np.int32,
                                                      shape=(1, ct.RangeDim(1, MAX_LEN)))]),
                RecurrentPhase(name=f"{p}_lstm", module=m.lstm, number_layers=True),
                ExportPhase(name=f"{p}_{hname}", wrapper=_Head(head).eval(), root_axis="n_tokens",
                            dummy_inputs=(torch.randn(1, t, 512),),
                            mil_inputs=[ct.TensorType(name="lstm_out", dtype=np.float32,
                                                      shape=(1, ct.RangeDim(1, MAX_LEN), 512))]),
            ]
        return out

    def driver_components(self):
        from loom_exporter.driver_components import (
            CALLER, BiRecurrentCall, DriverInputs, DriverReturn, LuaFragment, SubgraphCallComponent)
        from loom_exporter.driver_ir import Len, OutputRef, Var
        n = Len(Var("tokens"))
        comps = [DriverInputs(bindings=(("tokens", CALLER),), n_tokens=n)]
        for p, m, _, hname, outvar in self._heads():
            comps.append(SubgraphCallComponent(topology=f"{p}_embed", outputs=(), retain=True, length=n,
                                               inputs={"tokens": Var("tokens")}))
            prev = f"{p}_embed"
            for layer in range(m.lstm.num_layers):
                fwd = f"{p}_lstm_l{layer}_fwd"
                comps.append(BiRecurrentCall(
                    forward_topology=fwd, backward_topology=f"{p}_lstm_l{layer}_bwd",
                    out_var=f"{p}_lstm_{layer}_gen", sequence=OutputRef(prev), seq_len=n,
                    input_dim=m.lstm.input_size if layer == 0 else 2 * m.lstm.hidden_size,
                    hidden_dim=m.lstm.hidden_size))
                prev = fwd
            comps.append(SubgraphCallComponent(topology=f"{p}_{hname}", outputs=(outvar,), length=n,
                                               inputs={"lstm_out": OutputRef(prev)}))
        comps.append(LuaFragment(HERE / "10_gate.lua", reads=("v2_logits", "v3_value", "tokens"),
                                 defines=("gated_logits",)))
        comps.append(DriverReturn(values=("gated_logits",)))
        return comps


def use_fused_lstm(path: str) -> int:
    """Rewrite the driver's per-timestep BiLSTM calls to the one-graph binding (loom.cpp#144), for a
    loom-exporter that still emits the per-timestep call. Returns the number of calls rewritten."""
    import gguf

    reader = gguf.GGUFReader(path)
    field_ = reader.fields["model.driver_script"]
    script = bytes(field_.parts[field_.data[0]]).decode()
    n = script.count(STEPPED)
    if n == 0:
        return 0
    tmp = path + ".tmp"
    writer = gguf.GGUFWriter(tmp, arch=None)
    for key, f in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if f.types[0] != gguf.GGUFValueType.STRING:
            raise ValueError(f"{key}: only string metadata is expected in a loom GGUF")
        value = bytes(f.parts[f.data[0]]).decode()
        writer.add_string(key, value.replace(STEPPED, FUSED) if key == "model.driver_script" else value)
    for t in reader.tensors:
        writer.add_tensor(t.name, t.data, raw_shape=list(reversed([int(x) for x in t.shape])),
                          raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    Path(tmp).replace(path)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="output .gguf")
    ap.add_argument("--quantize", default="F16", help="F16, F32 or Q8_0_PC (Q8_0_PC breaks parity)")
    args = ap.parse_args()
    cfg = RawiEnsembleExportConfig(output_path=args.out)
    cfg.quantize = args.quantize
    print(cfg.export())
    print(f"{use_fused_lstm(args.out)} BiLSTM call(s) rewritten to {FUSED}")


if __name__ == "__main__":
    main()
