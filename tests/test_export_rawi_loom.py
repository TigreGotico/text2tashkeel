import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
pytest.importorskip("loom_exporter")
pytest.importorskip("coremltools")
pytest.importorskip("torch")

spec = importlib.util.spec_from_file_location(
    "export_rawi_loom", Path(__file__).parents[1] / "tools/loom/export_rawi_loom.py")
export_rawi_loom = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = export_rawi_loom  # dataclasses look the module up by name
spec.loader.exec_module(export_rawi_loom)


def _script(path):
    f = gguf.GGUFReader(str(path)).fields["model.driver_script"]
    return bytes(f.parts[f.data[0]]).decode()


def test_the_stepped_bilstm_calls_become_the_one_graph_binding(tmp_path):
    path = tmp_path / "m.gguf"
    w = gguf.GGUFWriter(str(path), arch=None)
    w.add_string("loom.architecture", "x")
    w.add_string("model.graph_topology.a", json.dumps({"version": 1}))
    w.add_string("model.driver_script", "loom.run_bi_recurrent_and_retain('a', 'b')\n" * 2)
    w.add_tensor("w", np.arange(6, dtype=np.float32).reshape(2, 3))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

    assert export_rawi_loom.use_fused_lstm(str(path)) == 2
    assert _script(path) == "loom.run_bi_lstm_and_retain('a', 'b')\n" * 2
    t = gguf.GGUFReader(str(path)).tensors[0]
    assert t.name == "w" and t.data.reshape(2, 3).tolist() == [[0, 1, 2], [3, 4, 5]]
    assert export_rawi_loom.use_fused_lstm(str(path)) == 0


def test_the_rawi_export_has_twelve_graphs_and_four_one_graph_bilstm_calls(tmp_path):
    out = tmp_path / "rawi.gguf"
    cfg = export_rawi_loom.RawiEnsembleExportConfig(output_path=str(out))
    cfg.quantize = "F32"
    cfg.export()
    export_rawi_loom.use_fused_lstm(str(out))
    graphs = sorted(k.split(".")[-1] for k in gguf.GGUFReader(str(out)).fields if k.startswith("model.graph_topology."))
    assert graphs == sorted(
        [f"{p}_embed" for p in ("v2", "v3")] + ["v2_head", "v3_value"]
        + [f"{p}_lstm_l{n}_{d}" for p in ("v2", "v3") for n in (0, 1) for d in ("fwd", "bwd")])
    script = _script(out)
    assert script.count("loom.run_bi_lstm_and_retain") == 4
    assert "loom.run_bi_recurrent_and_retain" not in script
    assert "gated_logits" in script
