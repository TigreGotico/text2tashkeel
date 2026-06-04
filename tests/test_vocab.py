"""Lock down the vocabularies. A silent mismatch here would corrupt every
prediction, so these are the most important tests in the suite."""

import json
from pathlib import Path

from text2tashkeel._models import (
    _BILSTM_C2I,
    _BILSTM_ID2LABEL,
    _MODELS_DIR,
)

_BILSTM_SPECIAL = 4  # <PAD> <UNK> <BOS> <EOS>


def test_bilstm_vocab_size_and_range():
    # 4 special tokens + letters + punctuation = 54 distinct IDs.
    ids = set(_BILSTM_C2I.values())
    assert len(_BILSTM_C2I) == 50              # non-special entries
    assert min(ids) == _BILSTM_SPECIAL         # specials occupy 0..3
    assert max(ids) == 53
    assert len(ids) == len(_BILSTM_C2I)        # no collisions


def test_bilstm_label_map_is_15_classes():
    assert len(_BILSTM_ID2LABEL) == 15
    assert _BILSTM_ID2LABEL[0] == ""
    # classes 9..14 are shadda compounds: shadda (U+0651) leads.
    for i in range(9, 15):
        assert _BILSTM_ID2LABEL[i].startswith("ّ")


def test_rawi_vocab_loads_and_has_73_classes():
    v = json.loads((Path(_MODELS_DIR) / "rawi.vocab.json").read_text(encoding="utf-8"))
    assert "char_to_idx" in v and "diac_to_idx" in v
    assert len(dict(v["diac_to_idx"])) == 73


def test_libtashkeel_maps_load():
    m = json.loads((Path(_MODELS_DIR) / "libtashkeel.maps.json").read_text(encoding="utf-8"))
    assert set(m) == {"input", "hint", "target"}
    assert m["input"]["_"] == 0          # PAD symbol
    assert len(m["target"]) == 15
    assert len(m["hint"]) == 16          # target + standalone shadda


def test_all_model_files_present():
    for f in ("bilstm.onnx", "bilstm.int8.onnx", "rawi.onnx", "libtashkeel.onnx"):
        assert (Path(_MODELS_DIR) / f).exists(), f
